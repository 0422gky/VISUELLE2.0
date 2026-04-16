import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import pipeline
from fairseq.optim.adafactor import Adafactor
from models.shared_modules import (
    DummyEmbedder,
    FusionNetwork,
    GTrendEmbedder,
    ImageEmbedder,
)
        
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        self.word_embedder = pipeline('feature-extraction', model='bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num

    def forward(self, category, color, fabric):
        textual_description = [self.col_dict[color.detach().cpu().numpy().tolist()[i]] + ' ' \
                + self.fab_dict[fabric.detach().cpu().numpy().tolist()[i]] + ' ' \
                + self.cat_dict[category.detach().cpu().numpy().tolist()[i]] for i in range(len(category))]


        # Use BERT to extract features
        word_embeddings = self.word_embedder(textual_description)

        # BERT gives us embeddings for [CLS] ..  [EOS], which is why we only average the embeddings in the range [1:-1] 
        # We're not fine tuning BERT and we don't want the noise coming from [CLS] or [EOS]
        word_embeddings = [torch.FloatTensor(x[1:-1]).mean(axis=0) for x in word_embeddings] 
        word_embeddings = torch.stack(word_embeddings).to('cuda:'+str(self.gpu_num))
        
        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings))

        return word_embeddings

class FCN(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, cat_dict, col_dict, fab_dict, \
        use_trends, use_text, use_img, trend_len, num_trends, use_encoder_mask=1, gpu_num=2):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.gpu_num = gpu_num
        self.use_trends = use_trends
        self.save_hyperparameters()

         # Encoder
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict, gpu_num)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_text)

        # Decoder
        decoder_in = hidden_dim + (use_trends*(trend_len*hidden_dim))
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*4, self.output_len)
        )

    def forward(self, category, color, fabric, temporal_features, gtrends, images):
        # Encode features and get inputs
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)

        # Decode
        if self.use_trends == 1:
            tgt = torch.cat([static_feature_fusion, gtrend_encoding.reshape(static_feature_fusion.shape[0], -1)], dim=-1)
        else:
            tgt = static_feature_fusion

        forecast = self.decoder(tgt)

        return forecast.view(-1, self.output_len)

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        item_sales, _recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = train_batch
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)
        forecasting_loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        loss = forecasting_loss
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales, _recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = val_batch
        forecasted_sales = self.forward(category, color, fabric, temporal_features, gtrends, images)
        
        return item_sales.squeeze(), forecasted_sales.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales, forecasted_sales = [x[0] for x in val_step_outputs], [x[1] for x in val_step_outputs]
        item_sales, forecasted_sales = torch.stack(item_sales), torch.stack(forecasted_sales)
        rescaled_item_sales, rescaled_forecasted_sales = item_sales*1065, forecasted_sales*1065 # 1065 is the normalization factor (max of the sales of the training set)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])