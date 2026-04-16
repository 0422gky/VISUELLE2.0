import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from fairseq.optim.adafactor import Adafactor
from models.shared_modules import (
    DummyEmbedder,
    FusionNetwork,
    GTrendEmbedder,
    ImageEmbedder,
    PositionalEncoding,
    Sales2WeekEmbedder,
    TimeDistributed,
)
        
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num
        self._text_cache = {}

    def forward(self, category, color, fabric):
        device = category.device
        c_ids = category.detach().cpu().tolist()
        o_ids = color.detach().cpu().tolist()
        f_ids = fabric.detach().cpu().tolist()
        textual_description = [
            f"{self.col_dict[o_ids[i]]} {self.fab_dict[f_ids[i]]} {self.cat_dict[c_ids[i]]}"
            for i in range(len(c_ids))
        ]

        # 先命中缓存，只对未命中样本走一次 BERT 批量编码。
        word_embeddings = [None] * len(textual_description)
        miss_idx, miss_texts = [], []
        for i, text in enumerate(textual_description):
            cached = self._text_cache.get(text)
            if cached is None:
                miss_idx.append(i)
                miss_texts.append(text)
            else:
                word_embeddings[i] = cached.to(device)

        if miss_texts:
            tok = self.tokenizer(
                miss_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            self.bert.to(device)
            with torch.no_grad():
                out = self.bert(**tok).last_hidden_state  # (B, L, 768)
            attn = tok["attention_mask"].unsqueeze(-1).float()
            valid = attn[:, 1:-1, :]
            token_emb = out[:, 1:-1, :]
            denom = valid.sum(dim=1).clamp(min=1.0)
            pooled = (token_emb * valid).sum(dim=1) / denom  # (B, 768)

            for j, i in enumerate(miss_idx):
                emb = pooled[j].detach()
                self._text_cache[miss_texts[j]] = emb.cpu()
                word_embeddings[i] = emb.to(device)

        word_embeddings = torch.stack(word_embeddings, dim=0)
        
        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings))

        return word_embeddings

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # PyTorch's nn.TransformerDecoder expects decoder layers to expose `self_attn`
        # (used only for internal shape bookkeeping). We reuse the same MultiheadAttention
        # module to satisfy that interface without changing our cross-attention forward.
        self.self_attn = self.multihead_attn

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        # Newer torch versions may pass these args from nn.TransformerDecoder
        tgt_is_causal=None,
        memory_is_causal=None,
        **kwargs,
    ):

        # Cross-attention: use `memory` as both key/value.
        # Note: `nn.TransformerDecoder` expects each decoder layer to return a tensor,
        # so we drop returning `attn_weights` to keep compatibility across torch versions.
        tgt2, _ = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class GTM(pl.LightningModule):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers,
        use_text,
        use_img,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        num_trends,
        gpu_num,
        use_encoder_mask=1,
        autoregressive=False,
        use_hist_sales=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.use_hist_sales = use_hist_sales
        self.gpu_num = gpu_num
        self.save_hyperparameters()

         # Encoder
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict, gpu_num)
        self.gtrend_encoder = GTrendEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.sales_2w_encoder = Sales2WeekEmbedder(embedding_dim) if use_hist_sales else None
        self.static_feature_encoder = FusionNetwork(
            embedding_dim, hidden_dim, use_img, use_text, use_hist_sales=use_hist_sales
        )

        # Decoder
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, \
                                                dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        
        if self.autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not self.autoregressive else 1),
            nn.Dropout(0.2)
        )
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(
        self,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        *,
        recent_sales_2w=None,
        return_embedding: bool = False,
    ):
        # Encode features and get inputs
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        ts_emb = None
        if self.use_hist_sales and self.sales_2w_encoder is not None:
            if recent_sales_2w is None:
                raise ValueError("use_hist_sales=1 时需要传入 recent_sales_2w (B, 2)。")
            ts_emb = self.sales_2w_encoder(recent_sales_2w)

        static_feature_fusion = self.static_feature_encoder(
            img_encoding, text_encoding, dummy_encoding, ts_emb=ts_emb
        )

        if self.autoregressive == 1:
            # Decode
            tgt = torch.zeros(self.output_len, gtrend_encoding.shape[1], gtrend_encoding.shape[-1]).to('cuda:'+str(self.gpu_num))
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            memory = gtrend_encoding
            decoder_ret = self.decoder(tgt, memory, tgt_mask)
            # Some PyTorch versions may return (output, attn_weights); handle both.
            decoder_out = decoder_ret[0] if isinstance(decoder_ret, tuple) else decoder_ret
            forecast = self.decoder_fc(decoder_out)
        else:
            # Decode (generatively/non-autoregressively)
            tgt = static_feature_fusion.unsqueeze(0)
            memory = gtrend_encoding
            decoder_ret = self.decoder(tgt, memory)
            decoder_out = decoder_ret[0] if isinstance(decoder_ret, tuple) else decoder_ret
            forecast = self.decoder_fc(decoder_out)

        pred = forecast.view(-1, self.output_len)
        if return_embedding:
            # Cross-attention output / multi-modal fused representation.
            # decoder_out is (T, B, D) -> (B, T, D)
            fused_feature = decoder_out.transpose(0, 1)
            return pred, fused_feature
        return pred

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(),scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    
        return [optimizer]


    def training_step(self, train_batch, batch_idx):
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = train_batch
        forecasted_sales = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            recent_sales_2w=recent_sales_2w,
        )
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log('train_loss', loss)

        return loss

    def validation_step(self, test_batch, batch_idx):
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = test_batch
        forecasted_sales = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            recent_sales_2w=recent_sales_2w,
        )
        
        return item_sales.squeeze(), forecasted_sales.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales, forecasted_sales = [x[0] for x in val_step_outputs], [x[1] for x in val_step_outputs]
        item_sales, forecasted_sales = torch.stack(item_sales), torch.stack(forecasted_sales)
        rescaled_item_sales, rescaled_forecasted_sales = item_sales*1065, forecasted_sales*1065 # 1065 is the normalization factor (max of the sales of the training set)
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        self.log('val_mae', mae)
        self.log('val_loss', loss)

        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])
