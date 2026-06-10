# 没有对比学习的网络结构
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor


class ImageEncoder(nn.Module):
    """Frozen ResNet-50 image encoder, pooled to a 2048-d vector."""

    def __init__(self):
        super().__init__()
        from torchvision import models

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images):
        features = self.backbone(images)
        return self.pool(features).flatten(1)


class TextEncoder(nn.Module):
    """Frozen BERT text encoder, pooled to a 768-d attribute vector."""

    def __init__(self, cat_dict, col_dict, fab_dict):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False
        self._text_cache = {}

    def forward(self, category, color, fabric):
        device = category.device
        c_ids = category.detach().cpu().tolist()
        o_ids = color.detach().cpu().tolist()
        f_ids = fabric.detach().cpu().tolist()
        texts = [
            f"{self.col_dict[o_ids[i]]} {self.fab_dict[f_ids[i]]} {self.cat_dict[c_ids[i]]}"
            for i in range(len(c_ids))
        ]

        embeddings = [None] * len(texts)
        miss_idx, miss_texts = [], []
        for i, text in enumerate(texts):
            cached = self._text_cache.get(text)
            if cached is None:
                miss_idx.append(i)
                miss_texts.append(text)
            else:
                embeddings[i] = cached.to(device)

        if miss_texts:
            tok = self.tokenizer(miss_texts, padding=True, truncation=True, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            self.bert.to(device)
            with torch.no_grad():
                out = self.bert(**tok).last_hidden_state
            attn = tok["attention_mask"].unsqueeze(-1).float()
            valid = attn[:, 1:-1, :]
            token_emb = out[:, 1:-1, :]
            denom = valid.sum(dim=1).clamp(min=1.0)
            pooled = (token_emb * valid).sum(dim=1) / denom

            for j, i in enumerate(miss_idx):
                emb = pooled[j].detach()
                self._text_cache[miss_texts[j]] = emb.cpu()
                embeddings[i] = emb.to(device)

        return torch.stack(embeddings, dim=0)


class Simple(pl.LightningModule):
    """
    Multi-modal representation model without contrastive learning.

    image(2048), text(768), temporal(5), and recent_sales_2w(2) are each
    projected to embedding_dim, concatenated, then trained through:
        h = ReLU(W1 x_concat + b1)
        y_hat = W2 h + b2
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        cat_dict,
        col_dict,
        fab_dict,
        gpu_num=0,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_len = output_dim
        self.gpu_num = gpu_num
        self.save_hyperparameters()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder(cat_dict, col_dict, fab_dict)

        self.image_projection = nn.Linear(2048, embedding_dim)
        self.text_projection = nn.Linear(768, embedding_dim)
        self.temporal_projection = nn.Linear(5, embedding_dim)
        self.sales_projection = nn.Linear(2, embedding_dim)

        concat_dim = embedding_dim * 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.forecast_head = nn.Linear(hidden_dim, self.output_len)

    def _target_from_sales(self, item_sales):
        if item_sales.shape[1] == self.output_len:
            return item_sales
        if item_sales.shape[1] < self.output_len:
            raise ValueError(
                f"item_sales has {item_sales.shape[1]} weeks, but output_len={self.output_len}"
            )
        return item_sales[:, -self.output_len :]

    def encode_modalities(self, category, color, fabric, temporal_features, images, recent_sales_2w):
        if recent_sales_2w is None:
            raise ValueError("Simple requires recent_sales_2w with shape (B, 2).")
        if temporal_features.shape[1] == 4:
            temporal_features = torch.cat(
                [
                    temporal_features,
                    torch.zeros(
                        temporal_features.shape[0],
                        1,
                        device=temporal_features.device,
                        dtype=temporal_features.dtype,
                    ),
                ],
                dim=1,
            )
        if temporal_features.shape[1] < 5:
            raise ValueError("Simple requires temporal_features with at least 5 columns.")

        image_vec = self.image_encoder(images)
        text_vec = self.text_encoder(category, color, fabric)
        temporal_vec = temporal_features[:, :5]

        z_img = self.image_projection(image_vec)
        z_text = self.text_projection(text_vec)
        z_temporal = self.temporal_projection(temporal_vec)
        z_sales = self.sales_projection(recent_sales_2w)
        x_concat = torch.cat([z_img, z_text, z_temporal, z_sales], dim=1)
        return x_concat

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
        return_embedding=False,
        return_concat=False,
    ):
        del gtrends
        x_concat = self.encode_modalities(
            category,
            color,
            fabric,
            temporal_features,
            images,
            recent_sales_2w,
        )
        h = self.fusion_layer(x_concat)
        pred = self.forecast_head(h)

        if return_concat and return_embedding:
            return pred, h, x_concat
        if return_concat:
            return pred, x_concat
        if return_embedding:
            return pred, h
        return pred

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = train_batch
        pred = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            recent_sales_2w=recent_sales_2w,
        )
        target = self._target_from_sales(item_sales)
        loss = F.mse_loss(target, pred)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = val_batch
        pred = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            recent_sales_2w=recent_sales_2w,
        )
        target = self._target_from_sales(item_sales)
        return target.squeeze(), pred.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.stack([x[0] for x in val_step_outputs])
        forecasted_sales = torch.stack([x[1] for x in val_step_outputs])
        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        self.log("val_loss", loss)
        self.log("val_mae", mae)
        print("Validation MAE:", mae.detach().cpu().numpy(), "LR:", self.optimizers().param_groups[0]["lr"])

