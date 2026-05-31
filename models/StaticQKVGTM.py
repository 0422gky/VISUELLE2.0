import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor

from models.GTM import (
    DummyEmbedder,
    FusionNetwork,
    ImageEmbedder,
    Sales2WeekEmbedder,
    TextEmbedder,
)


class StaticQKVGTM(pl.LightningModule):
    """
    GTM-style static-fusion forecaster.

    Difference from GTM:
    - GTM uses static_feature_fusion as decoder query and Google Trends encoding as key/value.
    - StaticQKVGTM generates Q/K/V horizon tokens all from static_feature_fusion.
    - Training adds a CLIP-style InfoNCE alignment between fused static semantics and
      image/text semantics.
    """

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
        gpu_num,
        use_hist_sales=0,
        contrastive_loss_weight=0.1,
        contrastive_temperature=0.07,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_len = output_dim
        self.use_text = int(use_text)
        self.use_img = int(use_img)
        self.use_hist_sales = int(use_hist_sales)
        self.gpu_num = gpu_num
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.save_hyperparameters()

        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict, gpu_num)
        self.sales_2w_encoder = Sales2WeekEmbedder(embedding_dim) if self.use_hist_sales else None
        self.static_feature_encoder = FusionNetwork(
            embedding_dim,
            hidden_dim,
            self.use_img,
            self.use_text,
            use_hist_sales=self.use_hist_sales,
        )

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_semantic = nn.Linear(2048, hidden_dim)
        self.text_semantic = nn.Linear(embedding_dim, hidden_dim)
        self.fused_semantic = nn.Linear(hidden_dim, hidden_dim)

        qkv_dim = output_dim * hidden_dim
        self.q_proj = nn.Linear(hidden_dim, qkv_dim)
        self.k_proj = nn.Linear(hidden_dim, qkv_dim)
        self.v_proj = nn.Linear(hidden_dim, qkv_dim)

        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
                for _ in range(num_layers)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
        self.decoder_fc = nn.Linear(hidden_dim, 1)

    def _target_from_sales(self, item_sales):
        if item_sales.shape[1] == self.output_len:
            return item_sales
        if item_sales.shape[1] < self.output_len:
            raise ValueError(
                f"item_sales has {item_sales.shape[1]} weeks, but output_len={self.output_len}"
            )
        return item_sales[:, -self.output_len :]

    def _encode_static(self, category, color, fabric, temporal_features, images, recent_sales_2w):
        img_encoding = self.image_encoder(images)
        pooled_img = self.img_pool(img_encoding).flatten(1)
        text_encoding = self.text_encoder(category, color, fabric)
        dummy_encoding = self.dummy_encoder(temporal_features)

        ts_emb = None
        if self.use_hist_sales and self.sales_2w_encoder is not None:
            if recent_sales_2w is None:
                raise ValueError("use_hist_sales=1 requires recent_sales_2w with shape (B, 2).")
            ts_emb = self.sales_2w_encoder(recent_sales_2w)

        static_feature = self.static_feature_encoder(
            img_encoding,
            text_encoding,
            dummy_encoding,
            ts_emb=ts_emb,
        )
        return static_feature, pooled_img, text_encoding

    def _static_qkv_tokens(self, static_feature):
        bsz = static_feature.shape[0]
        q = self.q_proj(static_feature).view(bsz, self.output_len, self.hidden_dim).transpose(0, 1)
        k = self.k_proj(static_feature).view(bsz, self.output_len, self.hidden_dim).transpose(0, 1)
        v = self.v_proj(static_feature).view(bsz, self.output_len, self.hidden_dim).transpose(0, 1)
        x = q
        for attn, norm, ffn, ffn_norm in zip(
            self.attn_layers,
            self.norm_layers,
            self.ffn_layers,
            self.ffn_norm_layers,
        ):
            attn_out, _ = attn(x, k, v)
            x = norm(x + self.dropout(attn_out))
            x = ffn_norm(x + self.dropout(ffn(x)))
        return x

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
    ):
        del gtrends  # StaticQKVGTM intentionally does not use trend memory as K/V.
        static_feature, _pooled_img, _text_encoding = self._encode_static(
            category,
            color,
            fabric,
            temporal_features,
            images,
            recent_sales_2w,
        )
        tokens = self._static_qkv_tokens(static_feature)
        pred = self.decoder_fc(tokens).squeeze(-1).transpose(0, 1)
        if return_embedding:
            return pred, tokens.transpose(0, 1)
        return pred

    def _bidirectional_infonce(self, a, b):
        if a.shape[0] <= 1:
            return torch.tensor(0.0, device=a.device, dtype=a.dtype)
        a = F.normalize(a, p=2, dim=-1, eps=1e-8)
        b = F.normalize(b, p=2, dim=-1, eps=1e-8)
        logits = (a @ b.T) / self.contrastive_temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def _clip_semantic_loss(self, static_feature, pooled_img, text_encoding):
        fused_z = self.fused_semantic(static_feature)
        losses = []
        if self.use_img == 1:
            losses.append(self._bidirectional_infonce(fused_z, self.image_semantic(pooled_img)))
        if self.use_text == 1:
            losses.append(self._bidirectional_infonce(fused_z, self.text_semantic(text_encoding)))
        if not losses:
            return torch.tensor(0.0, device=static_feature.device, dtype=static_feature.dtype)
        return sum(losses) / float(len(losses))

    def _loss_parts(self, batch):
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = batch
        del gtrends
        static_feature, pooled_img, text_encoding = self._encode_static(
            category,
            color,
            fabric,
            temporal_features,
            images,
            recent_sales_2w,
        )
        tokens = self._static_qkv_tokens(static_feature)
        pred = self.decoder_fc(tokens).squeeze(-1).transpose(0, 1)
        target = self._target_from_sales(item_sales)
        pred_loss = F.mse_loss(target, pred)
        clip_loss = self._clip_semantic_loss(static_feature, pooled_img, text_encoding)
        return pred, target, pred_loss, clip_loss

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
        _pred, _target, pred_loss, clip_loss = self._loss_parts(train_batch)
        loss = pred_loss + self.contrastive_loss_weight * clip_loss
        self.log("train_loss", loss)
        self.log("train_pred_loss", pred_loss)
        self.log("train_clip_loss", clip_loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pred, target, pred_loss, clip_loss = self._loss_parts(val_batch)
        self.log("val_pred_loss_step", pred_loss)
        self.log("val_clip_loss_step", clip_loss)
        return target.squeeze(), pred.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.stack([x[0] for x in val_step_outputs])
        forecasted_sales = torch.stack([x[1] for x in val_step_outputs])
        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        self.log("val_mae", mae)
        self.log("val_loss", loss)
        print(
            "Validation MAE:",
            mae.detach().cpu().numpy(),
            "LR:",
            self.optimizers().param_groups[0]["lr"],
        )
