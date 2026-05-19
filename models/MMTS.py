import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Sales2WeekEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(2, embedding_dim)

    def forward(self, recent_sales_2w):
        return self.fc(recent_sales_2w)


class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
        super().__init__()
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
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

        return self.dropout(self.fc(torch.stack(embeddings, dim=0)))


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models

        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        return self.resnet(images)


class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.aux_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 5, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
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
        d, w, m, y, a = (
            temporal_features[:, 0].unsqueeze(1),
            temporal_features[:, 1].unsqueeze(1),
            temporal_features[:, 2].unsqueeze(1),
            temporal_features[:, 3].unsqueeze(1),
            temporal_features[:, 4].unsqueeze(1),
        )
        return self.dropout(
            self.dummy_fusion(
                torch.cat(
                    [
                        self.day_embedding(d),
                        self.week_embedding(w),
                        self.month_embedding(m),
                        self.year_embedding(y),
                        self.aux_embedding(a),
                    ],
                    dim=1,
                )
            )
        )


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        # MultiheadAttention in older torch expects (seq, batch, dim).
        q = query.unsqueeze(0)
        kv = context.unsqueeze(0)
        out, _ = self.attn(q, kv, kv)
        out = out.squeeze(0)
        return self.norm(query + self.dropout(out))


class MMTS(pl.LightningModule):
    """
    Multi-modal time-series model:
    sales2week TS query attends image -> text -> temporal, with TS-modal InfoNCE alignment.
    """

    VALID_FORECAST_MODES = {"direct_2_10", "rolling_2_1"}

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        cat_dict,
        col_dict,
        fab_dict,
        use_text=1,
        use_img=1,
        gpu_num=0,
        forecast_mode="direct_2_10",
        contrastive_loss_weight=0.1,
        contrastive_temperature=0.07,
        num_attn_heads=4,
        rescale_values=None,
    ):
        super().__init__()
        if forecast_mode not in self.VALID_FORECAST_MODES:
            raise ValueError(
                f"forecast_mode must be one of {sorted(self.VALID_FORECAST_MODES)}, got {forecast_mode}"
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_len = 10
        self.use_text = int(use_text)
        self.use_img = int(use_img)
        self.gpu_num = gpu_num
        self.forecast_mode = forecast_mode
        self.contrastive_loss_weight = float(contrastive_loss_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        if rescale_values is None:
            rescale_values = [1065.0]
        self.register_buffer(
            "rescale_values",
            torch.as_tensor(rescale_values, dtype=torch.float32).flatten(),
            persistent=False,
        )
        self.save_hyperparameters()

        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict, gpu_num)
        self.temporal_encoder = DummyEmbedder(embedding_dim)
        self.ts_encoder = Sales2WeekEmbedder(embedding_dim)

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_projection = ProjectionMLP(2048, embedding_dim)
        self.text_projection = ProjectionMLP(embedding_dim, embedding_dim)
        self.temporal_projection = ProjectionMLP(embedding_dim, embedding_dim)
        self.ts_projection = ProjectionMLP(embedding_dim, embedding_dim)

        self.attend_img = CrossAttentionBlock(embedding_dim, num_heads=num_attn_heads)
        self.attend_text = CrossAttentionBlock(embedding_dim, num_heads=num_attn_heads)
        self.attend_temporal = CrossAttentionBlock(embedding_dim, num_heads=num_attn_heads)

        self.direct_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.output_len),
        )
        self.step_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def _zero_like_context(self, ref):
        return torch.zeros_like(ref)

    def encode_modalities(self, category, color, fabric, temporal_features, images, recent_sales_2w):
        img_features = self.image_encoder(images)
        pooled_img = self.img_pool(img_features).flatten(1)
        z_img = self.img_projection(pooled_img)

        z_text = self.text_projection(self.text_encoder(category, color, fabric))
        z_temporal = self.temporal_projection(self.temporal_encoder(temporal_features))
        z_ts = self.ts_projection(self.ts_encoder(recent_sales_2w))

        if self.use_img != 1:
            z_img = self._zero_like_context(z_ts)
        if self.use_text != 1:
            z_text = self._zero_like_context(z_ts)
        return z_ts, z_img, z_text, z_temporal

    def fuse(self, z_ts, z_img, z_text, z_temporal):
        x = self.attend_img(z_ts, z_img)
        x = self.attend_text(x, z_text)
        x = self.attend_temporal(x, z_temporal)
        return x

    def _predict_from_history(self, category, color, fabric, temporal_features, images, recent_sales_2w):
        z_ts, z_img, z_text, z_temporal = self.encode_modalities(
            category, color, fabric, temporal_features, images, recent_sales_2w
        )
        fused = self.fuse(z_ts, z_img, z_text, z_temporal)
        return self.step_head(fused).squeeze(-1), (z_ts, z_img, z_text, z_temporal)

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
        if recent_sales_2w is None:
            raise ValueError("MMTS requires recent_sales_2w with shape (B, 2).")

        if self.forecast_mode == "rolling_2_1":
            hist = recent_sales_2w
            preds = []
            last_fused = None
            for _ in range(self.output_len):
                z_ts, z_img, z_text, z_temporal = self.encode_modalities(
                    category, color, fabric, temporal_features, images, hist
                )
                last_fused = self.fuse(z_ts, z_img, z_text, z_temporal)
                y_next = self.step_head(last_fused).squeeze(-1)
                preds.append(y_next)
                hist = torch.stack([hist[:, 1], y_next], dim=1)
            pred = torch.stack(preds, dim=1)
            if return_embedding:
                return pred, last_fused.unsqueeze(1)
            return pred

        z_ts, z_img, z_text, z_temporal = self.encode_modalities(
            category, color, fabric, temporal_features, images, recent_sales_2w
        )
        fused = self.fuse(z_ts, z_img, z_text, z_temporal)
        pred = self.direct_head(fused)
        if return_embedding:
            return pred, fused.unsqueeze(1)
        return pred

    def _bidirectional_infonce(self, a, b):
        if a.shape[0] <= 1:
            return torch.tensor(0.0, device=a.device, dtype=a.dtype)
        a = F.normalize(a, p=2, dim=-1, eps=1e-8)
        b = F.normalize(b, p=2, dim=-1, eps=1e-8)
        logits = (a @ b.T) / self.contrastive_temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def contrastive_loss(self, z_ts, z_img, z_text, z_temporal):
        losses = [
            self._bidirectional_infonce(z_ts, z_img),
            self._bidirectional_infonce(z_ts, z_text),
            self._bidirectional_infonce(z_ts, z_temporal),
        ]
        return sum(losses) / float(len(losses))

    def _direct_loss(self, item_sales, category, color, fabric, temporal_features, images, recent_sales_2w):
        z_ts, z_img, z_text, z_temporal = self.encode_modalities(
            category, color, fabric, temporal_features, images, recent_sales_2w
        )
        fused = self.fuse(z_ts, z_img, z_text, z_temporal)
        pred = self.direct_head(fused)
        target = item_sales[:, 2:12]
        pred_loss = F.mse_loss(target, pred)
        c_loss = self.contrastive_loss(z_ts, z_img, z_text, z_temporal)
        return pred, target, pred_loss, c_loss

    def _rolling_teacher_forcing_loss(
        self, item_sales, category, color, fabric, temporal_features, images
    ):
        preds = []
        pred_losses = []
        contrastive_losses = []
        for week_idx in range(2, 12):
            hist = item_sales[:, week_idx - 2 : week_idx]
            z_ts, z_img, z_text, z_temporal = self.encode_modalities(
                category, color, fabric, temporal_features, images, hist
            )
            fused = self.fuse(z_ts, z_img, z_text, z_temporal)
            y_next = self.step_head(fused).squeeze(-1)
            target = item_sales[:, week_idx]
            preds.append(y_next)
            pred_losses.append(F.mse_loss(target, y_next))
            contrastive_losses.append(self.contrastive_loss(z_ts, z_img, z_text, z_temporal))
        pred = torch.stack(preds, dim=1)
        target = item_sales[:, 2:12]
        pred_loss = sum(pred_losses) / float(len(pred_losses))
        c_loss = sum(contrastive_losses) / float(len(contrastive_losses))
        return pred, target, pred_loss, c_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def training_step(self, train_batch, batch_idx):
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, _gtrends, images = train_batch
        if self.forecast_mode == "rolling_2_1":
            pred, target, pred_loss, c_loss = self._rolling_teacher_forcing_loss(
                item_sales, category, color, fabric, temporal_features, images
            )
        else:
            pred, target, pred_loss, c_loss = self._direct_loss(
                item_sales, category, color, fabric, temporal_features, images, recent_sales_2w
            )
        loss = pred_loss + self.contrastive_loss_weight * c_loss
        self.log("train_loss", loss)
        self.log("train_pred_loss", pred_loss)
        self.log("train_contrastive_loss", c_loss)
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
        return item_sales[:, 2:12].squeeze(), pred.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.stack([x[0] for x in val_step_outputs])
        forecasted_sales = torch.stack([x[1] for x in val_step_outputs])
        scale = self.rescale_values.to(device=item_sales.device, dtype=item_sales.dtype)
        if scale.numel() == 1:
            scale = scale.view(1, 1)
        else:
            scale = scale[: item_sales.shape[-1]].view(1, -1)
        rescaled_item_sales = item_sales * scale
        rescaled_forecasted_sales = forecasted_sales * scale
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
