import math
import torch
import torch.nn as nn
from torchvision import models


class PositionalEncoding(nn.Module):
    # sin, cos positional encoding
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class Sales2WeekEmbedder(nn.Module):
    """两周销量 (B, 2) 线性映射到与图像/文本一致的 embedding_dim，供 FusionNetwork 拼接。"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.fc = nn.Linear(2, embedding_dim)

    def forward(self, recent_sales_2w: torch.Tensor) -> torch.Tensor:
        return self.fc(recent_sales_2w)


class FusionNetwork(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, use_img, use_text, use_hist_sales=0, dropout=0.2
    ):
        super(FusionNetwork, self).__init__()

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text
        self.use_hist_sales = use_hist_sales
        input_dim = (
            embedding_dim
            + (embedding_dim * use_img)
            + (embedding_dim * use_text)
            + (embedding_dim * use_hist_sales)
        )
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding, ts_emb=None):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))

        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(condensed_img)
        if self.use_text == 1:
            decoder_inputs.append(text_encoding)
        if self.use_hist_sales == 1:
            decoder_inputs.append(ts_emb)
        decoder_inputs.append(dummy_encoding)
        concat_features = torch.cat(decoder_inputs, dim=1)

        final = self.feature_fusion(concat_features)
        return final


class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i : i + split, i : i + split] = 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .to("cuda:" + str(self.gpu_num))
        )
        return mask

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .to("cuda:" + str(self.gpu_num))
        )
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0, 2, 1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1, 0, 2))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon)
        if self.use_mask == 1:
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)

        return out.view(*size).contiguous()  # batch_size, 2048, image_size/32, image_size/32


class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.aux_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 5, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal: day, week, month, year, aux (restock / extra / 0)
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
        d_emb, w_emb, m_emb, y_emb, a_emb = (
            self.day_embedding(d),
            self.week_embedding(w),
            self.month_embedding(m),
            self.year_embedding(y),
            self.aux_embedding(a),
        )
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb, a_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings
