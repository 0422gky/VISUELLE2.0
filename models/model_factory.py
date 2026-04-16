from __future__ import annotations

from argparse import Namespace
from typing import Any

from models.FCN import FCN
from models.GTM import GTM


def build_model(
    *,
    model_type: str,
    embedding_dim: int,
    hidden_dim: int,
    output_dim: int,
    cat_dict: dict[Any, Any],
    col_dict: dict[Any, Any],
    fab_dict: dict[Any, Any],
    use_text: int,
    use_img: int,
    trend_len: int,
    num_trends: int,
    use_encoder_mask: int,
    gpu_num: int,
    use_trends: int = 1,
    num_attn_heads: int = 4,
    num_hidden_layers: int = 1,
    autoregressive: int = 0,
    use_hist_sales: int = 0,
):
    """
    Build FCN/GTM with a shared CLI-facing argument surface.

    Bridge mapping:
    - CLI `num_attn_heads` -> GTM `num_heads`
    - CLI `num_hidden_layers` -> GTM `num_layers`
    - CLI `use_hist_sales` only affects GTM (FCN ignores it)
    - CLI `use_trends` only affects FCN (GTM always encodes trends memory)
    """
    normalized_type = model_type.upper()
    if normalized_type == "FCN":
        return FCN(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=use_trends,
            use_text=use_text,
            use_img=use_img,
            trend_len=trend_len,
            num_trends=num_trends,
            use_encoder_mask=use_encoder_mask,
            gpu_num=gpu_num,
        )

    if normalized_type == "GTM":
        return GTM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_attn_heads,
            num_layers=num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=use_text,
            use_img=use_img,
            trend_len=trend_len,
            num_trends=num_trends,
            use_encoder_mask=use_encoder_mask,
            autoregressive=autoregressive,
            gpu_num=gpu_num,
            use_hist_sales=use_hist_sales,
        )

    raise ValueError(f"Unsupported --model_type: {model_type}. Expected 'GTM' or 'FCN'.")


def build_model_from_args(
    args: Namespace,
    *,
    cat_dict: dict[Any, Any],
    col_dict: dict[Any, Any],
    fab_dict: dict[Any, Any],
):
    """Build model directly from argparse namespace with stable CLI compatibility."""
    return build_model(
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        use_encoder_mask=args.use_encoder_mask,
        gpu_num=args.gpu_num,
        use_trends=args.use_trends,
        num_attn_heads=args.num_attn_heads,
        num_hidden_layers=args.num_hidden_layers,
        autoregressive=args.autoregressive,
        use_hist_sales=args.use_hist_sales,
    )
