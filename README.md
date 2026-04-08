# GTM-Transformer
Official Pytorch Implementation of [**Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends**](https://arxiv.org/abs/2109.09824) paper

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/well-googled-is-half-done-multimodal/new-product-sales-forecasting-on-visuelle)](https://paperswithcode.com/sota/new-product-sales-forecasting-on-visuelle?p=well-googled-is-half-done-multimodal)

## env settings
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 关闭wandb
# 设置pip为镜像源
tmux attach -t my_training_session
```

## 使用了全部`train`的ckpt
```
/data/coding/tmp_GTM_transformer/log/GTM/GTM_Run1---epoch=104---02-04-2026-21-00-31.ckpt
```

## Installation

We suggest the use of VirtualEnv.

```bash

python3 -m venv gtm_venv
source gtm_venv/bin/activate
# gtm_venv\Scripts\activate.bat # If you're running on Windows

pip install numpy pandas matplotlib opencv-python permetrics Pillow scikit-image scikit-learn scipy tqdm transformers fairseq wandb

pip install torch torchvision

# For CUDA11.1 (NVIDIA 3K Serie GPUs)
# Check official pytorch installation guidelines for your system
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install pytorch-lightning

export INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/HumaticsLAB/GTM-Transformer.git
cd GTM-Transformer
mkdir ckpt
mkdir dataset
mkdir results

unset INSTALL_DIR
```

## Dataset

**VISUELLE** dataset is publicly available to download [here](https://forms.gle/cVGQAmxhHf7eRJ937). Please download and extract it inside the dataset folder.

## Training
To train the model of GTM-Transformer please use the following scripts. Please check the arguments inside the script before launch.

```bash
python train.py --data_folder dataset
```

```bash
python train.py --data_folder "visuelle2/" --gpu_num 0 --model_type GTM --train_frac 1
```

## Inference
To evaluate the model of GTM-Transformer please use the following script .Please check the arguments inside the script before launch.

```bash
python forecast.py --data_folder dataset --ckpt_path ckpt/model.pth

python forecast.py --data_folder "visuelle2/" --ckpt_path "log/GTM/GTM_Run1---epoch=29---25-03-2026-13-17-24.ckpt"


python forecast_csv.py --data_folder "visuelle2/" --ckpt_path "log/GTM/GTM_Run1---epoch=169---27-03-2026-11-28-07.ckpt" --output_dim 12 --gpu_num 0 --output_csv results/my_forecast.csv


python forecast_csv.py --data_folder "visuelle2/" --ckpt_path "log/GTM/GTM_Run1---epoch=44---31-03-2026-13-15-01.ckpt" --output_dim 12 --gpu_num 0 --output_csv results/my_forecast.csv

python forecast_csv.py --data_folder "visuelle2/" --ckpt_path "log/GTM/GTM_Run1---epoch=104---02-04-2026-21-00-31.ckpt" --output_dim 12 --gpu_num 0 --output_csv results/my_forecast.csv

cd /data/coding/tmp_GTM_transformer
python similarity_wape_pipeline.py \
  --train_csv outputs/train_item_embeddings.csv \
  --test_csv outputs/test_item_embeddings.csv \
  --top_k 20 \
  --start_week 2 \
  --save_prefix results/sim_wape \
  --compare_topk 1,5,20
```

## Export Item Embedddings
export：test
```bash
python export_item_embeddings.py --checkpoint "path/to/your.ckpt" --data_folder "dataset/" --split test --output_dir "outputs/"
```

export: train
```bash
python export_item_embeddings.py --checkpoint "path/to/your.ckpt" --data_folder "dataset/" --split train --output_dir "outputs/"
```

export: all
```bash
python export_item_embeddings.py --checkpoint "path/to/your.ckpt" --data_folder "dataset/" --split all --output_dir "outputs/"
```

```bash
# example of ckpt path
tmp_GTM_transformer/log/GTM/GTM_Run1---epoch=29---25-03-2026-13-17-24.ckpt

python export_item_embeddings.py --checkpoint "log/GTM/GTM_Run1---epoch=29---25-03-2026-13-17-24.ckpt" --data_folder "visuelle2/" --split all --output_dir "outputs/"

python export_item_embeddings.py --checkpoint "log/GTM/GTM_Run1---epoch=104---02-04-2026-21-00-31.ckpt" --data_folder "visuelle2/" --split all --output_dir "outputs/"
```

## Citation
```
@misc{skenderi2021googled,
      title={Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends}, 
      author={Geri Skenderi and Christian Joppi and Matteo Denitto and Marco Cristani},
      year={2021},
      eprint={2109.09824},
}
```

## 对比学习 metric learning projector
```bash
# 1) 在 GTM-Transformer 目录，用已导出的 train 向量训练（曲线与 npy 行对齐，如 train.csv 或带 sales_wk_* 的导出表）
python train_curve_projector.py --train_embeddings_npy outputs/train_item_embeddings.npy --train_curves_csv outputs/train_item_embeddings.parquet --output_dir results/curve_projector --epochs 20 --pca_components 0 

# 2) 生成投影后的 train/test npy
python apply_curve_projector.py --projector_dir results/curve_projector --train_embeddings_npy outputs/train_item_embeddings.npy  --test_embeddings_npy outputs/test_item_embeddings.npy --output_dir results/curve_projector/projected

# 3) WAPE（ GTM-Transformer 下用 run_similarity_wape_with_projected.py） 这个是使用了对比学习的metric
python similarity_wape_pipeline.py --train_csv outputs/train_item_embeddings.parquet --test_csv outputs/test_item_embeddings.parquet   --train_emb_npy results/curve_projector/projected/train_item_embeddings_projected.npy   --test_emb_npy results/curve_projector/projected/test_item_embeddings_projected.npy --save_prefix results/curve_projector/WAPE_results


# 不使用对比学习trick跑的 WAPE
python similarity_wape_pipeline.py --train_csv outputs/train_item_embeddings.parquet --test_csv outputs/test_item_embeddings.parquet   --train_emb_npy outputs/train_item_embeddings.npy   --test_emb_npy outputs/test_item_embeddings.npy --save_prefix results/curve_nonprojected/WAPE_results
```


## 带有PCA的projector

```bash
python train_curve_projector.py \
  --train_embeddings_npy outputs/train_item_embeddings.npy \
  --train_curves_csv outputs/train_item_embeddings.parquet \
  --output_dir results/curve_projector_pca \
  --pca_components 32 \
  --epochs 20


python apply_curve_projector.py \
  --projector_dir results/curve_projector_pca \
  --train_embeddings_npy outputs/train_item_embeddings.npy \
  --test_embeddings_npy outputs/test_item_embeddings.npy \
  --output_dir results/curve_projector_pca/projected \
  --device cuda

python similarity_wape_pipeline.py --train_csv outputs/train_item_embeddings.parquet --test_csv outputs/test_item_embeddings.parquet   --train_emb_npy results/curve_projector_pca/projected/train_item_embeddings_projected.npy   --test_emb_npy results/curve_projector_pca/projected/test_item_embeddings_projected.npy --save_prefix results/curve_projector_pca/WAPE_results

```