# Generative Representation Learning for Scalable Latent Tokenization

[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg)](https://github.com/AlanBaade/repr)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2610.20533-b31b1b.svg)](https://arxiv.org/abs/2610.20533)&nbsp;
[![Blog](https://img.shields.io/badge/Blog-Read-3366cc.svg)](https://alanbaade.github.io/repr/)&nbsp;

## Codebase Setup

This codebase is a fork of  [JiT](https://github.com/LTH14/JiT/), which includes installation instructions.

The code mainly uses four files
* main_jit: Model parameter setup
* engine_jit: Epoch training setup
* denoiser_repr: Model setup and method calling
* model_repr: RePR model
  * forward_mar: 3 out of 4 training steps only perform MAR training
  * forward_repr: 1 out of 4 training steps optimize RePR and MAR. The RePR code has 5 unique input masks, and two axes for repeating. It is dense but readable.

## Model Training

### First Iteration

```bash
python -m torch.distributed.run --nproc_per_node=8 --standalone main_jit.py \
  --model REPR-B/16 --img_size 256 --in_context_len 16 --repr_dim 1024 \
  --denoiser_mode repr_mar --is_first_iter True \
  --batch_size 256 --batch_repeat 1 --batch_repeat_mar 1 --global_batch_repeat 8 \
  --blr 3e-5 --weight_decay 0.00 \
  --epochs 611 --warmup_epochs 20 --decay_epochs 10 --lr_schedule wsd \
  --diff_repeat 4 --diff_repeat_repr 32 \
  --p_repeat 16 --b_max_visible 16 --a_min_mask 192 \
  --mask_strategy_mar block --mask_strategy_repr block \
  --repr_loss_weight 0.005 --repr_delta 0.2 \
  --pad_dropout_start 0.5 --pad_dropout 0.01 --pad_dropout_decay_epochs 100 \
  --mu_mar -1.2 --std_mar 1.2 --mu_repr -1.2 --std_repr 1.2 \
  --dataset imagenet --data_path /path/to/imagenet-webdataset --train_aug \
  --output_dir ./output/first_iter --resume ./output/first_iter \
  --save_last_freq 5
```

### Second Iteration

```bash
python -m torch.distributed.run --nproc_per_node=4 --standalone main_jit.py \
  --model REPR-B/16 --img_size 256 --in_context_len 16 --repr_dim 1024 \
  --denoiser_mode repr_mar_featspred --is_first_iter False \
  --batch_size 400 --batch_repeat 1 --batch_repeat_mar 1 --global_batch_repeat 8 \
  --blr 3e-5 --weight_decay 0.00 \
  --epochs 211 --warmup_epochs 20 --decay_epochs 10 --lr_schedule wsd \
  --diff_repeat 4 --diff_mlp_depth 3 --diff_repeat_repr 32 \
  --p_repeat 16 --b_max_visible 16 --a_min_mask 192 \
  --mask_strategy_mar block --mask_strategy_repr block \
  --repr_loss_weight 0.005 --repr_delta 0.2 \
  --pad_dropout_start 0.5 --pad_dropout 0.01 --pad_dropout_decay_epochs 100 \
  --mu_mar -1.6 --std_mar 1.2 --mu_repr -1.6 --std_repr 1.2 \
  --frozen_model REPR-B/16 --frozen_is_first_iter True --teacher_feats_mul 0.66 \
  --frozen_checkpoint_model_path ./output/first_iter/checkpoint-last.pth \
  --dataset imagenet --data_path /path/to/imagenet-webdataset --train_aug \
  --output_dir ./output/second_iter --resume ./output/second_iter \
  --save_last_freq 5 --log_freq 41
```

## Demo
[Demo: Extracting Exact Categorical Probabilities From a Diffusion Model is Easy](https://colab.research.google.com/drive/1T90VK1sxLusFrOwiIoVSvTvlQZxV0qj2)


## Citation

```bibtex
@article{baade2026repr,
  title = {Generative Representation Learning for Scalable Latent Tokenization},
  author = {Baade, Alan and Sargent, Kyle and Sitzmann, Vincent and Chan, Eric Ryan and Johnson, Justin and Fei-Fei, Li and Wetzstein, Gordon},
  journal = {arXiv preprint arXiv:2610.20533},
  year = {2026}
}
```
