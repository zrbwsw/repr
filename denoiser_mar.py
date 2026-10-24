import torch
import torch.nn as nn
from model_repr_mar_2 import REPR_models as REPR_models_new


class DenoiserMAR(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.denoiser_mode == "repr_noiseaug_mar4":
            args.find_unused_parameters = True

        self.net = REPR_models_new[args.model](
            input_size=args.img_size,
            in_channels=3,
            in_context_len=args.in_context_len,
            repr_dim=args.repr_dim,
            diff_mlp_depth=args.diff_mlp_depth,
            batch_repeat=args.batch_repeat * args.global_batch_repeat,
            diff_repeat=args.diff_repeat,
            d_num_p=args.d_num_p,
            diff_repeat_repr=args.diff_repeat_repr,
            d_max=args.d_max,
            mask_m=args.mask_m,
            mask_strategy_mar=args.mask_strategy_mar,
            mask_strategy_repr=args.mask_strategy_repr,
            repr_loss_weight=args.repr_loss_weight,
            repr_delta=args.repr_delta,
            pad_dropout=args.pad_dropout,
            repr_noise_bins=args.repr_noise_bins,
            mu_repr=args.mu_repr,
            std_repr=args.std_repr,
            mu_mar=args.mu_mar,
            std_mar=args.std_mar,
            batch_repeat_mar=args.batch_repeat_mar * args.global_batch_repeat,
            is_first_iter=args.is_first_iter,
            is_repr_encoderdecoder=args.is_repr_encoderdecoder,
        )

        if args.denoiser_mode == "repr_mar4_seconditer":
            self.frozen_net = REPR_models_new[args.frozen_model](
                input_size=args.img_size,
                in_channels=3,
                in_context_len=16,
                repr_dim=args.repr_dim,
                diff_mlp_depth=args.diff_mlp_depth,
                batch_repeat=args.batch_repeat,
                diff_repeat=args.diff_repeat,
                d_num_p=args.d_num_p,
                diff_repeat_repr=args.diff_repeat_repr,
                d_max=args.d_max,
                mask_m=args.mask_m,
                mask_strategy_mar=args.mask_strategy_mar,
                mask_strategy_repr=args.mask_strategy_repr,
                repr_loss_weight=args.repr_loss_weight,
                repr_delta=args.repr_delta,
                pad_dropout=args.pad_dropout,
                repr_noise_bins=args.repr_noise_bins,
                mu_repr=args.mu_repr,
                std_repr=args.std_repr,
                mu_mar=args.mu_mar,
                std_mar=args.std_mar,
                batch_repeat_mar=args.batch_repeat_mar,
                is_first_iter=args.frozen_is_first_iter, # Frozen only parameter
                is_repr_encoderdecoder=True,
            )
            self.frozen_net.requires_grad_(False)

        self.args = args
        self.denoiser_mode = args.denoiser_mode
        self.batch_repeat = args.batch_repeat
        self.batch_repeat_mar = args.batch_repeat_mar
        self.register_buffer("train_step", torch.zeros((), dtype=torch.long))
        self.teacher_feats_mul = args.teacher_feats_mul

    def get_optimizer_param_groups(self, weight_decay, lr, effective_batch_size):
        return self.net.get_mup_param_groups(weight_decay)

    def forward_mar(self, x):
        return self.net.forward_mar_only(x), {}

    @torch.no_grad()
    def get_stats(self, x, labels=None):
        self.net.eval()
        spatial, global_features = self.net.forward_reprv2_4_feats(x)
        spatial = spatial.reshape(x.size(0), self.net.pad_hw, self.net.pad_hw, self.net.repr_dim)[:, self.net.factor_pad:self.net.factor_pad+self.net.hw_len, self.net.factor_pad:self.net.factor_pad+self.net.hw_len].reshape(-1, self.net.repr_dim).float()
        features = {"spatial": spatial, "global": global_features.reshape(-1, self.net.repr_dim).float()}
        return {f"{name}_{stat}": getattr(feats, stat)(0) for name, feats in features.items() for stat in ("mean", "std")}

    def forward_repr_noiseaug_mar4(self, x, labels=None):
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            return self.net.forward_repr_22(x)
        else:
            loss = self.net.forward_mar_only(x)
            return loss, {'mse_loss': loss.item()}

    def forward_repr_noiseaug_mar4_second_iter(self, x, labels=None):
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            with torch.no_grad():
                self.frozen_net.eval()
                x_target, _ = self.frozen_net.forward_reprv2_4_feats(x, return_padded=False)
            return self.net.forward_repr_22(x, x_target * self.teacher_feats_mul)
        else:
            x = x[:x.size(0)//self.batch_repeat_mar]
            with torch.no_grad():
                self.frozen_net.eval()
                x_target, _ = self.frozen_net.forward_reprv2_4_feats(x, return_padded=False)
            loss = self.net.forward_mar_only(x, x_target * self.teacher_feats_mul)
            return loss, {'mse_loss': loss.item()}

    def set_epoch(self, epoch):
        end = self.args.pad_dropout
        start = self.args.pad_dropout_start
        if start is None:
            start = end
        decay_epochs = self.args.pad_dropout_decay_epochs
        if start <= end or epoch <= 0:
            pad_dropout = start
        elif decay_epochs <= 0 or epoch >= decay_epochs:
            pad_dropout = end
        else:
            pad_dropout = start + (end - start) * epoch / decay_epochs
        self.net.pad_dropout = pad_dropout

    def forward(self, x, labels):
        if self.denoiser_mode == "mar":
            return self.forward_mar(x)
        elif self.denoiser_mode == "repr_noiseaug_mar4":
            return self.forward_repr_noiseaug_mar4(x)
        elif self.denoiser_mode == "repr_mar4_seconditer":
            return self.forward_repr_noiseaug_mar4_second_iter(x)
        raise ValueError(f"Unknown training mode: {self.denoiser_mode}")

    def update_ema(self):
        pass
