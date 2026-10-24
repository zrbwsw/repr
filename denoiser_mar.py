import copy
import torch
import torch.distributed as dist
from model_repr_mar_2 import REPR_models as REPR_models_new
from model_repr_mar_3 import REPR_models as REPR_Models_MultiIter
from model_repr_mar import REPR_models
from model_repr_mar_old import REPR_models as REPR_models_old
from slop.feature_whitening import FullWhiteningPair
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributed.nn.functional import all_gather
from einops import rearrange

class DenoiserMAR(nn.Module):
    def __init__(
        self,
        args,
        mode='v2'
    ):
        super().__init__()
        repr_version = getattr(args, "repr_version", getattr(args, "version", "8"))
        if args.denoiser_mode == "mar_in_ft":
            repr_version = "mar_in_ft"
        if args.denoiser_mode == "mar_frozen_feats_imagenet_labels":
            repr_version = "10"
        if args.denoiser_mode in ("repr_noiseaug_mar4", "repr_noiseaug_mar4_crossview", "contrastive_probe", "maskednoised_selfdistill", "get_stats"):
            repr_version = "22"
        if args.denoiser_mode in ("repr_mar4_seconditer",):
            repr_version = "22"; assert not args.is_first_iter
        if args.denoiser_mode == "imagenet_dual_probe" and repr_version != "23":
            repr_version = "22"
        if args.denoiser_mode == "repr_noiseaug_mar4_multiiter":
            repr_version = "23"
        if args.denoiser_mode in ("repr_noiseaug_mar4", "repr_noiseaug_mar4_crossview", "repr_noiseaug_mar4_multiiter", "repr_finetune"):
            args.find_unused_parameters = True
        if repr_version == "22":
            mode = "v3"
        if repr_version == "23":
            mode = "v4"

        self.net = {'v1': REPR_models_old, 'v2':REPR_models, 'v3': REPR_models_new, 'v4': REPR_Models_MultiIter}[mode][args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            condition_on_class=args.condition_on_class,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            in_context_len=args.in_context_len,
            repr_dim=args.repr_dim,
            num_patch_heads=args.num_patch_heads,
            diff_mlp_width=args.diff_mlp_width,
            diff_mlp_depth=args.diff_mlp_depth,
            batch_repeat=args.batch_repeat * args.global_batch_repeat,
            diff_repeat=args.diff_repeat,
            d_num_p=args.d_num_p,
            diff_repeat_repr=args.diff_repeat_repr,
            d_max=args.d_max,
            mask_m=args.mask_m,
            pmi_target_mul=args.pmi_target_mul,
            source_time_repeat=args.source_time_repeat,
            num_repr_heads=args.num_repr_heads,
            masked_selfdistill_mask_ratio=args.masked_selfdistill_mask_ratio,
            masked_selfdistill_loss_weight=args.masked_selfdistill_loss_weight,
            repr_loss_weight=args.repr_loss_weight,
            repr_delta=args.repr_delta,
            repr_loss_weight_iter2=args.repr_loss_weight_iter2,
            imagenet_loss_weight=args.imagenet_loss_weight,
            pad_dropout=args.pad_dropout,
            repr_noise_bins=args.repr_noise_bins,
            is_raw_pmi=getattr(args, "is_raw_pmi", False),
            add_input_noise=args.add_input_noise,
            mu_repr=args.mu_repr,
            std_repr=args.std_repr,
            mu_mar=args.mu_mar,
            std_mar=args.std_mar,
            version=repr_version,
            mask_strategy_mar=args.mask_strategy_mar,
            mask_strategy_repr=args.mask_strategy_repr,
            do_simclr_augs=args.do_simclr_augs,
            noise_a=args.noise_a,
            batch_repeat_mar=args.batch_repeat_mar * args.global_batch_repeat,
            mu_repr_2=args.mu_repr_2,
            std_repr_2=args.std_repr_2,
            mu_mar_2=args.mu_mar_2,
            std_mar_2=args.std_mar_2,
            is_first_iter=args.is_first_iter,
            is_rope_bug_fixed=args.is_rope_bug_fixed,
            is_repr_encoderdecoder=args.is_repr_encoderdecoder,
            diff_mlp_swiglu=args.diff_mlp_swiglu,
        )
        if args.denoiser_mode == "repr_finetune":
            self.frozen_mar = copy.deepcopy(self.net)
            self.frozen_mar.requires_grad_(False)

        self.img_size = args.img_size
        self.num_classes = args.class_num

        if args.denoiser_mode == "imagenet_eval":
            self.imagenet_eval_dim = args.repr_dim
            self.net.requires_grad_(False)
            self.imagenet_eval_head = nn.Linear(self.imagenet_eval_dim, args.class_num)

        self.dual_probe_whitening = None
        if args.denoiser_mode == "imagenet_dual_probe":
            self.net.requires_grad_(False)
            if args.dino_probe_features:
                print("WARNING: imagenet_dual_probe is using frozen DINOv2 features")
                self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg").cuda().eval().requires_grad_(False)
                args.repr_dim = 768; args.in_context_len = 1
            elif args.d2v2_probe_features:
                import sys
                print("WARNING: imagenet_dual_probe is using frozen Data2Vec2 features")
                sys.path.append("/home/abaade/cot_tokenizer")
                from data2vec2 import Data2Vec2Encoder
                sys.path.remove("/home/abaade/cot_tokenizer")
                self.d2v2_model = Data2Vec2Encoder().cuda().eval().requires_grad_(False)
                self.d2v2_model.latent_std = self.d2v2_model.latent_std * 0 + 1
                self.d2v2_model.latent_mean = self.d2v2_model.latent_mean * 0
                args.repr_dim = 1024; args.in_context_len = 1
            feature_dim_multiplier = 2 if mode == "v4" and not (args.dino_probe_features or args.d2v2_probe_features) else 1
            self.imagenet_spatial_dim = args.repr_dim * feature_dim_multiplier
            self.imagenet_global_dim = args.repr_dim * args.in_context_len * feature_dim_multiplier
            self.imagenet_spatial_probes = nn.ModuleList([
                nn.Linear(self.imagenet_spatial_dim, args.class_num) for _ in args.probe_learning_rates
            ])
            self.imagenet_global_probes = nn.ModuleList([
                nn.Linear(self.imagenet_global_dim, args.class_num)
                for _ in args.probe_learning_rates
            ])
            self.imagenet_spatial_bn = nn.SyncBatchNorm(1, affine=False, momentum=0.01)
            self.imagenet_global_bn = nn.SyncBatchNorm(1, affine=False, momentum=0.01)
            if args.whiten_dual_probe_features:
                self.dual_probe_whitening = FullWhiteningPair(
                    self.imagenet_spatial_dim,
                    self.imagenet_global_dim,
                    args.num_pca_spatial,
                    args.num_pca_global,
                )

        if args.denoiser_mode == "contrastive_probe":
            self.net.requires_grad_(False)
            self.contrastive_augment = T.Compose([
                T.Lambda(lambda x: (x + 1) / 2),
                # T.RandomResizedCrop(args.img_size, scale=(0.08, 1.0), antialias=True),  # big ? with spatial
                # T.RandomHorizontalFlip(),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                # T.RandomApply([T.Lambda(lambda x: torch.rot90(x, torch.randint(1, 4, ()).item(), (-2, -1)))], p=0.5),
                T.RandomErasing(p=0.5),
                T.RandomApply([T.Lambda(lambda x: (x + 0.05 * torch.randn_like(x)).clamp(0, 1))], p=0.5),
                T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
                T.Lambda(lambda x: x * 2 - 1),
            ]) if args.contrastive_augmentations else nn.Identity()
            self.contrastive_dim_global = 1024
            self.contrastive_dim_spatial = 256
            self.contrastive_global_probes = nn.ModuleList([
                nn.Linear(args.repr_dim * args.in_context_len, self.contrastive_dim_global)
                for _ in args.probe_learning_rates
            ])
            self.contrastive_spatial_probes = nn.ModuleList([
                nn.Linear(args.repr_dim, self.contrastive_dim_spatial)
                for _ in args.probe_learning_rates
            ])

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # frozen net
        if args.denoiser_mode in ("repr_offlinefeats", "mar_frozen_feats_imagenet_labels", "masked_selfdistill", "maskednoised_selfdistill", "repr_mar4_seconditer", "repr_finetune"):
            self.frozen_net = {'v1': REPR_models_old, 'v2':REPR_models, 'v3': REPR_models_new}[mode][args.frozen_model](
                input_size=args.img_size,
                in_channels=3,
                num_classes=args.class_num,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                in_context_len=16,#args.in_context_len,
                repr_dim=args.repr_dim,
                num_patch_heads=args.num_patch_heads,
                diff_mlp_width=args.diff_mlp_width,
                diff_mlp_depth=args.diff_mlp_depth,
                batch_repeat=args.batch_repeat,
                diff_repeat=args.diff_repeat,
                d_num_p=args.d_num_p,
                diff_repeat_repr=args.diff_repeat_repr,
                d_max=args.d_max,
                mask_m=args.mask_m,
                pmi_target_mul=args.pmi_target_mul,
                source_time_repeat=args.source_time_repeat,
                num_repr_heads=args.num_repr_heads,
                masked_selfdistill_mask_ratio=args.masked_selfdistill_mask_ratio,
                masked_selfdistill_loss_weight=args.masked_selfdistill_loss_weight,
                repr_loss_weight=args.repr_loss_weight,
                repr_delta=args.repr_delta,
                repr_loss_weight_iter2=args.repr_loss_weight_iter2,
                imagenet_loss_weight=args.imagenet_loss_weight,
                pad_dropout=args.pad_dropout,
                repr_noise_bins=args.repr_noise_bins,
                add_input_noise=args.add_input_noise,
                mu_repr=args.mu_repr,
                std_repr=args.std_repr,
                mu_mar=args.mu_mar,
                std_mar=args.std_mar,
                version=repr_version,
                mask_strategy_mar=args.mask_strategy_mar,
                mask_strategy_repr=args.mask_strategy_repr,
                do_simclr_augs=args.do_simclr_augs,
                batch_repeat_mar=args.batch_repeat_mar,
                mu_repr_2=args.mu_repr_2,
                std_repr_2=args.std_repr_2,
                mu_mar_2=args.mu_mar_2,
                std_mar_2=args.std_mar_2,
                is_first_iter=args.frozen_is_first_iter, # Frozen only parameter
                is_rope_bug_fixed=True,
                is_repr_encoderdecoder=True,
                diff_mlp_swiglu=args.diff_mlp_swiglu,
            )
            self.frozen_net.requires_grad_(False)

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        self.args = args

        # alan
        self.denoiser_mode = args.denoiser_mode # mae
        self.repr_mode = mode
        self.repr_version = repr_version
        self.frozen_version = args.frozen_version
        self.pmi_attn_mode = args.pmi_attn_mode
        self.hide_pmi_source = args.hide_pmi_source
        self.batch_repeat = args.batch_repeat
        self.batch_repeat_mar = args.batch_repeat_mar
        self.offline_feats_mul = args.offline_feats_mul
        self.register_buffer("train_step", torch.zeros((), dtype=torch.long))
        self.teacher_feats_mul = args.teacher_feats_mul

        # reprv2_probe_path = '/home/abaade/exps/latentvar/cached_stats/debug_reprv2_1b_feats_scratchv2_v1_600ep_whiten_probe.pt'
        # self.probe = nn.Linear(768,768,bias=True)
        # self.probe.load_state_dict(torch.load(reprv2_probe_path, weights_only=True))
        # self.probe.eval(); self.probe.requires_grad_(False)

    def get_optimizer_param_groups(self, weight_decay, lr, effective_batch_size):
        if self.denoiser_mode in ("imagenet_dual_probe", "contrastive_probe"):
            param_groups = []
            prefix = "imagenet" if self.denoiser_mode == "imagenet_dual_probe" else "contrastive"
            spatial_probes = getattr(self, f"{prefix}_spatial_probes")
            global_probes = getattr(self, f"{prefix}_global_probes")
            probes = zip(self.args.probe_learning_rates, spatial_probes, global_probes)
            for i, (probe_lr, spatial_probe, global_probe) in enumerate(probes):
                param_groups.append({
                    "params": list(spatial_probe.parameters()) + list(global_probe.parameters()),
                    "lr_scale": probe_lr * effective_batch_size / 256 / lr,
                    "weight_decay": weight_decay,
                    "name": f"probe_{i}_lr_{probe_lr:g}",
                })
            return param_groups
        if hasattr(self.net, "get_mup_param_groups") and any(
            parameter.requires_grad for parameter in self.net.parameters()
        ):
            print("Using model_repr_mar_2 MuP parameter groups")
            return self.net.get_mup_param_groups(weight_decay)
        return None

    def forward_mae(self, x):
        return self.net.forward_mae(x)
    
    def forward_mar(self, x):
        return self.net.forward_mar_only(x), {}

    def forward_mar_twotime(self, x, labels):
        loss, mar_loss, imagenet_loss, imagenet_acc = self.net.forward_mar_twotime(x, labels)
        return loss, {
            'mar_loss': mar_loss.item(),
            'imagenet_loss': imagenet_loss.item(),
            'imagenet_acc': imagenet_acc.item(),
        }
    
    def forward_repr(self, x):
        return self.net.forward_repr_8(x)

    @torch.no_grad()
    def get_stats(self, x, labels=None):
        self.net.eval()
        spatial, global_features = self.net.forward_reprv2_4_feats(x, labels=labels)
        spatial = spatial.reshape(x.size(0), self.net.pad_hw, self.net.pad_hw, self.net.repr_dim)[:, self.net.factor_pad:self.net.factor_pad+self.net.hw_len, self.net.factor_pad:self.net.factor_pad+self.net.hw_len].reshape(-1, self.net.repr_dim).float()
        features = {"spatial": spatial, "global": global_features.reshape(-1, self.net.repr_dim).float()}
        return {f"{name}_{stat}": getattr(feats, stat)(0) for name, feats in features.items() for stat in ("mean", "std")}

    def forward_repr_offlinefeats(self, x):
        x = x[:x.size(0)//self.batch_repeat]  # GIGA TODO HORRIBLY INEFFICIENT DATALOADING
        self.frozen_net.eval()
        
        if self.frozen_version == "11":
            offline_feats = self.frozen_net.forward_reprv2_features(x)
        else:
            offline_feats = self.frozen_net.forward_features(x)
        offline_feats = offline_feats * self.offline_feats_mul

        if self.repr_version == "17":
            return self.net.forward_repr_17(x, offline_feats, offline_feats)
        else:
            return self.net.forward_repr_11(x, offline_feats, offline_feats)

    def forward_repr_noiseaug(self, x):
        x = x[:x.size(0)//self.batch_repeat]
        if self.repr_version == "22":
            return self.net.forward_repr_22(x)
        return self.net.forward_repr_20(x)

    def forward_repr_noiseaug_mar4(self, x, labels=None):
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            return self.net.forward_repr_22(x, labels=labels)
        else:
            loss = self.net.forward_mar_only(x, labels=labels)
            return loss, {'mse_loss': loss.item()}

    def forward_repr_noiseaug_mar4_second_iter(self, x, labels=None):
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            with torch.no_grad():
                self.frozen_net.eval()
                x_target, _ = self.frozen_net.forward_reprv2_4_feats(x, return_padded=False)
            return self.net.forward_repr_22(x, x_target * self.teacher_feats_mul, labels=labels)
        else:
            x = x[:x.size(0)//self.batch_repeat_mar]
            with torch.no_grad():
                self.frozen_net.eval()
                x_target, _ = self.frozen_net.forward_reprv2_4_feats(x, return_padded=False)
            loss = self.net.forward_mar_only(x, x_target * self.teacher_feats_mul, labels=labels)
            return loss, {'mse_loss': loss.item()}

    def forward_repr_noiseaug_mar4_multiiter(self, x):
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            return self.net.forward_repr(x)
        else:
            x = x[:x.size(0)//self.batch_repeat_mar]
            return self.net.forward_mar_only(x)

    def forward_repr_noiseaug_mar4_multiiter_frozenmar(self, x):
        x = x[:x.size(0)//self.batch_repeat]
        self.frozen_mar.eval()

        with torch.no_grad():
            self.frozen_net.eval()
            x_target, _ = self.frozen_net.forward_reprv2_4_feats(x, return_padded=False)

        loss, repr_loss, repr_loss_global = self.net.forward_reprv2_4_repr_jointenc_frozenmar(x, self.frozen_mar, x_target * self.teacher_feats_mul)

        return loss, {
            'repr_loss': repr_loss.item(),
            'repr_loss_global': repr_loss_global.item(),
        }

    def forward_repr_noiseaug_mar4_crossview(self, x):
        step = int(self.train_step.item())
        self.train_step.add_(1)

        if step % 4 == 0:
            x = x[:x.size(0)//self.batch_repeat]
            loss, mse_loss, repr_loss, repr_loss_global = self.net.forward_reprv2_4_repr_jointenc_crossview(x)
            return loss, {'mse_loss': mse_loss.item(), 'repr_loss': repr_loss.item(), 'repr_loss_global': repr_loss_global.item()}
        loss = self.net.forward_mar_only_crossview(x)
        return loss, {'mse_loss': loss.item()}

    def forward_mar_in_ft(self, x, labels):
        return self.net.forward_repr_mar_in_ft(x, labels)

    def forward_mar_frozen_feats_imagenet_labels(self, x, labels):
        self.frozen_net.eval()
        with torch.no_grad():
            feats = self.frozen_net.forward_features(x)
        return self.net.forward_repr_mar_frozen_feats_imagenet_labels(x, feats, labels)

    def forward_masked_selfdistill(self, x):
        self.frozen_net.eval()
        feats_target = self.frozen_net.forward_features(x, int(256*self.args.masked_selfdistill_mask_ratio))
        feats_pred = self.net.forward_features(x)
        loss = F.mse_loss(feats_pred, feats_target)
        return loss, {}

    def forward_maskednoised_selfdistill(self, x):
        self.frozen_net.eval()
        with torch.no_grad():
            t = self.args.selfdistill_t
            corrupted_x = t * x + (1 - t) * torch.randn_like(x)
            corrupted_feats = self.frozen_net.forward_reprv2_4_feats(corrupted_x, self.args.selfdistill_num_mask_tokens)
        clean_feats = self.net.forward_reprv2_4_feats(x)
        loss = sum(F.mse_loss(clean, corrupted) for clean, corrupted in zip(clean_feats, corrupted_feats))
        return loss, {'selfdistill_loss': loss.item()}

    @torch.no_grad()
    def forward_imagenet_eval_features(self, x):
        self.net.eval()
        feats = self.net.forward_features(x)
        return feats.mean(dim=1).float()

    def forward_imagenet_eval(self, x, labels):
        labels = labels.long()
        logits = self.imagenet_eval_head(self.forward_imagenet_eval_features(x))
        loss = F.cross_entropy(logits, labels)
        acc1 = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, {"imagenet_eval_loss": loss.item(), "imagenet_eval_top1": acc1.item()}

    @torch.no_grad()
    def forward_imagenet_dual_probe_features(self, x):
        if self.args.dino_probe_features:
            self.dino_model.eval()
            x = F.interpolate((x + 1) / 2, (224, 224), mode="bicubic", align_corners=False, antialias=True)
            x = (x - x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]) / x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            features = self.dino_model.forward_features(x)
            spatial_features = features["x_norm_patchtokens"].mean(dim=1).float()
            global_features = features["x_norm_clstoken"].float()
            return spatial_features, global_features
        if self.args.d2v2_probe_features:
            self.d2v2_model.eval()
            spatial_features = self.d2v2_model.encode((x + 1) / 2).flatten(2).mean(-1).float()
            return spatial_features, spatial_features
        self.net.eval()
        if self.repr_mode == "v4":
            features = self.net.forward_feats_bothiters(x)
            spatial_features = torch.cat([
                features["repr_spatial_iter1"],
                features["repr_spatial_iter2"],
            ], dim=-1)
            global_features = torch.cat([
                features["repr_global_iter1"],
                features["repr_global_iter2"],
            ], dim=-1).flatten(1)
        else:
            spatial_features, global_features = self.net.forward_reprv2_4_feats(x)
        spatial_features = spatial_features.mean(dim=1).float()
        global_features = global_features.float()

        if self.dual_probe_whitening is not None:
            if self.training and not self.dual_probe_whitening.frozen:
                self.dual_probe_whitening.update(spatial_features, global_features)
            if self.dual_probe_whitening.frozen:
                return self.dual_probe_whitening(spatial_features, global_features)

        spatial_features = self.imagenet_spatial_bn(spatial_features.flatten()[:, None]).view_as(spatial_features)
        global_features = self.imagenet_global_bn(global_features.flatten()[:, None]).view_as(global_features)
        spatial_features *= (768 / self.imagenet_spatial_dim) ** 0.5
        global_features *= (768 / self.imagenet_global_dim) ** 0.5
        return spatial_features, global_features

    def set_epoch(self, epoch):
        if epoch > 0 and self.dual_probe_whitening is not None:
            self.dual_probe_whitening.freeze()

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

    def forward_imagenet_dual_probe(self, x, labels):
        labels = labels.long()
        spatial_features, global_features = self.forward_imagenet_dual_probe_features(x)

        if self.args.dual_probe_bce:
            targets = F.one_hot(labels, self.num_classes).float()
            weights = targets * 0.999 + (1 - targets) * 0.001
            probe_loss = lambda logits: F.binary_cross_entropy_with_logits(
                logits, targets, weight=weights, reduction="sum"
            ) / labels.numel()
        else:
            probe_loss = lambda logits: F.cross_entropy(logits, labels)

        loss, results = 0, {}
        for i, (probe_lr, spatial_probe, global_probe) in enumerate(zip(
            self.args.probe_learning_rates,
            self.imagenet_spatial_probes,
            self.imagenet_global_probes,
        )):
            spatial_logits = spatial_probe(spatial_features)
            global_logits = global_probe(global_features)
            spatial_loss = probe_loss(spatial_logits)
            global_loss = probe_loss(global_logits)
            loss = loss + spatial_loss + global_loss
            split = "train" if self.training else "val"
            probe = f"probe_{i}_lr_{probe_lr:g}"
            results[f"{split}/spatial_loss/{probe}"] = spatial_loss.item()
            results[f"{split}/spatial_top1/{probe}"] = (spatial_logits.argmax(-1) == labels).float().mean().item()
            results[f"{split}/global_loss/{probe}"] = global_loss.item()
            results[f"{split}/global_top1/{probe}"] = (global_logits.argmax(-1) == labels).float().mean().item()
        return loss, results

    @torch.no_grad()
    def forward_contrastive_probe_features(self, x):
        self.net.eval()
        return self.net.forward_reprv2_4_feats(x, num_mask_tokens=256-16)

    @staticmethod
    def contrastive_loss(view1, view2):
        view1 = F.normalize(view1.float(), dim=-1)
        view2 = F.normalize(view2.float(), dim=-1)
        if dist.is_available() and dist.is_initialized():
            view1_all = torch.cat(all_gather(view1), dim=0)
            view2_all = torch.cat(all_gather(view2), dim=0)
            labels = torch.arange(view1.size(0), device=view1.device) + dist.get_rank() * view1.size(0)
        else:
            view1_all, view2_all = view1, view2
            labels = torch.arange(view1.size(0), device=view1.device)
        return (
            F.cross_entropy(view1 @ view2_all.T / 0.1, labels)
            + F.cross_entropy(view2 @ view1_all.T / 0.1, labels)
        ) / 2

    def forward_contrastive_probe(self, x):
        spatial1, global1 = self.forward_contrastive_probe_features(self.contrastive_augment(x))
        spatial2, global2 = self.forward_contrastive_probe_features(self.contrastive_augment(x))

        loss, results = 0, {}
        for i, (probe_lr, spatial_probe, global_probe) in enumerate(zip(
            self.args.probe_learning_rates,
            self.contrastive_spatial_probes,
            self.contrastive_global_probes,
        )):
            spatial1_probe = rearrange(spatial_probe(spatial1), 'B N D -> B (N D)')
            spatial2_probe = rearrange(spatial_probe(spatial2), 'B N D -> B (N D)')
            global1_probe = global_probe(global1)
            global2_probe = global_probe(global2)
            spatial_loss = self.contrastive_loss(spatial1_probe, spatial2_probe)
            global_loss = self.contrastive_loss(global1_probe, global2_probe)
            loss = loss + spatial_loss + global_loss
            split = "train" if self.training else "val"
            probe = f"probe_{i}_lr_{probe_lr:g}"
            results[f"{split}/spatial_loss/{probe}"] = spatial_loss.item()
            results[f"{split}/global_loss/{probe}"] = global_loss.item()
        return loss, results

    def forward(self, x, labels):
        if self.denoiser_mode == "mae":
            return self.forward_mae(x)
        elif self.denoiser_mode == "mar":
            return self.forward_mar(x)
        elif self.denoiser_mode == "mar_twotime":
            return self.forward_mar_twotime(x, labels)
        elif self.denoiser_mode == "repr":
            return self.forward_repr(x)
        elif self.denoiser_mode == "repr_offlinefeats":
            return self.forward_repr_offlinefeats(x)
        elif self.denoiser_mode == "repr_noiseaug":
            return self.forward_repr_noiseaug(x)
        elif self.denoiser_mode == "repr_noiseaug_mar4":
            return self.forward_repr_noiseaug_mar4(x, labels)
        elif self.denoiser_mode == "repr_noiseaug_mar4_multiiter":
            return self.forward_repr_noiseaug_mar4_multiiter(x)
        elif self.denoiser_mode == "repr_finetune":
            return self.forward_repr_noiseaug_mar4_multiiter_frozenmar(x)
        elif self.denoiser_mode == "repr_mar4_seconditer":
            return self.forward_repr_noiseaug_mar4_second_iter(x, labels)
        elif self.denoiser_mode == "repr_noiseaug_mar4_crossview":
            return self.forward_repr_noiseaug_mar4_crossview(x)
        elif self.denoiser_mode == "mar_in_ft":
            return self.forward_mar_in_ft(x, labels)
        elif self.denoiser_mode == "mar_frozen_feats_imagenet_labels":
            return self.forward_mar_frozen_feats_imagenet_labels(x, labels)
        elif self.denoiser_mode == "masked_selfdistill":
            return self.forward_masked_selfdistill(x)
        elif self.denoiser_mode == "maskednoised_selfdistill":
            return self.forward_maskednoised_selfdistill(x)
        elif self.denoiser_mode == "imagenet_eval":
            return self.forward_imagenet_eval(x, labels)
        elif self.denoiser_mode == "imagenet_dual_probe":
            return self.forward_imagenet_dual_probe(x, labels)
        elif self.denoiser_mode == "contrastive_probe":
            return self.forward_contrastive_probe(x)
        assert False, "self.denoiser_mode"
    
    @torch.no_grad()
    def update_ema(self):
        pass
        # source_params = list(self.parameters())
        # for targ, src in zip(self.ema_params1, source_params):
            # targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        # for targ, src in zip(self.ema_params2, source_params):
            # targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
