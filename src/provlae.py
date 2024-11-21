import math
from math import ceil, log2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal


class ProVLAE(nn.Module):
    def __init__(
        self,
        image_size,
        z_dim=3,
        chn_num=3,
        num_ladders=3,
        hidden_dim=32,
        fc_dim=256,
        beta=1.0,
        learning_rate=5e-4,
        fade_in_duration=5000,
        pre_kl=True,
        coff=0.5,
        train_seq=1,
        use_kl_annealing=False,
        kl_annealing_mode="linear",
        cycle_period=4,
        max_kl_weight=1,
        min_kl_weight=0.1,
        ratio=1.0,
        use_capacity_increase=False,
        gamma=1000.0,
        max_capacity=25,
        capacity_max_iter=1e-5,
    ):
        super(ProVLAE, self).__init__()

        # Calculate network architecture parameters
        self.hidden_dims = [hidden_dim] * num_ladders
        self.image_size = image_size
        self.target_size = 2 ** ceil(log2(image_size))  # Nearest power of 2
        self.min_size = 4  # Minimum feature map size

        # Calculate the number of possible downsampling steps
        self.max_steps = int(log2(self.target_size // self.min_size))
        self.num_stages = max(self.max_steps, 1)
        self.num_ladders = min(num_ladders, self.num_stages)

        self.z_dim = max(1, z_dim)
        self.chn_num = chn_num
        self.beta = beta
        self.pre_kl = pre_kl
        self.coff = coff
        self.learning_rate = learning_rate
        self.fade_in_duration = fade_in_duration
        self.train_seq = min(train_seq, self.num_ladders)

        # for kl annealing
        self.use_kl_annealing = use_kl_annealing
        self.kl_annealing_mode = kl_annealing_mode
        self.current_epoch = None
        self.num_epochs = None
        self.cycle_period = cycle_period
        self.max_kl_weight = max_kl_weight
        self.min_kl_weight = min_kl_weight
        self.ratio = ratio

        # Improving disentangling in β-VAE with controlled capacity increase
        self.use_capacity_increase = use_capacity_increase
        self.gamma = gamma
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter

        # Calculate encoder sizes
        self.encoder_sizes = [self.target_size]
        current_size = self.target_size
        for _ in range(self.num_ladders + 1):  # +1 for initial size
            current_size = max(current_size // 2, self.min_size)
            self.encoder_sizes.append(current_size)

        # Dynamic hidden dimensions
        if len(self.hidden_dims) < self.num_ladders:
            self.hidden_dims.extend([self.hidden_dims[-1]] * (self.num_ladders - len(self.hidden_dims)))
        self.hidden_dims = self.hidden_dims[: self.num_ladders]

        self.activation = nn.LeakyReLU()  # or ELU
        self.q_dist = Normal
        self.x_dist = Bernoulli
        self.prior_params = nn.Parameter(torch.zeros(self.z_dim, 2))

        # Create encoder layers
        self.encoder_layers = nn.ModuleList()
        current_channels = chn_num
        for dim in self.hidden_dims:
            self.encoder_layers.append(self._create_conv_block(current_channels, dim))
            current_channels = dim

        # Create ladder networks
        self.ladders = nn.ModuleList()
        for i in range(self.num_ladders):
            ladder_input_size = self.encoder_sizes[i + 1]
            self.ladders.append(self._create_ladder_block(self.hidden_dims[i], fc_dim, self.z_dim, ladder_input_size))

        # Create generator networks
        self.generators = nn.ModuleList()
        for i in range(self.num_ladders):
            size = self.encoder_sizes[i + 1]
            self.generators.append(self._create_generator_block(self.z_dim, fc_dim, (self.hidden_dims[i], size, size)))

        # Create decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_ladders - 1):
            out_size = self.encoder_sizes[i]
            self.decoder_layers.append(
                self._create_decoder_block(
                    self.hidden_dims[-(i + 1)] * 2,  # Account for concatenation
                    self.hidden_dims[-(i + 2)],
                    out_size,
                )
            )

        # Additional upsampling to reach target size
        self.additional_ups = nn.ModuleList()
        current_size = self.encoder_sizes[1]  # Start from size after first encoder
        while current_size < self.target_size:
            next_size = min(current_size * 2, self.target_size)
            self.additional_ups.append(self._create_upsampling_block(self.hidden_dims[0], next_size))
            current_size = next_size

        # Final output layer
        self.output_layer = nn.Conv2d(self.hidden_dims[0], chn_num, kernel_size=3, padding=1)

    def _create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )

    def _create_ladder_block(self, in_channels, fc_dim, z_dim, input_size):
        def get_conv_output_size(input_size, kernel_size=4, stride=2, padding=1):
            return ((input_size + 2 * padding - (kernel_size - 1) - 1) // stride) + 1

        conv_size = get_conv_output_size(input_size)
        total_flatten_size = in_channels * conv_size * conv_size

        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            self.activation,
            nn.Flatten(),
            nn.Linear(total_flatten_size, fc_dim),
            nn.BatchNorm1d(fc_dim),
            self.activation,
            nn.Linear(fc_dim, z_dim * 2),
        )

    def _create_generator_block(self, z_dim, fc_dim, output_shape):
        total_dim = output_shape[0] * output_shape[1] * output_shape[2]
        return nn.Sequential(
            nn.Linear(z_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            self.activation,
            nn.Linear(fc_dim, total_dim),
            nn.BatchNorm1d(total_dim),
            self.activation,
            nn.Unflatten(1, output_shape),
        )

    def _create_decoder_block(self, in_channels, out_channels, target_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
            nn.Upsample(size=(target_size, target_size), mode="nearest"),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation,
        )

    def _create_upsampling_block(self, channels, target_size):
        return nn.Sequential(
            nn.Upsample(size=(target_size, target_size), mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            self.activation,
        )

    def _sample_latent(self, z_params):
        z_mean, z_log_var = torch.chunk(z_params, 2, dim=1)
        z_log_var = torch.clamp(z_log_var, min=-1e2, max=3)

        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)

        return z_mean + eps * std, z_mean, z_log_var

    def _kl_divergence(self, z_mean, z_log_var):
        return -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    def fade_in_alpha(self, step):
        if step > self.fade_in_duration:
            return 1.0
        return step / self.fade_in_duration

    def frange_cycle_linear(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period / ratio)

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and int(i + c * period) < n_epoch:
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L

    def frange_cycle_sigmoid(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)  # step is in [0,1]

        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop:
                L[int(i + c * period)] = 1.0 / (1.0 + np.exp(-(v * 12.0 - 6.0)))
                v += step
                i += 1
        return L

    def frange_cycle_cosine(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)  # step is in [0,1]

        # transform into [0, pi] for plots:

        for c in range(n_cycle):

            v, i = start, 0
            while v <= stop:
                L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
                v += step
                i += 1
        return L

    def cycle_kl_weights(self, epoch, n_epoch, cycle_period=4, max_kl_weight=1.0, min_kl_weight=0.1, ratio=0.5):
        if self.kl_annealing_mode == "linear":
            kl_weights = self.frange_cycle_linear(min_kl_weight, max_kl_weight, n_epoch, cycle_period, ratio)
        if self.kl_annealing_mode == "sigmoid":
            kl_weights = self.frange_cycle_sigmoid(min_kl_weight, max_kl_weight, n_epoch, cycle_period, ratio)
        if self.kl_annealing_mode == "cosine":
            kl_weights = self.frange_cycle_cosine(min_kl_weight, max_kl_weight, n_epoch, cycle_period, ratio)

        return kl_weights[epoch]

    def encode(self, x):
        original_size = x.size()[-2:]  # Store original size

        # Resize to target size
        if original_size != (self.target_size, self.target_size):
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=True,
            )

        h_list = []
        h = x

        # Track encoder outputs
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            expected_size = self.encoder_sizes[i + 1]
            assert h.size(-1) == expected_size
            h_list.append(h)

            if i + 1 == self.train_seq - 1:
                h = h * self.fade_in

        z_params = []
        for i in range(self.num_ladders):
            ladder_output = self.ladders[i](h_list[i])
            z, z_mean, z_log_var = self._sample_latent(ladder_output)
            z_params.append((z, z_mean, z_log_var))

        return z_params, original_size

    def decode(self, z_list, original_size):
        # Generate features from latent vectors
        features = []
        for i, z in enumerate(z_list):
            f = self.generators[i](z)
            if i > self.train_seq - 1:
                f = f * 0
            elif i == self.train_seq - 1:
                f = f * self.fade_in
            features.append(f)

        # Progressive decoding with explicit size management
        x = features[-1]  # Start from deepest layer
        for i in range(self.num_ladders - 2, -1, -1):
            # Ensure feature maps have matching spatial dimensions
            target_size = features[i].size(-1)
            if x.size(-1) != target_size:
                x = F.interpolate(x, size=(target_size, target_size), mode="nearest")

            # Concatenate features
            x = torch.cat([features[i], x], dim=1)
            if i < len(self.decoder_layers):
                x = self.decoder_layers[i](x)

        # Additional upsampling if needed
        for up_layer in self.additional_ups:
            x = up_layer(x)

        x = self.output_layer(x)  # Final convolution

        # Resize to original input size
        if original_size != (x.size(-2), x.size(-1)):
            x = F.interpolate(x, size=original_size, mode="bilinear", align_corners=True)

        return x

    def forward(self, x, step=0):
        self.fade_in = self.fade_in_alpha(step)
        kl_weight = self.cycle_kl_weights(
            epoch=self.current_epoch,
            n_epoch=self.num_epochs,
            cycle_period=self.cycle_period,
            max_kl_weight=self.max_kl_weight,
            min_kl_weight=self.min_kl_weight,
            ratio=self.ratio,
        )

        z_params, original_size = self.encode(x)

        latent_losses = []
        zs = []
        for z, z_mean, z_log_var in z_params:
            latent_losses.append(self._kl_divergence(z_mean, z_log_var))
            zs.append(z)

        latent_loss = sum(latent_losses)
        x_recon = self.decode(zs, original_size)

        # Reconstruction loss
        bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        recon_loss = bce_loss(x_recon, x)

        # prekl loss
        active_latents = latent_losses[self.train_seq - 1 :]
        inactive_latents = latent_losses[: self.train_seq - 1]
        if self.use_kl_annealing:
            if self.use_capacity_increase:
                # https://arxiv.org/pdf/1804.03599.pdf
                self.C_max = self.C_max.to(x.device)
                C = torch.clamp(self.C_max / self.C_stop_iter * step, 0, self.C_max.data[0])
                kl_term = self.gamma * kl_weight * (sum(active_latents) - C).abs()
            else:
                # https://openreview.net/forum?id=Sy2fzU9gl
                kl_term = kl_weight * self.beta * sum(active_latents)
        else:
            kl_term = self.beta * sum(active_latents)

        loss = recon_loss + kl_term
        if self.pre_kl:
            loss += self.coff * sum(inactive_latents)

        return torch.sigmoid(x_recon), loss, kl_term, recon_loss, kl_weight

    def inference(self, x):
        with torch.no_grad():
            z_params, _ = self.encode(x)
            return z_params

    def generate(self, z_list):
        with torch.no_grad():
            return torch.sigmoid(self.decode(z_list, (self.image_size, self.image_size)))
