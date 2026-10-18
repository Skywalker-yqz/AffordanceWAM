"""Frame-preserving Scalar Affordance codec aligned to the video latent grid.

Each four-frame future group is packed into channels, spatially encoded, and
decoded back to four distinct scalar fields. Three slots represent twelve
future frames.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def causal_temporal_slices(num_frames: int, compression: int = 4) -> list[slice]:
    """Return the causal-VAE-aligned pixel-frame groups.

    For the repository default, 17 frames become ``[0:1, 1:5, 5:9, 9:13,
    13:17]``.  This exactly aligns scalar Afd with the five RGB/overlay latent
    time slots without pretending that scalar Afd is an RGB VAE input.
    """

    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if compression < 1:
        raise ValueError("compression must be positive")
    groups = [slice(0, 1)]
    start = 1
    while start < num_frames:
        stop = min(start + compression, num_frames)
        groups.append(slice(start, stop))
        start = stop
    return groups


def future_temporal_slices(num_frames: int, compression: int = 4) -> list[slice]:
    """Return strict future-only groups such as ``[0:4,4:8,8:12]``.

    Unlike a causal RGB stream, future-only scalar Afd has no special first
    frame.  Requiring complete groups prevents an accidental 12-frame input
    from becoming four misaligned slots.
    """

    if num_frames < 1 or compression < 1:
        raise ValueError("num_frames/compression must be positive")
    if num_frames % compression:
        raise ValueError(
            f"future scalar frames ({num_frames}) must be divisible by "
            f"temporal_compression ({compression})"
        )
    return [
        slice(start, start + compression) for start in range(0, num_frames, compression)
    ]


def temporal_group_mean(x: torch.Tensor, compression: int = 4) -> torch.Tensor:
    """Reduce ``[B,C,T,H,W]`` into causal temporal groups by a mean."""

    if x.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(x.shape)}")
    return torch.stack(
        [
            x[:, :, group].mean(dim=2)
            for group in causal_temporal_slices(x.shape[2], compression)
        ],
        dim=2,
    )


def temporal_group_expand(
    x: torch.Tensor,
    lengths: Sequence[int] | None = None,
    *,
    compression: int = 4,
) -> torch.Tensor:
    """Expand grouped ``[B,C,Tg,H,W]`` back to pixel time by repetition."""

    if x.ndim != 5:
        raise ValueError(f"expected [B,C,Tg,H,W], got {tuple(x.shape)}")
    if lengths is None:
        lengths = [1] + [compression] * (x.shape[2] - 1)
    if len(lengths) != x.shape[2] or any(int(n) < 1 for n in lengths):
        raise ValueError(
            f"invalid temporal lengths {list(lengths)} for Tg={x.shape[2]}"
        )
    return torch.cat(
        [x[:, :, i : i + 1].repeat(1, 1, int(n), 1, 1) for i, n in enumerate(lengths)],
        dim=2,
    )


class ScalarAffordanceCodec(nn.Module):
    """Trainable scalar-Afd codec aligned to causal four-frame latent groups.

    Each latent time slot packs up to four *distinct* one-channel frames into
    the convolution input channels.  Thus three future slots still decode to
    twelve 10-fps heatmaps; temporal compression does not collapse them to
    three repeated maps.  A tiny asymmetric initialization perturbation avoids
    a permanently rank-one latent representation while retaining an
    average-pool/patch-repeat starting point.
    """

    def __init__(
        self,
        latent_channels: int,
        spatial_compression: int,
        *,
        temporal_compression: int = 4,
    ) -> None:
        super().__init__()
        if latent_channels < 1 or spatial_compression < 1 or temporal_compression < 1:
            raise ValueError("codec dimensions/compression must be positive")
        self.latent_channels = int(latent_channels)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)
        p = self.spatial_compression
        self.patch_encoder = nn.Conv2d(
            self.temporal_compression,
            self.latent_channels,
            kernel_size=p,
            stride=p,
        )
        self.patch_decoder = nn.ConvTranspose2d(
            self.latent_channels,
            self.temporal_compression,
            kernel_size=p,
            stride=p,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        p = self.spatial_compression
        with torch.no_grad():
            self.patch_encoder.weight.normal_(mean=0.0, std=1e-4)
            self.patch_encoder.bias.zero_()
            self.patch_decoder.weight.normal_(mean=0.0, std=1e-4)
            self.patch_decoder.bias.zero_()
            counts = [0] * self.temporal_compression
            for channel in range(self.latent_channels):
                temporal_channel = channel % self.temporal_compression
                counts[temporal_channel] += 1
                self.patch_encoder.weight[channel, temporal_channel].add_(
                    1.0 / float(p * p)
                )
            for channel in range(self.latent_channels):
                temporal_channel = channel % self.temporal_compression
                self.patch_decoder.weight[channel, temporal_channel].add_(
                    1.0 / float(counts[temporal_channel])
                )

    @staticmethod
    def _check_scalar(x: torch.Tensor) -> None:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"scalar Afd must be [B,1,T,H,W], got {tuple(x.shape)}")

    def _pack_groups(self, scalar_afd: torch.Tensor) -> torch.Tensor:
        """Pack ``[B,1,T,H,W]`` as ``[B,P,T_group,H,W]`` (``P=4``)."""

        packed = []
        for group in causal_temporal_slices(
            scalar_afd.shape[2], self.temporal_compression
        ):
            frames = scalar_afd[:, 0, group]  # [B,L,H,W]
            missing = self.temporal_compression - frames.shape[1]
            if missing:
                frames = torch.cat(
                    [frames, frames[:, -1:].repeat(1, missing, 1, 1)], dim=1
                )
            packed.append(frames)
        return torch.stack(packed, dim=2)

    def _pack_future_groups(self, scalar_future: torch.Tensor) -> torch.Tensor:
        """Pack complete future groups without causal first-frame semantics."""

        return torch.stack(
            [
                scalar_future[:, 0, group]
                for group in future_temporal_slices(
                    scalar_future.shape[2], self.temporal_compression
                )
            ],
            dim=2,
        )

    def _encode_packed(self, grouped: torch.Tensor) -> torch.Tensor:
        grouped = grouped.float().to(
            device=self.patch_encoder.weight.device,
            dtype=self.patch_encoder.weight.dtype,
        )
        b, channels, t, h, w = grouped.shape
        y = self.patch_encoder(
            grouped.permute(0, 2, 1, 3, 4).reshape(b * t, channels, h, w)
        )
        return y.reshape(b, t, self.latent_channels, y.shape[-2], y.shape[-1]).permute(
            0, 2, 1, 3, 4
        )

    def _unpack_groups(
        self, packed: torch.Tensor, lengths: Sequence[int]
    ) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != self.temporal_compression:
            raise ValueError(
                f"packed scalar groups must be [B,{self.temporal_compression},T,H,W], "
                f"got {tuple(packed.shape)}"
            )
        if len(lengths) != packed.shape[2]:
            raise ValueError("temporal group length mismatch")
        return torch.cat(
            [
                packed[:, : int(length), i].unsqueeze(1)
                for i, length in enumerate(lengths)
            ],
            dim=2,
        )

    def encode(self, scalar_afd: torch.Tensor) -> torch.Tensor:
        """Encode scalar Afd into ``[B,Z,T_lat,H/p,W/p]``."""

        self._check_scalar(scalar_afd)
        if (
            scalar_afd.shape[-2] % self.spatial_compression
            or scalar_afd.shape[-1] % self.spatial_compression
        ):
            raise ValueError(
                f"H/W {tuple(scalar_afd.shape[-2:])} must be divisible by {self.spatial_compression}"
            )
        return self._encode_packed(self._pack_groups(scalar_afd))

    def encode_future(self, scalar_future: torch.Tensor) -> torch.Tensor:
        """Encode future-only scalar Afd, normally 12 frames into 3 slots."""

        self._check_scalar(scalar_future)
        if (
            scalar_future.shape[-2] % self.spatial_compression
            or scalar_future.shape[-1] % self.spatial_compression
        ):
            raise ValueError(
                f"H/W {tuple(scalar_future.shape[-2:])} must be divisible by "
                f"{self.spatial_compression}"
            )
        return self._encode_packed(self._pack_future_groups(scalar_future))

    def decode_groups(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode to packed frames ``[B,P,T_group,H,W]`` without flattening."""

        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"latent must be [B,{self.latent_channels},T,h,w], got {tuple(latent.shape)}"
            )
        b, _z, t, h, w = latent.shape
        x = (
            latent.to(
                device=self.patch_decoder.weight.device,
                dtype=self.patch_decoder.weight.dtype,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(b * t, self.latent_channels, h, w)
        )
        y = self.patch_decoder(x)
        return y.reshape(
            b, t, self.temporal_compression, y.shape[-2], y.shape[-1]
        ).permute(0, 2, 1, 3, 4)

    def decode(
        self, latent: torch.Tensor, *, num_frames: int | None = None, clamp: bool = True
    ) -> torch.Tensor:
        """Decode and expand all causal groups to a one-channel video."""

        packed = self.decode_groups(latent)
        if num_frames is None:
            lengths = [1] + [self.temporal_compression] * (packed.shape[2] - 1)
        else:
            lengths = [
                s.stop - s.start
                for s in causal_temporal_slices(num_frames, self.temporal_compression)
            ]
            if len(lengths) != packed.shape[2]:
                raise ValueError(
                    f"num_frames={num_frames} implies {len(lengths)} groups, latent has {packed.shape[2]}"
                )
        out = self._unpack_groups(packed, lengths)
        return out.clamp(0.0, 1.0) if clamp else out

    def decode_future(
        self, latent_future: torch.Tensor, *, clamp: bool = True
    ) -> torch.Tensor:
        """Decode future-only slots (three slots -> twelve frames by default)."""

        packed = self.decode_groups(latent_future)
        out = self._unpack_groups(packed, [self.temporal_compression] * packed.shape[2])
        return out.clamp(0.0, 1.0) if clamp else out

    def reconstruction_loss(self, scalar_afd: torch.Tensor) -> torch.Tensor:
        """Frame-preserving MSE on the packed causal groups."""

        self._check_scalar(scalar_afd)
        target = self._pack_groups(scalar_afd.float())
        pred = self.decode_groups(self.encode(scalar_afd))
        return (pred.float() - target).square().mean()

    def future_reconstruction_loss(self, scalar_future: torch.Tensor) -> torch.Tensor:
        """Frame-preserving reconstruction loss for future-only groups."""

        self._check_scalar(scalar_future)
        target = self._pack_future_groups(scalar_future.float())
        pred = self.decode_groups(self.encode_future(scalar_future))
        return (pred.float() - target).square().mean()


__all__ = [
    "ScalarAffordanceCodec",
    "causal_temporal_slices",
    "future_temporal_slices",
    "temporal_group_expand",
    "temporal_group_mean",
]
