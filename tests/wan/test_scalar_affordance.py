import torch

from affordance_wam.wan.scalar_affordance import (
    ScalarAffordanceCodec,
    causal_temporal_slices,
    future_temporal_slices,
)


def test_causal_group_geometry_is_1_plus_four():
    assert causal_temporal_slices(17, 4) == [
        slice(0, 1),
        slice(1, 5),
        slice(5, 9),
        slice(9, 13),
        slice(13, 17),
    ]


def test_codec_preserves_all_twelve_future_frames_and_bf16_boundary():
    codec = ScalarAffordanceCodec(16, 2).to(dtype=torch.bfloat16)
    scalar = (
        torch.arange(17, dtype=torch.float32)
        .view(1, 1, 17, 1, 1)
        .expand(1, 1, 17, 8, 10)
        / 16.0
    )
    latent = codec.encode(scalar)
    decoded = codec.decode(latent, num_frames=17, clamp=False).float()
    future = codec.decode_future(latent[:, :, 2:], clamp=False).float()
    assert latent.shape == (1, 16, 5, 4, 5)
    assert future.shape == (1, 1, 12, 8, 10)
    means = future.mean(dim=(0, 1, 3, 4))
    assert torch.unique(means).numel() == 12
    assert torch.allclose(decoded, scalar, atol=2e-2)


def test_future_only_codec_uses_exact_four_frame_groups():
    assert future_temporal_slices(12, 4) == [
        slice(0, 4),
        slice(4, 8),
        slice(8, 12),
    ]
    codec = ScalarAffordanceCodec(16, 2)
    future = torch.rand(2, 1, 12, 8, 10)
    latent = codec.encode_future(future)
    assert latent.shape == (2, 16, 3, 4, 5)
    assert codec.decode_future(latent).shape == future.shape
    assert torch.isfinite(codec.future_reconstruction_loss(future))
    with torch.no_grad():
        try:
            codec.encode_future(future[:, :, :-1])
        except ValueError as exc:
            assert "divisible" in str(exc)
        else:
            raise AssertionError("incomplete temporal groups must be rejected")


def test_codec_initial_latent_is_not_rank_one():
    codec = ScalarAffordanceCodec(16, 2)
    scalar = torch.rand(2, 1, 17, 8, 10)
    latent = codec.encode(scalar)
    assert not torch.allclose(latent[:, 0], latent[:, 1])
    assert torch.isfinite(codec.reconstruction_loss(scalar))
