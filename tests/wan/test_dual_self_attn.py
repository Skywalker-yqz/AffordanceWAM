import torch
import torch.nn.functional as F

from affordance_wam.wan.dual_self_attn import action_sees_world_mask
from affordance_wam.wan.action_sampler import _joint_events
from affordance_wam.wan.expert_blocks import (
    _stable_training_context_mask,
    mixed_attention,
)
from affordance_wam.wan_vendor.models import dit as vendor_dit


def test_one_way_visibility_mask():
    mask = action_sees_world_mask(3, 2, torch.device("cpu"))
    assert mask.shape == (5, 5)
    assert mask[:3, :3].all()
    assert not mask[:3, 3:].any()
    assert mask[3:, :].all()


def test_full_visibility_mask_matches_unmasked_attention_and_gradients():
    torch.manual_seed(7)
    inputs = tuple(torch.randn(2, 5, 8) for _ in range(3))

    def evaluate(mask):
        q, k, v = (tensor.clone().requires_grad_(True) for tensor in inputs)
        output = mixed_attention(q, k, v, num_heads=2, mask=mask)
        gradients = torch.autograd.grad(output.square().mean(), (q, k, v))
        return output.detach(), tuple(gradient.detach() for gradient in gradients)

    unmasked_output, unmasked_gradients = evaluate(None)
    full_mask = torch.ones(5, 5, dtype=torch.bool)
    masked_output, masked_gradients = evaluate(full_mask)
    torch.testing.assert_close(masked_output, unmasked_output, rtol=1e-5, atol=1e-6)
    for masked, unmasked in zip(masked_gradients, unmasked_gradients):
        torch.testing.assert_close(masked, unmasked, rtol=1e-5, atol=1e-6)


def test_stable_context_mask_is_full_in_autograd_and_absent_in_inference():
    query = torch.randn(2, 5, 8, requires_grad=True)
    context = torch.randn(2, 3, 8)

    mask = _stable_training_context_mask(query, context)
    assert mask.shape == (5, 3)
    assert mask.dtype == torch.bool
    assert bool(mask.all())

    with torch.no_grad():
        assert _stable_training_context_mask(query, context) is None


def test_masked_vendor_attention_promotes_only_autograd_kernel_to_fp32(monkeypatch):
    observed_dtypes = []
    original_sdpa = F.scaled_dot_product_attention

    def observed_sdpa(q, k, v, attn_mask=None, *args, **kwargs):
        observed_dtypes.append((q.dtype, k.dtype, v.dtype))
        return original_sdpa(q, k, v, attn_mask=attn_mask, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", observed_sdpa)
    inputs = [
        torch.randn(1, 4, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    mask = torch.ones(4, 4, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = vendor_dit.flash_attention(*inputs, num_heads=2, attn_mask=mask)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert observed_dtypes == [(torch.float32, torch.float32, torch.float32)]
    assert all(tensor.grad is not None for tensor in inputs)
    assert all(torch.isfinite(tensor.grad).all() for tensor in inputs)

    observed_dtypes.clear()
    with torch.no_grad():
        inference_output = vendor_dit.flash_attention(
            *(tensor.detach() for tensor in inputs),
            num_heads=2,
            attn_mask=mask,
        )
    assert inference_output.dtype == torch.bfloat16
    assert observed_dtypes == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]


def test_masked_mixed_attention_promotes_only_autograd_kernel_to_fp32(monkeypatch):
    observed_dtypes = []
    original_sdpa = F.scaled_dot_product_attention

    def observed_sdpa(q, k, v, attn_mask=None, *args, **kwargs):
        observed_dtypes.append((q.dtype, k.dtype, v.dtype))
        return original_sdpa(q, k, v, attn_mask=attn_mask, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", observed_sdpa)
    inputs = [
        torch.randn(1, 5, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    ]
    mask = torch.ones(5, 5, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = mixed_attention(*inputs, num_heads=2, mask=mask)
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert observed_dtypes == [(torch.float32, torch.float32, torch.float32)]
    assert all(tensor.grad is not None for tensor in inputs)
    assert all(torch.isfinite(tensor.grad).all() for tensor in inputs)

    observed_dtypes.clear()
    with torch.no_grad():
        inference_output = mixed_attention(
            *(tensor.detach() for tensor in inputs), num_heads=2, mask=mask
        )
    assert inference_output.dtype == torch.bfloat16
    assert observed_dtypes == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]


def test_unequal_step_event_union_executes_every_update_once():
    for world_steps, action_steps in ((3, 2), (2, 3), (35, 20), (35, 35)):
        world_done = action_done = 0
        for progress in _joint_events(world_steps, action_steps):
            if progress * world_steps == world_done:
                world_done += 1
            if progress * action_steps == action_done:
                action_done += 1
        assert world_done == world_steps
        assert action_done == action_steps
