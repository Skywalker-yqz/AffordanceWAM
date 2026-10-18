import json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from tests.wan.helpers import _model
from affordance_wam.wan import expert_blocks as experts
from affordance_wam.wan.dual_self_attn import run_dual_self_attn_loop
from affordance_wam.wan.action_sampler import run_wan_action_inference
from affordance_wam.wan.slot_latent import build_world_token_layout
from affordance_wam.data.dataset import AffordanceWAMDataset
from affordance_wam.wan.checkpoint import load_model_parameters


def batch(size=1):
    return {
        "video": torch.randint(0, 256, (size, 3, 17, 8, 8), dtype=torch.uint8),
        "affordance": torch.rand(size, 1, 17, 8, 8),
        "action": torch.randn(size, 12, 7),
        "proprio_current": torch.randn(size, 7),
        "has_action": torch.ones(size),
        "has_affordance": torch.ones(size),
        "ai_caption": ["move the object"] * size,
    }


def test_two_layer_mask_blocks_direct_and_indirect_heatmap_and_action_information():
    torch.manual_seed(5)
    model = _model().eval()
    world = torch.randn(1, 4, 11, 4, 4)
    action = torch.randn(1, 12, 7)
    context = model.encode_text([""], 1)
    time = torch.tensor([500.0])

    def evaluate(w, a):
        ws = experts.video_prepare(model, w, time, model.dit.embed_text(context))
        ac = model.action_backbone.prepare(a, time, context, proprio=torch.zeros(1, 7))
        return run_dual_self_attn_loop(model.dit, model.action_backbone, ws, ac)

    with torch.no_grad():
        w0, a0 = evaluate(world, action)
        changed_heat = world.clone()
        changed_heat[:, :, 8:] += torch.randn_like(changed_heat[:, :, 8:]) * 20
        wh, ah = evaluate(changed_heat, action)
        wa, aa = evaluate(world, action + 10)
    core = model.world_layout_for(world).core_token_count
    torch.testing.assert_close(w0[:, :core], wh[:, :core], atol=0, rtol=0)
    torch.testing.assert_close(a0, ah, atol=0, rtol=0)
    torch.testing.assert_close(w0, wa, atol=0, rtol=0)
    assert not torch.allclose(w0[:, core:], wh[:, core:])
    assert not torch.allclose(a0, aa)


def test_robot_and_human_gradients_follow_supervision_routes():
    torch.manual_seed(2)
    robot = _model().train()
    output = robot(batch(), iteration=3)
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in robot.dit.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in robot.action_backbone.parameters()
    )
    human = _model().train()
    data = batch()
    del data["action"]
    del data["proprio_current"]
    data["has_action"].zero_()
    human(data, iteration=3)["loss"].backward()
    assert all(p.grad is None for p in human.action_backbone.parameters())
    assert any(p.grad is not None for p in human.dit.parameters())


def test_scalar_codec_receives_flow_and_reconstruction_gradients():
    model = _model().train()
    data = batch()
    world, reconstruction = model._encode_world(data)
    reconstruction.backward(retain_graph=True)
    assert model.scalar_afd_codec.patch_decoder.weight.grad.abs().sum() > 0
    assert model.scalar_afd_codec.patch_encoder.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    sigma, time, weight, noisy, target = model._sample_world_flow(world)
    pred = model._forward_world(noisy, time, model.encode_text([""], 1))
    losses = model._world_losses(
        pred, target, world, noisy, sigma, weight, torch.ones(1)
    )
    losses["L_scalar_afd"].backward()
    assert model.scalar_afd_codec.patch_encoder.weight.grad.abs().sum() > 0
    assert model.scalar_afd_codec.patch_decoder.weight.grad is None


def test_posttrain_mixes_samples_and_freezes_complete_world():
    model = _model().train()
    model.set_training_stage("posttrain")
    predicted = torch.randn(4, 4, 11, 4, 4, requires_grad=True)
    condition, times, noisy = model._posttrain_world_condition(predicted)
    assert noisy.sum().item() == 3
    assert (times == 0).sum().item() == 1
    torch.testing.assert_close(condition[:, :, :2], predicted[:, :, :2])
    torch.testing.assert_close(condition[~noisy], predicted[~noisy])
    data = batch(4)
    del data["affordance"]
    data["predicted_world_latent"] = predicted
    result = model(data, iteration=0)
    result["loss"].backward()
    assert result["posttrain_noisy_world_fraction"].item() == 0.75
    assert predicted.grad is None
    for module in [model.dit, model.view_embedder, model.scalar_afd_codec]:
        assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.action_backbone.parameters()
    )


def test_history_only_inference_ignores_future_targets_and_reclamps_history():
    model = _model().eval()
    data = batch()
    history = {
        "video": data["video"][:, :, :5].clone(),
        "proprio_current": data["proprio_current"],
        "ai_caption": data["ai_caption"],
    }
    full, actions_full = run_wan_action_inference(
        model, data, seed=7, num_steps=3, action_num_steps=2, action_refinement_steps=1
    )
    partial, actions_partial = run_wan_action_inference(
        model,
        history,
        seed=7,
        num_steps=3,
        action_num_steps=2,
        action_refinement_steps=1,
    )
    torch.testing.assert_close(
        full.samples_latent, partial.samples_latent, atol=0, rtol=0
    )
    np.testing.assert_array_equal(actions_full, actions_partial)
    torch.testing.assert_close(full.samples_latent[:, :, :2], full.rgb_history_latent)
    assert partial.scalar_affordance_future_pixel.shape == (1, 1, 12, 8, 8)
    assert partial.rgb_pixel.shape == (1, 3, 17, 8, 8)


def test_uniform_flow_times_and_exact_history():
    model = _model()
    x0 = torch.randn(5000, 4, 11, 2, 2)
    sigma, timestep, weight, noisy, target = model._sample_world_flow(x0)
    assert sigma.mean().item() == pytest.approx(0.5, abs=0.02)
    assert ((sigma >= 0) & (sigma < 1)).all()
    assert torch.equal(weight, torch.ones_like(weight))
    torch.testing.assert_close(noisy[:, :, :2], x0[:, :, :2])
    torch.testing.assert_close(timestep, sigma * 1000)


def test_history_only_dataset_needs_no_actions_or_affordance(tmp_path):
    np.savez(
        tmp_path / "sample.npz",
        video=np.zeros((5, 8, 8, 3), dtype=np.uint8),
        proprio_current=np.zeros(7, dtype=np.float32),
    )
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"sample": "sample.npz", "instruction": "move"}) + "\n"
    )
    dataset = AffordanceWAMDataset(
        tmp_path / "manifest.jsonl",
        data_form="vl_action",
        inference=True,
        video_size=(8, 8),
    )
    item = dataset[0]
    assert item["video"].shape == (3, 5, 8, 8)
    assert "action" not in item and "affordance" not in item


def test_checkpoint_roundtrip_restores_both_experts(tmp_path):
    model = _model()
    metadata = {
        "schema_version": 1,
        "stage": "pretrain",
        "model_config": asdict(model.config),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    torch.save({"module": model.state_dict()}, tmp_path / "mp_rank_00_model_states.pt")
    restored = _model()
    load_model_parameters(restored, tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_curriculum_progression():
    model = _model()
    model.config.fixed_world_prob_start, model.config.fixed_world_prob_end = 0.8, 0.4
    start = model.config.pretrain_world_only_updates
    end = start + model.config.curriculum_updates
    assert model._curriculum(start, 0.8, 0.4) == 0.8
    assert model._curriculum(end, 0.8, 0.4) == 0.4
    assert model._predicted_condition_probability(start) == 0
    assert model._predicted_condition_probability(end) == 1
