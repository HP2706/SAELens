from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from torch import Tensor
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sparse_autoencoder import ForwardOutput, SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import (
    SAETrainContext,
    SAETrainer,
    TrainStepOutput,
    _build_train_context,
    _build_train_step_log_dict,
    _log_feature_sparsity,
    _train_step,
    _update_sae_lens_training_version,
)
from tests.unit.helpers import build_sae_cfg


def build_train_ctx(
    sae: SparseAutoencoder,
    act_freq_scores: Tensor | None = None,
    n_forward_passes_since_fired: Tensor | None = None,
    n_frac_active_tokens: int = 0,
) -> SAETrainContext:
    """Builds a training context. We need to have this version so we can override some attributes."""
    # Build train context
    ctx = _build_train_context(sae, sae.cfg.training_tokens)
    # Override attributes if required for testing
    ctx.n_frac_active_tokens = n_frac_active_tokens
    if n_forward_passes_since_fired is not None:
        ctx.n_forward_passes_since_fired = n_forward_passes_since_fired
    else:
        ctx.n_forward_passes_since_fired = torch.zeros(sae.cfg.d_sae)  # type: ignore
    if act_freq_scores is not None:
        ctx.act_freq_scores = act_freq_scores
    else:
        ctx.act_freq_scores = torch.zeros(sae.cfg.d_sae)  # type: ignore
    return ctx


def modify_sae_output(
    sae: SparseAutoencoder, modifier: Callable[[ForwardOutput], ForwardOutput]
):
    """
    Helper to modify the output of the SAE forward pass for use in patching, for use in patch side_effect.
    We need real grads during training, so we can't just mock the whole forward pass directly.
    """

    def modified_forward(*args: Any, **kwargs: Any):
        output = SparseAutoencoder.forward(sae, *args, **kwargs)
        return modifier(output)

    return modified_forward


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts() -> None:
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0)
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(sae)

    layer_acts = torch.randn(10, 1, 64)

    # intentionally train on the same activations 5 times to ensure loss decreases
    train_outputs = [
        _train_step(
            sparse_autoencoder=sae,
            ctx=ctx,
            sae_in=layer_acts[:, 0, :],
            feature_sampling_window=1000,
            use_wandb=False,
            n_training_steps=10,
            batch_size=10,
            wandb_suffix="",
        )
        for _ in range(5)
    ]

    # ensure loss decreases with each training step
    for output, next_output in zip(train_outputs[:-1], train_outputs[1:]):
        assert output.loss > next_output.loss
    assert ctx.n_frac_active_tokens == 50  # should increment each step by batch_size


def test_train_step__output_looks_reasonable() -> None:
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0)
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(sae)

    layer_acts = torch.randn(10, 2, 64)

    output = _train_step(
        sparse_autoencoder=sae,
        ctx=ctx,
        sae_in=layer_acts[:, 0, :],
        feature_sampling_window=1000,
        use_wandb=False,
        n_training_steps=10,
        batch_size=10,
        wandb_suffix="",
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert torch.allclose(output.sae_in, layer_acts[:, 0, :])
    assert output.sae_out.shape == output.sae_in.shape
    assert output.feature_acts.shape == (10, 128)  # batch_size, d_sae
    assert output.ghost_grad_neuron_mask.shape == (128,)
    assert output.loss.shape == ()
    assert output.mse_loss.shape == ()
    assert output.ghost_grad_loss == 0
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert torch.all(output.ghost_grad_neuron_mask == False)  # noqa
    assert output.ghost_grad_loss == 0
    assert ctx.n_frac_active_tokens == 10
    assert ctx.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert torch.allclose(
        ctx.act_freq_scores, (output.feature_acts.abs() > 0).float().sum(0)
    )


def test_train_step__ghost_grads_mask() -> None:
    cfg = build_sae_cfg(d_in=2, d_sae=4, dead_feature_window=5)
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(
        sae, n_forward_passes_since_fired=torch.tensor([0, 4, 7, 9]).float()
    )

    output = _train_step(
        sparse_autoencoder=sae,
        ctx=ctx,
        sae_in=torch.randn(10, 2),
        feature_sampling_window=1000,
        use_wandb=False,
        n_training_steps=10,
        batch_size=10,
        wandb_suffix="",
    )
    assert torch.all(
        output.ghost_grad_neuron_mask == torch.Tensor([False, False, True, True])
    )


def test_train_step__sparsity_updates_based_on_feature_act_sparsity() -> None:
    cfg = build_sae_cfg(d_in=2, d_sae=4, hook_point_layer=0)
    sae = SparseAutoencoder(cfg)

    feature_acts = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1]]).float()
    layer_acts = torch.randn(3, 1, 2)

    ctx = build_train_ctx(
        sae,
        n_frac_active_tokens=9,
        act_freq_scores=torch.tensor([0, 3, 7, 1]).float(),
        n_forward_passes_since_fired=torch.tensor([8, 2, 0, 0]).float(),
    )
    with patch.object(
        sae,
        "forward",
        side_effect=modify_sae_output(
            sae, lambda out: out._replace(feature_acts=feature_acts)
        ),
    ):
        train_output = _train_step(
            sparse_autoencoder=sae,
            ctx=ctx,
            sae_in=layer_acts[:, 0, :],
            feature_sampling_window=1000,
            use_wandb=False,
            n_training_steps=10,
            batch_size=3,
            wandb_suffix="",
        )

    # should increase by batch_size
    assert ctx.n_frac_active_tokens == 12
    # add freq scores for all non-zero feature acts
    assert torch.allclose(
        ctx.act_freq_scores,
        torch.tensor([2, 3, 8, 3]).float(),
    )
    assert torch.allclose(
        ctx.n_forward_passes_since_fired,
        torch.tensor([0, 3, 0, 0]).float(),
    )

    # the outputs from the SAE should be included in the train output
    assert train_output.feature_acts is feature_acts


def test_log_feature_sparsity__handles_zeroes_by_default_fp32() -> None:
    fp32_zeroes = torch.tensor([0], dtype=torch.float32)
    assert _log_feature_sparsity(fp32_zeroes).item() != float("-inf")


# TODO: currently doesn't work for fp16, we should address this
@pytest.mark.skip(reason="Currently doesn't work for fp16")
def test_log_feature_sparsity__handles_zeroes_by_default_fp16() -> None:
    fp16_zeroes = torch.tensor([0], dtype=torch.float16)
    assert _log_feature_sparsity(fp16_zeroes).item() != float("-inf")


def test_build_train_step_log_dict() -> None:
    cfg = build_sae_cfg(
        d_in=2, d_sae=4, hook_point_layer=0, lr=2e-4, l1_coefficient=1e-2
    )
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(
        sae,
        act_freq_scores=torch.tensor([0, 3, 1, 0]).float(),
        n_frac_active_tokens=10,
        n_forward_passes_since_fired=torch.tensor([4, 0, 0, 0]).float(),
    )
    train_output = TrainStepOutput(
        sae_in=torch.tensor([[-1, 0], [0, 2], [1, 1]]).float(),
        sae_out=torch.tensor([[0, 0], [0, 2], [0.5, 1]]).float(),
        feature_acts=torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]]).float(),
        loss=torch.tensor(0.5),
        mse_loss=torch.tensor(0.25),
        l1_loss=torch.tensor(0.1),
        ghost_grad_loss=torch.tensor(0.15),
        ghost_grad_neuron_mask=torch.tensor([False, True, False, True]),
    )

    log_dict = _build_train_step_log_dict(
        sae, train_output, ctx, wandb_suffix="-wandbftw", n_training_tokens=123
    )
    assert log_dict == {
        "losses/mse_loss-wandbftw": 0.25,
        # l1 loss is scaled by l1_coefficient
        "losses/l1_loss-wandbftw": pytest.approx(10),
        "losses/ghost_grad_loss-wandbftw": pytest.approx(0.15),
        "losses/overall_loss-wandbftw": 0.5,
        "metrics/explained_variance-wandbftw": 0.75,
        "metrics/explained_variance_std-wandbftw": 0.25,
        "metrics/l0-wandbftw": 2.0,
        "sparsity/mean_passes_since_fired-wandbftw": 1.0,
        "sparsity/dead_features-wandbftw": 2,
        "details/current_learning_rate-wandbftw": 2e-4,
        "details/current_l1_coefficient-wandbftw": 0.01,
        "details/n_training_tokens": 123,
    }


def test_train_sae_group_on_language_model__runs(
    ts_model: HookedTransformer,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = build_sae_cfg(
        checkpoint_path=str(checkpoint_dir),
        train_batch_size=32,
        training_tokens=100,
        context_size=8,
    )
    # just a tiny datast which will run quickly
    dataset = Dataset.from_list([{"text": "hello world"}] * 2000)
    activation_store = ActivationsStore.from_config(ts_model, cfg, dataset=dataset)
    sae = SparseAutoencoder(cfg)
    sae = SAETrainer(
        model=ts_model,
        sae=sae,
        activation_store=activation_store,
        batch_size=32,
    ).fit()

    assert isinstance(sae, SparseAutoencoder)


def test_update_sae_lens_training_version_sets_the_current_version():
    cfg = build_sae_cfg(sae_lens_training_version="0.1.0")
    sae = SparseAutoencoder(cfg)
    _update_sae_lens_training_version(sae)
    assert sae.cfg.sae_lens_training_version == __version__
