"""Shape and export checks for the edge models (CPU only, no data needed)."""

import pytest
import torch

from models import mobilenet1d, resnet1d_tiny
from train import load_model

MODELS = {"resnet1d_tiny": resnet1d_tiny, "mobilenet1d": mobilenet1d}
SLEEP_WINDOW = 3000  # 30 s epochs at 100 Hz (Sleep-EDF)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("num_classes", [3, 5])
def test_output_shape(name, num_classes):
    model = MODELS[name].create_model(in_ch=1, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 1, SLEEP_WINDOW))
    assert out.shape == (4, num_classes)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", MODELS)
def test_multichannel_input(name):
    # WESAD uses several signals (e.g. PPG, EDA, temperature) as channels.
    model = MODELS[name].create_model(in_ch=3, num_classes=2).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 1920)).shape == (2, 2)


@pytest.mark.parametrize("name", MODELS)
def test_handles_different_window_lengths(name):
    # Global average pooling means the head does not depend on window length.
    model = MODELS[name].create_model().eval()
    with torch.no_grad():
        for length in (500, 1000, 3000):
            assert model(torch.randn(1, 1, length)).shape == (1, 3)


@pytest.mark.parametrize("name", MODELS)
def test_small_enough_for_edge(name):
    params = sum(p.numel() for p in MODELS[name].create_model().parameters())
    assert params < 200_000, f"{name} has {params:,} parameters"


@pytest.mark.parametrize("name", MODELS)
def test_train_loader_builds_each_model(name):
    model = load_model(name, num_classes=3)
    assert isinstance(model, torch.nn.Module)


def test_train_loader_rejects_unknown_model():
    with pytest.raises(ValueError):
        load_model("not_a_model", num_classes=3)


@pytest.mark.parametrize("name", MODELS)
def test_backward_pass(name):
    model = MODELS[name].create_model()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 1, 1000)), torch.randint(0, 3, (8,)))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
