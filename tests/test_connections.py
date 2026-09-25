import copy

import pytest
import torch

from hcfactory import GPT, ModelConfig
from hcfactory.connections import REGISTRY

VARIANTS = [
    ("prenorm", {}),
    ("postnorm", {}),
    ("hc", {"n": 4, "dynamic": False}),
    ("hc", {"n": 4, "dynamic": True}),
    ("mhc", {"n": 4}),
    ("frac", {"m": 2, "dynamic": False}),
    ("frac", {"m": 2, "dynamic": True}),
    ("denseformer", {}),
    ("denseformer", {"dilation": 2}),
    ("muddformer", {}),
    ("muddformer", {"dynamic": False}),
    ("muddformer", {"qkv": False}),
    ("laurel", {"variant": "rw"}),
    ("laurel", {"variant": "lr"}),
    ("laurel", {"variant": "rw_lr"}),
    ("attnres", {}),
    ("attnres", {"block_size": 4}),
    ("mhar", {"heads": 4}),
    ("dar", {}),
    ("dar", {"block": True}),
    ("dar", {"gate": "zero", "heads": 2}),
]

# Variants whose documented initialisation reproduces a pre-norm transformer
# exactly (same body weights -> same logits).
PRENORM_AT_INIT = [
    ("hc", {"n": 4, "dynamic": False}),
    ("hc", {"n": 4, "dynamic": True}),
    ("frac", {"m": 2, "dynamic": False}),
    ("frac", {"m": 2, "dynamic": True}),
    ("denseformer", {}),
    ("muddformer", {}),
    ("muddformer", {"dynamic": False}),
    ("muddformer", {"qkv": False}),
    ("dar", {"gate": "zero"}),
    ("laurel", {"variant": "rw"}),
    ("laurel", {"variant": "lr"}),
    ("laurel", {"variant": "rw_lr"}),
]


def make(conn, kw, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=97, n_layer=4, n_head=4, d_model=64, max_seq_len=32,
                      connection=conn, connection_kwargs=kw)
    return GPT(cfg)


def test_registry_covered():
    assert {c for c, _ in VARIANTS} == set(REGISTRY)


@pytest.mark.parametrize("conn,kw", VARIANTS)
def test_forward_backward(conn, kw):
    model = make(conn, kw)
    x = torch.randint(0, 97, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 97)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    # Every connection parameter must receive gradient (catches dead params).
    for name, p in model.connection.named_parameters():
        assert p.grad is not None, name


@pytest.mark.parametrize("conn,kw", VARIANTS)
def test_causal(conn, kw):
    model = make(conn, kw).eval()
    x = torch.randint(0, 97, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 97
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-5)


@pytest.mark.parametrize("conn,kw", PRENORM_AT_INIT)
def test_equivalent_to_prenorm_at_init(conn, kw):
    base = make("prenorm", {}).eval()
    model = make(conn, kw, seed=1).eval()
    body = {k: v for k, v in base.state_dict().items() if not k.startswith("connection.")}
    model.load_state_dict(body, strict=False)
    x = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        a, _ = base(x)
        b, _ = model(x)
    # HC collapses n identical streams by summing (n*h); the final RMSNorm
    # removes the scale up to its epsilon, hence the looser tolerance.
    tol = 5e-3 if conn == "hc" else 1e-4
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


def test_mhar_one_head_is_attnres():
    a = make("attnres", {})
    b = make("mhar", {"heads": 1})
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    x = torch.randint(0, 97, (2, 16))
    torch.testing.assert_close(a(x)[0], b(x)[0])
