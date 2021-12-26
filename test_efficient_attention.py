import pytest
import torch

from efficient_attention import efficient_attention, self_attention


@pytest.mark.parametrize('out_of_place', [True, False])
@pytest.mark.parametrize('chunk_size', [None, 1, 2, 4, 8])
def test_sanity_check(chunk_size, out_of_place):
    q, k, v = [torch.randn(2, 4, 64, 8) for _ in range(3)]
    assert torch.allclose(self_attention(q, k, v), efficient_attention(q, k, v, chunk_size=chunk_size,
                                                                       out_of_place=out_of_place),
                          rtol=1e-4, atol=1e-6)


@pytest.fixture
def qkv():
    return torch.nn.Linear(8, 3 * 8)


@pytest.mark.parametrize('out_of_place', [True, False])
@pytest.mark.parametrize('checkpointing', [True, False])
def test_autograd(qkv, checkpointing, out_of_place):
    input = torch.randn(2, 4, 64, 8)
    q, k, v = qkv(input).chunk(3, dim=-1)
    out = efficient_attention(q, k, v, checkpointing=checkpointing, out_of_place=out_of_place)
    out.sum().backward()
    eff_grad = qkv.weight.grad.clone()
    qkv.zero_grad()

    q, k, v = qkv(input).chunk(3, dim=-1)
    out = self_attention(q, k, v)
    out.sum().backward()
    sl_grad = qkv.weight.grad.clone()
    qkv.zero_grad()

    assert torch.allclose(eff_grad, sl_grad, rtol=1e-4, atol=1e-5)
