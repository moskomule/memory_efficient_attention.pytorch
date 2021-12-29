import math

import torch
from torch.utils.checkpoint import checkpoint


@torch.jit.script
def self_attention(query: torch.Tensor,
                   key: torch.Tensor,
                   value: torch.Tensor
                   ) -> torch.Tensor:
    """ A naive O(n^2) complexity implementation of self-attention

    Args:
        query: query of shape BxHxNxD
        key: key of shape BxHxN'xD
        value: value of shape BxHxN'xD

        where B is the batch size, H is the number of heads, N is the sequence length of the query,
        N' is the sequence length of the key and value (can be N), and D is the feature size.

    Returns: output of self-attention of shape BxHxNxD

    """

    score = torch.einsum("bhqd,bhkd->bhqk", query, key)
    score = score.softmax(dim=-1)
    return torch.einsum("bhqv,bhvd->bhqd", score, value)


@torch.jit.script
def _efficient_attention(query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         chunk_size: int = 1,
                         out_of_place: bool = False
                         ) -> torch.Tensor:
    out = torch.empty_like(query)
    for i, query in enumerate(query.split(chunk_size, dim=-2)):
        score = torch.einsum("bhqd,bhkd->bhqk", query, key)
        if out_of_place:
            score = (score - score.amax(dim=-1, keepdim=True).detach()).exp()
        else:
            score -= score.amax(dim=-1, keepdim=True).detach()
            score.exp_()
        num = torch.einsum("bhqv,bhvd->bhqd", score, value)
        div = score.sum(dim=-1, keepdim=True)
        out[:, :, i * chunk_size:(i + 1) * chunk_size, :] = num / div
    return out


def efficient_attention(query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        chunk_size: int = None,
                        checkpointing: bool = False,
                        out_of_place: bool = False
                        ) -> torch.Tensor:
    """ A sub-quadratic complexity implementation of self-attention

    Args:
        query: query of shape BxHxNxD
        key: key of shape BxHxN'xD
        value: value of shape BxHxN'xD
        chunk_size: chunk size to divide the query. If None (default), sqrt(N) is used.
        checkpointing: True to enable checkpointing.
        out_of_place: True to disable inplace operations.

        where B is the batch size, H is the number of heads, N is the sequence length of the query,
        N' is the sequence length of the key and value (can be N), and D is the feature size.

    Returns: output of self-attention of shape BxHxNxD

    """

    if chunk_size is not None and chunk_size > query.size(-2):
        raise RuntimeError("chunk_size is expected to be smaller than the sequence length of the query")

    if chunk_size is None:
        chunk_size = int(math.sqrt(query.size(-2)))

    if checkpointing:
        return checkpoint(_efficient_attention, query, key, value, chunk_size)
    else:
        return _efficient_attention(query, key, value, chunk_size, out_of_place)
