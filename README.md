# memory_efficient_attention.pytorch ![pytest](https://github.com/moskomule/memory_efficient_attention.pytorch/workflows/pytest/badge.svg)

A human-readable PyTorch implementation of "Self-attention Does Not Need O(n^2) Memory" (Rabe&Staats'21).

```python

def efficient_attention(query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        chunk_size: int = None,
                        checkpointing: bool = False,
                        out_of_place: bool = False
                        ) -> torch.Tensor:
    """ A sub-square complexity implementation of self-attention

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
    ...
```