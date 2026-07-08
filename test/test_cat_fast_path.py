"""Tests for the fastCatOutDimN fast path in aten::cat (TensorShape.cpp).

The fast path fires when: all inputs are contiguous, same dtype, result is
in Contiguous memory format, and numel < GRAIN_SIZE (32768) or single thread.
fastCatOutDimN handles dim>0 using direct memcpy of contiguous slices.
"""

import torch
import pytest


def _cat_is_bit_exact(tensors, dim):
    """Verify that cat result matches element-wise reference construction."""
    result = torch.cat(tensors, dim=dim)
    # Build expected by indexing (does not go through fastCatOutDimN)
    expected = torch.empty_like(result)
    offset = 0
    for t in tensors:
        slices = [slice(None)] * result.dim()
        slices[dim] = slice(offset, offset + t.size(dim))
        expected[tuple(slices)] = t
        offset += t.size(dim)
    return torch.equal(result, expected)


class TestCatFastPath:
    """Exercises the fastCatOutDimN fast path — cat along dim>0 with
    contiguous same-dtype inputs below GRAIN_SIZE."""

    def test_cat_fast_path_dim1_2d(self):
        a = torch.randn(4, 6)
        b = torch.randn(4, 3)
        assert _cat_is_bit_exact([a, b], dim=1)

    def test_cat_fast_path_dim1_3d(self):
        # KV-cache pattern: cat along seq_len
        a = torch.randn(1, 10, 64)
        b = torch.randn(1, 1, 64)
        assert _cat_is_bit_exact([a, b], dim=1)

    def test_cat_fast_path_dim2_3d(self):
        a = torch.randn(2, 3, 8)
        b = torch.randn(2, 3, 4)
        assert _cat_is_bit_exact([a, b], dim=2)

    def test_cat_fast_path_dim1_4d(self):
        a = torch.randn(1, 12, 1, 64)
        b = torch.randn(1, 4, 1, 64)
        assert _cat_is_bit_exact([a, b], dim=1)

    def test_cat_fast_path_three_inputs(self):
        a = torch.randn(2, 5, 8)
        b = torch.randn(2, 3, 8)
        c = torch.randn(2, 2, 8)
        assert _cat_is_bit_exact([a, b, c], dim=1)

    def test_cat_fast_path_float64(self):
        a = torch.randn(4, 6, dtype=torch.float64)
        b = torch.randn(4, 3, dtype=torch.float64)
        assert _cat_is_bit_exact([a, b], dim=1)

    def test_cat_fast_path_values_correct(self):
        # Use known values so we can check exact output
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        result = torch.cat([a, b], dim=1)
        expected = torch.tensor([[1.0, 2.0, 5.0, 6.0, 7.0],
                                  [3.0, 4.0, 8.0, 9.0, 10.0]])
        assert torch.equal(result, expected)

    def test_cat_fast_path_empty_input(self):
        a = torch.randn(3, 0, 8)
        b = torch.randn(3, 4, 8)
        assert _cat_is_bit_exact([a, b], dim=1)

    def test_cat_fast_path_single_input(self):
        a = torch.randn(2, 5, 8)
        result = torch.cat([a], dim=1)
        assert torch.equal(result, a)

    def test_cat_fast_path_dim3_4d(self):
        a = torch.randn(2, 3, 4, 5)
        b = torch.randn(2, 3, 4, 7)
        assert _cat_is_bit_exact([a, b], dim=3)

    def test_cat_fast_path_output_contiguous(self):
        a = torch.randn(4, 6)
        b = torch.randn(4, 3)
        result = torch.cat([a, b], dim=1)
        assert result.is_contiguous()

    def test_cat_fast_path_int32(self):
        a = torch.randint(0, 100, (3, 4), dtype=torch.int32)
        b = torch.randint(0, 100, (3, 2), dtype=torch.int32)
        assert _cat_is_bit_exact([a, b], dim=1)
