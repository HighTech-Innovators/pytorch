#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/NumericUtils.h>
#include <numeric>

namespace at::native {

#ifdef CPU_CAPABILITY
inline namespace CPU_CAPABILITY {
#else
inline namespace DEFAULT {
#endif

// Core topk loop, shared between CPU and QuantizedCPU
template <typename scalar_t, typename accscalar_t>
void topk_impl_loop(
    const int64_t mode_values_stride,
    const int64_t mode_indices_stride,
    const int64_t tmp_values_stride,
    const int64_t k,
    const int64_t dim_size,
    const bool largest,
    const bool sorted,
    char** data, const int64_t* strides, const int64_t n) {

  // If k is zero, then output values and indices are empty tensors
  // So iterating over other dims is pointless
  if (k == 0) {
    return;
  }

  // Fast path: contiguous input — sort an index vector instead of
  // a vector of (value, index) pairs.  This avoids copying all
  // dim_size values into the pair vector and halves cache pressure
  // during partial_sort (indices are 8 bytes vs 16 for the pair).
  if (tmp_values_stride == 1) {
    std::vector<int64_t> idx(dim_size);
    for (const auto i : c10::irange(n)) {
      TensorAccessor<scalar_t, 1> mode_values(
          reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
          &k, &mode_values_stride);
      TensorAccessor<int64_t, 1> mode_indices(
          reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
          &k, &mode_indices_stride);
      const scalar_t* input_ptr =
          reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);

      std::iota(idx.begin(), idx.end(), int64_t{0});

      auto n_2 = dim_size;
      auto use_partial_sort = k * 64 <= n_2;

      // we want nan to be sorted as top for numpy compatibility
      if (use_partial_sort) {
        if (largest) {
          std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [input_ptr](int64_t a, int64_t b) -> bool {
              accscalar_t va = input_ptr[a], vb = input_ptr[b];
              return (_isnan<accscalar_t>(va) && !_isnan<accscalar_t>(vb)) || (va > vb);
            });
        } else {
          std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [input_ptr](int64_t a, int64_t b) -> bool {
              accscalar_t va = input_ptr[a], vb = input_ptr[b];
              return (!_isnan<accscalar_t>(va) && _isnan<accscalar_t>(vb)) || (va < vb);
            });
        }
      } else {
        if (largest) {
          std::nth_element(idx.begin(), idx.begin() + k - 1, idx.end(),
            [input_ptr](int64_t a, int64_t b) -> bool {
              accscalar_t va = input_ptr[a], vb = input_ptr[b];
              return (_isnan<accscalar_t>(va) && !_isnan<accscalar_t>(vb)) || (va > vb);
            });
          if (sorted) {
            std::sort(idx.begin(), idx.begin() + k - 1,
              [input_ptr](int64_t a, int64_t b) -> bool {
                accscalar_t va = input_ptr[a], vb = input_ptr[b];
                return (_isnan<accscalar_t>(va) && !_isnan<accscalar_t>(vb)) || (va > vb);
              });
          }
        } else {
          std::nth_element(idx.begin(), idx.begin() + k - 1, idx.end(),
            [input_ptr](int64_t a, int64_t b) -> bool {
              accscalar_t va = input_ptr[a], vb = input_ptr[b];
              return (!_isnan<accscalar_t>(va) && _isnan<accscalar_t>(vb)) || (va < vb);
            });
          if (sorted) {
            std::sort(idx.begin(), idx.begin() + k - 1,
              [input_ptr](int64_t a, int64_t b) -> bool {
                accscalar_t va = input_ptr[a], vb = input_ptr[b];
                return (!_isnan<accscalar_t>(va) && _isnan<accscalar_t>(vb)) || (va < vb);
              });
          }
        }
      }

      for (const auto j : c10::irange(k)) {
        mode_values[j] = input_ptr[idx[j]];
        mode_indices[j] = idx[j];
      }
    }
    return;
  }

  // Fallback: non-contiguous input — original pair-vector path
  using elem_t = std::pair<accscalar_t, int64_t>;
  std::vector<elem_t> queue(dim_size);
  for (const auto i : c10::irange(n)) {
    TensorAccessor<scalar_t, 1> mode_values(
        reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
        &k, &mode_values_stride);
    TensorAccessor<int64_t, 1> mode_indices(
        reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
        &k, &mode_indices_stride);
    TensorAccessor<const scalar_t, 1> tmp_values(
        reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
        &dim_size, &tmp_values_stride);

    auto n_2 = dim_size;
    auto use_partial_sort = k * 64 <= n_2;

    for (const auto j : c10::irange(n_2)) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
    }

    // we want nan to be sorted as top for numpy compatibility
    if (use_partial_sort) {
      if (largest) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
      } else {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
      }
    } else {
      if (largest) {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k - 1,
            [](const elem_t& x, const elem_t& y) -> bool {
              return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
            });
        }
      } else {
        std::nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k -1,
            [](const elem_t& x, const elem_t& y) -> bool {
              return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
            });
        }
      }
    }

    for (const auto j : c10::irange(k)) {
      mode_values[j] = queue[j].first;
      mode_indices[j] = queue[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
