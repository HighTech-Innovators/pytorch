#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {
namespace {

template <typename scalar_t>
typename std::enable_if_t<!is_reduced_floating_point_v<scalar_t>, void>
multinomial_with_replacement_apply(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<CPUGeneratorImpl>(
      generator, detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  int64_t n_categories = self.size(-1);
  int64_t n_dist = self.dim() > 1 ? self.size(-2) : 1;

  const scalar_t* const self_ptr = self.const_data_ptr<scalar_t>();
  int64_t* const result_ptr = result.data_ptr<int64_t>();

  auto self_stride_0 = self.dim() > 1 ? self.stride(-2) : 0;
  auto self_stride_1 = self.stride(-1);

  auto result_dist_stride_0 = result.dim() > 1 ? result.stride(-2) : 0;
  auto result_dist_stride_1 = result.stride(-1);

  // Fast path for n_sample == 1: skip full CDF allocation and normalization.
  // Validate + compute sum in one pass, then linear scan with early exit.
  if (n_sample == 1) {
    for (const auto i : c10::irange(n_dist)) {
      scalar_t sum = 0;
      for (const auto j : c10::irange(n_categories)) {
        scalar_t val = self_ptr[i * self_stride_0 + j * self_stride_1];
        TORCH_CHECK(
            val >= 0,
            "invalid multinomial distribution (encountering probability entry < 0)");
#if defined(_LIBCPP_VERSION)
        TORCH_CHECK(
            std::isfinite(static_cast<double>(val)),
            "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#else
        TORCH_CHECK(
            std::isfinite(val),
            "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#endif
        sum += val;
      }
      TORCH_CHECK(
          sum > 0,
          "invalid multinomial distribution (sum of probabilities <= 0)");

      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);

      // Linear scan: accumulate running sum, normalize inline, compare.
      // Matches the original semantics: find first j where
      // (running_sum / sum) >= uniform_sample, with last bucket forced to 1.
      scalar_t running_sum = 0;
      int64_t sample_idx = n_categories - 1;
      for (const auto j : c10::irange(n_categories)) {
        running_sum += self_ptr[i * self_stride_0 + j * self_stride_1];
        scalar_t normalized = (j == n_categories - 1)
            ? static_cast<scalar_t>(1)
            : running_sum / sum;
        if (!(static_cast<double>(normalized) < uniform_sample)) {
          sample_idx = j;
          break;
        }
      }
      result_ptr[i * result_dist_stride_0] = sample_idx;
    }
    return;
  }

  /* cumulative probability distribution vector */
  Tensor cum_dist = at::empty({n_categories}, self.options());

  scalar_t* const cum_dist_ptr = cum_dist.data_ptr<scalar_t>();

  auto cum_dist_stride_0 = cum_dist.stride(0);

  for (const auto i : c10::irange(n_dist)) {
    /* Get normalized cumulative distribution from prob distribution */
    scalar_t sum = 0;
    for (const auto j : c10::irange(n_categories)) {
      scalar_t val = self_ptr[i * self_stride_0 + j * self_stride_1];
      TORCH_CHECK(
          val >= 0,
          "invalid multinomial distribution (encountering probability entry < 0)");
// NB: std::isfinite doesn't bode well with libc++ for half datatypes,
// so we manually cast it to a double and perform the check.
#if defined(_LIBCPP_VERSION)
      TORCH_CHECK(
          std::isfinite(static_cast<double>(val)),
          "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#else
      TORCH_CHECK(
          std::isfinite(val),
          "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#endif

      sum += val;
      cum_dist_ptr[j * cum_dist_stride_0] = sum;
    }

    TORCH_CHECK(
        sum > 0,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    /* normalize cumulative probability distribution so that last val is 1
    i.e. doesn't assume original self row sums to one */
    if ((sum > 0) || ((sum < 1.00001) && (sum > 0.99999))) {
      for (const auto j : c10::irange(n_categories)) {
        cum_dist_ptr[j * cum_dist_stride_0] /= sum;
      }
    }

    for (const auto j : c10::irange(n_sample)) {
      /* sample a probability mass from a uniform distribution */
      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);
      /* Do a binary search for the slot in which the prob falls
      ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
      int64_t left_pointer = 0;
      int64_t right_pointer = n_categories;
      /* Make sure the last cumulative distribution bucket sums to 1 */
      cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;

      while (right_pointer - left_pointer > 0) {
        int64_t mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        scalar_t cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
        if (cum_prob < uniform_sample) {
          left_pointer = mid_pointer + 1;
        } else {
          right_pointer = mid_pointer;
        }
      }
      auto sample_idx = left_pointer;

      /* store in result tensor (will be incremented for lua compat by wrapper)
       */
      result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] =
          sample_idx;
    }
  }
}

template <typename scalar_t>
typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, void>
multinomial_with_replacement_apply(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<CPUGeneratorImpl>(
      generator, detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);

  int64_t n_categories = self.size(-1);
  int64_t n_dist = self.dim() > 1 ? self.size(-2) : 1;

  const scalar_t* const self_ptr = self.const_data_ptr<scalar_t>();
  int64_t* const result_ptr = result.data_ptr<int64_t>();

  auto self_stride_0 = self.dim() > 1 ? self.stride(-2) : 0;
  auto self_stride_1 = self.stride(-1);

  auto result_dist_stride_0 = result.dim() > 1 ? result.stride(-2) : 0;
  auto result_dist_stride_1 = result.stride(-1);

  // Fast path for n_sample == 1: skip full CDF allocation and normalization.
  if (n_sample == 1) {
    for (const auto i : c10::irange(n_dist)) {
      float sum = 0;
      for (const auto j : c10::irange(n_categories)) {
        float val = self_ptr[i * self_stride_0 + j * self_stride_1];
        TORCH_CHECK(
            val >= 0,
            "invalid multinomial distribution (encountering probability entry < 0)");
#if defined(_LIBCPP_VERSION)
        TORCH_CHECK(
            std::isfinite(static_cast<double>(val)),
            "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#else
        TORCH_CHECK(
            std::isfinite(val),
            "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#endif
        sum += val;
      }
      TORCH_CHECK(
          sum > 0,
          "invalid multinomial distribution (sum of probabilities <= 0)");

      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);

      float running_sum = 0;
      int64_t sample_idx = n_categories - 1;
      for (const auto j : c10::irange(n_categories)) {
        running_sum += static_cast<float>(
            self_ptr[i * self_stride_0 + j * self_stride_1]);
        float normalized = (j == n_categories - 1)
            ? 1.0f
            : running_sum / sum;
        if (!(static_cast<double>(normalized) < uniform_sample)) {
          sample_idx = j;
          break;
        }
      }
      result_ptr[i * result_dist_stride_0] = sample_idx;
    }
    return;
  }

  /* cumulative probability distribution vector */
  Tensor cum_dist = at::empty({n_categories}, self.options().dtype(kFloat));

  float* const cum_dist_ptr = cum_dist.data_ptr<float>();

  auto cum_dist_stride_0 = cum_dist.stride(0);

  for (const auto i : c10::irange(n_dist)) {
    /* Get normalized cumulative distribution from prob distribution */
    float sum = 0;
    for (const auto j : c10::irange(n_categories)) {
      float val = self_ptr[i * self_stride_0 + j * self_stride_1];
      TORCH_CHECK(
          val >= 0,
          "invalid multinomial distribution (encountering probability entry < 0)");
// NB: std::isfinite doesn't bode well with libc++ for half datatypes,
// so we manually cast it to a double and perform the check.
#if defined(_LIBCPP_VERSION)
      TORCH_CHECK(
          std::isfinite(static_cast<double>(val)),
          "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#else
      TORCH_CHECK(
          std::isfinite(val),
          "invalid multinomial distribution (encountering probability entry = infinity or NaN)");
#endif

      sum += val;
      cum_dist_ptr[j * cum_dist_stride_0] = sum;
    }

    TORCH_CHECK(
        sum > 0,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    /* normalize cumulative probability distribution so that last val is 1
    i.e. doesn't assume original self row sums to one */
    if ((sum > 0) || ((sum < 1.00001) && (sum > 0.99999))) {
      for (const auto j : c10::irange(n_categories)) {
        cum_dist_ptr[j * cum_dist_stride_0] /= sum;
      }
    }

    for (const auto j : c10::irange(n_sample)) {
      /* sample a probability mass from a uniform distribution */
      at::uniform_real_distribution<double> uniform(0, 1);
      double uniform_sample = uniform(gen);
      /* Do a binary search for the slot in which the prob falls
      ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
      int64_t left_pointer = 0;
      int64_t right_pointer = n_categories;
      /* Make sure the last cumulative distribution bucket sums to 1 */
      cum_dist_ptr[(n_categories - 1) * cum_dist_stride_0] = 1;

      while (right_pointer - left_pointer > 0) {
        int64_t mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
        float cum_prob = cum_dist_ptr[mid_pointer * cum_dist_stride_0];
        if (cum_prob < uniform_sample) {
          left_pointer = mid_pointer + 1;
        } else {
          right_pointer = mid_pointer;
        }
      }
      auto sample_idx = left_pointer;

      /* store in result tensor (will be incremented for lua compat by wrapper)
       */
      result_ptr[i * result_dist_stride_0 + j * result_dist_stride_1] =
          sample_idx;
    }
  }
}

void multinomial_with_replacement_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "multinomial", [&] {
        multinomial_with_replacement_apply<scalar_t>(
            result, self, n_sample, gen);
      });
}
} // namespace

REGISTER_DISPATCH(
    multinomial_with_replacement_stub,
    &multinomial_with_replacement_kernel_impl)
} // namespace at::native
