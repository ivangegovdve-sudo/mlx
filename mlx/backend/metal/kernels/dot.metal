// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/dot.h"

#define instantiate_dot(name, itype, acctype, nc, axpby) \
  instantiate_kernel(                                    \
      "dot_" #name "_nc" #nc "_axpby" #axpby,          \
      dot,                                               \
      itype,                                             \
      acctype,                                           \
      axpby,                                             \
      nc,                                                \
      256)

#define instantiate_dot_type(name, itype, acctype) \
  instantiate_dot(name, itype, acctype, 0, 0)      \
  instantiate_dot(name, itype, acctype, 0, 1)      \
  instantiate_dot(name, itype, acctype, 1, 0)      \
  instantiate_dot(name, itype, acctype, 1, 1)

instantiate_dot_type(float32, float, float);
instantiate_dot_type(float16, half, float);
instantiate_dot_type(bfloat16, bfloat16_t, float);
instantiate_dot_type(complex64, complex64_t, complex64_t); // clang-format on
