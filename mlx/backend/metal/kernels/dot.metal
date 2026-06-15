// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/dot.h"

#define instantiate_dot(name, itype, acctype, axpby) \
  instantiate_kernel(                                 \
      "dot_" #name "_axpby" #axpby,                 \
      dot,                                             \
      itype,                                           \
      acctype,                                         \
      axpby,                                           \
      256)

#define instantiate_dot_type(name, itype, acctype) \
  instantiate_dot(name, itype, acctype, 0)         \
  instantiate_dot(name, itype, acctype, 1)

instantiate_dot_type(float32, float, float);
instantiate_dot_type(float16, half, float);
instantiate_dot_type(bfloat16, bfloat16_t, float);
instantiate_dot_type(complex64, complex64_t, complex64_t);

#define instantiate_dot_splitk(name, itype, acctype, nc) \
  instantiate_kernel(                                     \
      "dot_splitk_" #name "_nc" #nc,                     \
      dot_splitk,                                          \
      itype,                                               \
      acctype,                                             \
      nc,                                                  \
      256)

#define instantiate_dot_splitk_type(name, itype, acctype) \
  instantiate_dot_splitk(name, itype, acctype, 0)         \
  instantiate_dot_splitk(name, itype, acctype, 1)

instantiate_dot_splitk_type(float32, float, float);
instantiate_dot_splitk_type(float16, half, float);
instantiate_dot_splitk_type(bfloat16, bfloat16_t, float);
instantiate_dot_splitk_type(complex64, complex64_t, complex64_t);

#define instantiate_dot_splitk_accum(name, itype, acctype, nc, axpby) \
  instantiate_kernel(                                                  \
      "dot_splitk_accum_" #name "_nc" #nc "_axpby" #axpby,          \
      dot_splitk_accum,                                                \
      itype,                                                           \
      acctype,                                                         \
      axpby,                                                           \
      nc)

#define instantiate_dot_splitk_accum_type(name, itype, acctype) \
  instantiate_dot_splitk_accum(name, itype, acctype, 0, 0)      \
  instantiate_dot_splitk_accum(name, itype, acctype, 0, 1)      \
  instantiate_dot_splitk_accum(name, itype, acctype, 1, 0)      \
  instantiate_dot_splitk_accum(name, itype, acctype, 1, 1)

instantiate_dot_splitk_accum_type(float32, float, float);
instantiate_dot_splitk_accum_type(float16, half, float);
instantiate_dot_splitk_accum_type(bfloat16, bfloat16_t, float);
instantiate_dot_splitk_accum_type(complex64, complex64_t, complex64_t); // clang-format on
