// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename U>
struct DotAccT {
  using type = float;
};
template <>
struct DotAccT<float> {
  using type = float;
};
template <>
struct DotAccT<complex64_t> {
  using type = complex64_t;
};

template <
    typename T,
    typename AccT,
    bool kDoAxpby,
    bool kDoNCBatch,
    const int TPG>
[[kernel, max_total_threads_per_threadgroup(TPG)]] void dot(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    const device T* c [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int& lda [[buffer(5)]],
    const constant int& ldb [[buffer(6)]],
    const constant int& transpose_a [[buffer(7)]],
    const constant int& transpose_b [[buffer(8)]],
    const constant float& alpha [[buffer(9)]],
    const constant float& beta [[buffer(10)]],
    const constant int& batch_ndim [[buffer(11)]],
    const constant int* batch_shape [[buffer(12)]],
    const constant int64_t* a_batch_stride [[buffer(13)]],
    const constant int64_t* b_batch_stride [[buffer(14)]],
    const constant int64_t* c_batch_stride [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Allocate threadgroup memory for inter-simdgroup reduction
  threadgroup AccT tgp_memory[TPG / 32];

  // Compute batch offsets
  if (kDoNCBatch) {
    a += elem_to_loc(tid.z, batch_shape, a_batch_stride, batch_ndim);
    b += elem_to_loc(tid.z, batch_shape, b_batch_stride, batch_ndim);
    if (kDoAxpby) {
      c += elem_to_loc(tid.z, batch_shape, c_batch_stride, batch_ndim);
    }
  } else {
    a += tid.z * a_batch_stride[0];
    b += tid.z * b_batch_stride[0];
    if (kDoAxpby) {
      c += tid.z * c_batch_stride[0];
    }
  }
  out += tid.z;

  // Per-thread accumulation over K with coalesced reads
  AccT acc = 0;
  int a_stride = transpose_a ? lda : 1;
  int b_stride = transpose_b ? 1 : ldb;

  for (int i = lid.x; i < K; i += TPG) {
    acc +=
        static_cast<AccT>(a[i * a_stride]) * static_cast<AccT>(b[i * b_stride]);
  }

  // SIMD reduction
  for (ushort s = 16; s > 0; s >>= 1) {
    acc += simd_shuffle_down(acc, s);
  }

  // Threadgroup reduction
  if (simd_lid == 0) {
    tgp_memory[simd_gid] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid.x == 0) {
    int n_simdgroups = TPG / 32;
    AccT total = 0;
    for (int i = 0; i < n_simdgroups; ++i) {
      total += tgp_memory[i];
    }
    if (kDoAxpby) {
      out[0] = static_cast<T>(
          total * static_cast<AccT>(alpha) +
          static_cast<AccT>(beta) * static_cast<AccT>(c[0]));
    } else {
      out[0] = static_cast<T>(total);
    }
  }
}
