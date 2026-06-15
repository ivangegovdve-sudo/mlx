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

template <typename T, typename AccT, bool kDoAxpby, const int TPG>
[[kernel, max_total_threads_per_threadgroup(TPG)]] void dot(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    const device T* c [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    const constant int64_t& a_batch_stride [[buffer(5)]],
    const constant int64_t& b_batch_stride [[buffer(6)]],
    const constant int64_t& c_batch_stride [[buffer(7)]],
    const constant float& alpha [[buffer(8)]],
    const constant float& beta [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Allocate threadgroup memory for inter-simdgroup reduction
  threadgroup AccT tgp_memory[TPG / 32];

  // Batch offsets
  a += tid.z * a_batch_stride;
  b += tid.z * b_batch_stride;
  out += tid.z;
  if (kDoAxpby) {
    c += tid.z * c_batch_stride;
  }

  // Per-thread accumulation over K with vectorized contiguous reads
  constexpr int N = 16 / sizeof(T);
  AccT acc = 0;

  int i = lid.x * N;
  for (; i + N <= K; i += TPG * N) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      acc += static_cast<AccT>(a[i + j]) * static_cast<AccT>(b[i + j]);
    }
  }
  // Scalar tail
  for (; i < K; ++i) {
    acc += static_cast<AccT>(a[i]) * static_cast<AccT>(b[i]);
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

template <typename T, typename AccT, bool kDoNCBatch, const int TPG>
[[kernel, max_total_threads_per_threadgroup(TPG)]] void dot_splitk(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    device AccT* C_split [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& lda [[buffer(4)]],
    const constant int& ldb [[buffer(5)]],
    const constant int& transpose_a [[buffer(6)]],
    const constant int& transpose_b [[buffer(7)]],
    const constant int& split_k_partitions [[buffer(8)]],
    const constant int& partition_size [[buffer(9)]],
    const constant int& batch_ndim [[buffer(10)]],
    const constant int* batch_shape [[buffer(11)]],
    const constant int64_t* a_batch_stride [[buffer(12)]],
    const constant int64_t* b_batch_stride [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  // Allocate threadgroup memory for inter-simdgroup reduction
  threadgroup AccT tgp_memory[TPG / 32];

  int partition = tid.x;
  int batch = tid.z;

  // Compute batch offsets
  if (kDoNCBatch) {
    a += elem_to_loc(batch, batch_shape, a_batch_stride, batch_ndim);
    b += elem_to_loc(batch, batch_shape, b_batch_stride, batch_ndim);
  } else {
    a += batch * a_batch_stride[0];
    b += batch * b_batch_stride[0];
  }

  // Compute partition range
  int k_start = partition * partition_size;
  int k_end = k_start + partition_size < K ? k_start + partition_size : K;

  // Per-thread accumulation over partition with coalesced reads
  AccT acc = 0;
  int a_stride = transpose_a ? lda : 1;
  int b_stride = transpose_b ? 1 : ldb;

  for (int i = k_start + lid.x; i < k_end; i += TPG) {
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
    C_split[batch * split_k_partitions + partition] = total;
  }
}

template <typename T, typename AccT, bool kDoAxpby, bool kDoNCBatch>
[[kernel]] void dot_splitk_accum(
    const device AccT* C_split [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device T* c [[buffer(2)]],
    const constant int& split_k_partitions [[buffer(3)]],
    const constant float& alpha [[buffer(4)]],
    const constant float& beta [[buffer(5)]],
    const constant int& batch_ndim [[buffer(6)]],
    const constant int* batch_shape [[buffer(7)]],
    const constant int64_t* c_batch_stride [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;

  int batch = tid.z;

  // Compute batch offsets
  if (kDoNCBatch) {
    c += elem_to_loc(batch, batch_shape, c_batch_stride, batch_ndim);
  } else {
    c += batch * c_batch_stride[0];
  }
  out += batch;
  C_split += batch * split_k_partitions;

  // Reduce partial sums
  AccT total = 0;
  for (int i = 0; i < split_k_partitions; ++i) {
    total += C_split[i];
  }

  if (kDoAxpby) {
    out[0] = static_cast<T>(
        total * static_cast<AccT>(alpha) +
        static_cast<AccT>(beta) * static_cast<AccT>(c[0]));
  } else {
    out[0] = static_cast<T>(total);
  }
}
