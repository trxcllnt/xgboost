/*!
 * Copyright 2022 by XGBoost Contributors
 * \file common.h
 * \brief cuda pinned allocator for usage with thrust containers
 */

#pragma once

#include <cstddef>
#include <limits>

#include "common.h"

namespace xgboost {
namespace common {
namespace cuda {

// \p pinned_allocator is a CUDA-specific host memory allocator
//  that employs \c cudaMallocHost for allocation.
//
// This implementation is ported from the experimental/pinned_allocator
// that Thrust used to provide.
//
//  \see https://en.cppreference.com/w/cpp/memory/allocator
template <typename T>
class pinned_allocator;

template <>
class pinned_allocator<void> {
 public:
  using value_type      = void;            ///< The type of the elements in the allocator
  using pointer         = void*;           ///< The type returned by address() / allocate()
  using const_pointer   = const void*;     ///< The type returned by address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  template <typename U>
  struct rebind {
    using other = pinned_allocator<U>;  ///< The rebound type
  };
};


template <typename T>
class pinned_allocator {
 public:
  using value_type      = T;               ///< The type of the elements in the allocator
  using pointer         = T*;              ///< The type returned by address() / allocate()
  using const_pointer   = const T*;        ///< The type returned by address()
  using reference       = T&;              ///< The parameter type for address()
  using const_reference = const T&;        ///< The parameter type for address()
  using size_type       = std::size_t;     ///< The type used for the size of the allocation
  using difference_type = std::ptrdiff_t;  ///< The type of the distance between two pointers

  template <typename U>
  struct rebind {
    using other = pinned_allocator<U>;  ///< The rebound type
  };

  XGBOOST_DEVICE inline pinned_allocator() {}
  XGBOOST_DEVICE inline ~pinned_allocator() {}
  XGBOOST_DEVICE inline pinned_allocator(pinned_allocator const&) {}
  template <typename U>
  XGBOOST_DEVICE inline pinned_allocator(pinned_allocator<U> const&) {}

  XGBOOST_DEVICE inline pointer address(reference r) { return &r; }
  XGBOOST_DEVICE inline const_pointer address(const_reference r) { return &r; }

  inline pointer allocate(size_type cnt, const_pointer = 0) {
    if (cnt > this->max_size()) { throw std::bad_alloc(); }  // end if

    pointer result(0);
    dh::safe_cuda(cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  inline void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFreeHost(p)); }

  inline size_type max_size() const { return (std::numeric_limits<size_type>::max)() / sizeof(T); }

  XGBOOST_DEVICE inline bool operator==(pinned_allocator const& x) const { return true; }

  XGBOOST_DEVICE inline bool operator!=(pinned_allocator const& x) const {
    return !operator==(x);
  }
};
}  // namespace cuda
}  // namespace common
}  // namespace xgboost
