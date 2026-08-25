// Portable replacement for ooz's MSVC-only stdafx.h so it builds with clang
// on macOS (arm64 via sse2neon, x86_64 natively). scripts/build_bin.py copies
// this over stdafx.h in a pinned checkout of https://github.com/powzix/ooz
// (GPL-3.0-or-later, compatible with this project's GPL-2.0-or-later).
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#include <tchar.h>
#include <intrin.h>
#include <Windows.h>
#else
#if defined(__aarch64__) || defined(__arm64__)
#include "sse2neon.h"
#else
#include <immintrin.h>
#endif

#define __forceinline inline __attribute__((always_inline))
#undef __int64
#define __int64 long long

static __forceinline uint32_t _rotl(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

static __forceinline unsigned char _BitScanReverse(unsigned long *index, unsigned long mask) {
  if (mask == 0) return 0;
  *index = 31 - __builtin_clz((unsigned int)mask);
  return 1;
}

static __forceinline unsigned char _BitScanForward(unsigned long *index, unsigned long mask) {
  if (mask == 0) return 0;
  *index = __builtin_ctz((unsigned int)mask);
  return 1;
}

#define _byteswap_ushort __builtin_bswap16
#define _byteswap_ulong __builtin_bswap32
#define _byteswap_uint64 __builtin_bswap64
#endif

typedef unsigned char byte;
typedef unsigned char uint8;
typedef unsigned int uint32;
typedef unsigned __int64 uint64;
typedef signed __int64 int64;
typedef signed int int32;
typedef unsigned short uint16;
typedef signed short int16;
typedef unsigned int uint;
