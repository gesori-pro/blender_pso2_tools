// Minimal OodleLZ-compatible ABI on top of ooz, in the style of linoodle.
// ZamboniLib P/Invokes oo2core_8_win64_ with the official Oodle signatures to
// decompress NGS ICE archives; on macOS this shim, built as
// liboo2core_8_win64_.dylib, answers those imports using ooz's decompressor.
// The add-on only reads ICE archives, so compression is unsupported and
// reports failure (0) in the official API's convention.
#include "stdafx.h"

extern "C" int Kraken_Decompress(const byte *src, size_t src_len, byte *dst, size_t dst_len);

#define OOZ_EXPORT extern "C" __attribute__((visibility("default")))

OOZ_EXPORT long long OodleLZ_Decompress(
    const unsigned char *buffer, long long bufferSize,
    unsigned char *result, long long outputBufferSize,
    int, int, int, long long, long long, long long, long long, long long,
    long long, int) {
  if (!buffer || !result || bufferSize <= 0 || outputBufferSize < 0) {
    return 0;
  }

  int outbytes = Kraken_Decompress(buffer, (size_t)bufferSize, result,
                                   (size_t)outputBufferSize);
  return outbytes < 0 ? 0 : outbytes;
}

OOZ_EXPORT unsigned long long OodleLZ_Compress(int, const unsigned char *,
                                               long long, unsigned char *, int,
                                               void *, long long,
                                               unsigned long long, void *,
                                               long long) {
  return 0;
}

OOZ_EXPORT long long OodleLZ_GetCompressedBufferSizeNeeded(long long bufferSize) {
  return bufferSize + 274 * ((bufferSize + 0x3FFFF) / 0x40000);
}
