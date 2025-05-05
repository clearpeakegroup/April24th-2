import importlib

try:
    nvcomp = importlib.import_module('nvcomp')
    NVCOMP_AVAILABLE = True
except ImportError:
    NVCOMP_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

import zstandard as zstd


def decompress_zstd(data: bytes) -> bytes:
    """
    Decompress zstd-compressed data using GPU (nvCOMP) if available, else CPU zstandard.
    Args:
        data (bytes): Compressed data
    Returns:
        bytes: Decompressed data
    Raises:
        RuntimeError: If decompression fails
    """
    if NVCOMP_AVAILABLE and GPU_AVAILABLE:
        # nvCOMP Python bindings (assume nvcomp.decompress is available)
        try:
            # nvCOMP expects device memory, so use cupy
            d_in = cp.asarray(bytearray(data), dtype=cp.uint8)
            d_out = nvcomp.decompress(d_in)
            return cp.asnumpy(d_out).tobytes()
        except Exception as e:
            raise RuntimeError(f"nvCOMP GPU decompression failed: {e}")
    else:
        # CPU fallback
        try:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except Exception as e:
            raise RuntimeError(f"CPU zstd decompression failed: {e}") 