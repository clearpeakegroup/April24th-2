import pytest
import zstandard as zstd
from backend.data_ingestion.decompression import decompress_zstd

@pytest.mark.parametrize("payload", [b"test data", b"1234567890", b"", b"\x00\x01\x02\x03"])
def test_decompress_zstd_cpu(payload):
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(payload)
    decompressed = decompress_zstd(compressed)
    assert decompressed == payload

# Optionally test GPU path if nvcomp+cupy are available
def test_decompress_zstd_gpu_available():
    try:
        import nvcomp, cupy
        cctx = zstd.ZstdCompressor()
        payload = b"gpu test data"
        compressed = cctx.compress(payload)
        decompressed = decompress_zstd(compressed)
        assert decompressed == payload
    except ImportError:
        pytest.skip("nvcomp or cupy not available") 