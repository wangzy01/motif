"""Process-local guards for native dataline libraries under multi-rank startup."""

from __future__ import annotations

import threading

PARQUET_NATIVE_LOCK = threading.RLock()


def configure_pyarrow_single_thread() -> None:
    """Best-effort clamp for PyArrow's native CPU/IO thread pools."""
    try:
        import pyarrow as pa

        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except Exception:
        return
