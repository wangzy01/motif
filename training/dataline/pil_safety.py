"""Process-local guard for PIL/libjpeg/libpng decode and resize calls.

The ARB item queue uses multiple producer threads per rank. Under 2x8 startup
load, native image codecs/resizers have shown SIGSEGVs while several producer
threads decode/resize concurrently. A single process-local lock preserves all
image math and ordering semantics while keeping native image operations out of
thread races inside one Python process.
"""

from __future__ import annotations

import threading


PIL_NATIVE_LOCK = threading.RLock()
