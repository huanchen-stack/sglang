"""Request lifecycle tracing for rollout precision experiments.

This module is intentionally lightweight and gated by environment variables so
normal serving does not pay for JSON serialization or file IO.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


class RolloutLifecycleTracer:
    """Append-only JSONL writer for scheduler request phase spans."""

    def __init__(self, path: Optional[str], metadata: dict[str, Any]):
        self.path = path
        self.metadata = metadata
        self._file = None

        if path:
            trace_path = Path(path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = trace_path.open("a", buffering=1)
            self.write("trace_start", timestamp=time.perf_counter())

    @classmethod
    def from_env(
        cls,
        *,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        attn_tp_rank: int,
        attn_cp_rank: int,
        dp_rank: Optional[int],
        model_path: str,
        dtype: str,
        quantization: Optional[str],
        tp_size: int,
    ) -> "RolloutLifecycleTracer":
        # Only emit from the scheduler rank that receives tokenizer traffic.
        if pp_rank != 0 or attn_tp_rank != 0 or attn_cp_rank != 0:
            return cls(None, {})

        trace_file = os.getenv("SGLANG_ROLLOUT_TRACE_FILE")
        trace_dir = os.getenv("SGLANG_ROLLOUT_TRACE_DIR")
        if not trace_file and trace_dir:
            dp_part = "none" if dp_rank is None else str(dp_rank)
            name = (
                f"scheduler_gpu{gpu_id}_tp{tp_rank}_pp{pp_rank}"
                f"_dp{dp_part}_pid{os.getpid()}.jsonl"
            )
            trace_file = str(Path(trace_dir) / name)

        metadata = {
            "gpu_id": gpu_id,
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "attn_tp_rank": attn_tp_rank,
            "attn_cp_rank": attn_cp_rank,
            "dp_rank": dp_rank,
            "model_path": model_path,
            "dtype": dtype,
            "quantization": quantization,
            "tp_size": tp_size,
            "pid": os.getpid(),
        }
        return cls(trace_file, metadata)

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def close(self) -> None:
        if self._file is not None:
            self.write("trace_end", timestamp=time.perf_counter())
            self._file.close()
            self._file = None

    def write(
        self,
        event: str,
        *,
        timestamp: Optional[float] = None,
        req: Optional[Any] = None,
        phase: Optional[str] = None,
        batch_id: Optional[int] = None,
        batch_size: Optional[int] = None,
        forward_mode: Optional[str] = None,
        **extra: Any,
    ) -> None:
        if self._file is None:
            return

        record = {
            "event": event,
            "ts": time.perf_counter() if timestamp is None else timestamp,
            **self.metadata,
        }
        if req is not None:
            record.update(
                {
                    "rid": req.rid,
                    "input_len": len(req.origin_input_ids),
                    "output_len": len(req.output_ids),
                    "max_new_tokens": getattr(req.sampling_params, "max_new_tokens", None),
                    "is_chunked": getattr(req, "is_chunked", 0),
                    "is_retracted": getattr(req, "is_retracted", False),
                }
            )
        if phase is not None:
            record["phase"] = phase
        if batch_id is not None:
            record["batch_id"] = batch_id
        if batch_size is not None:
            record["batch_size"] = batch_size
        if forward_mode is not None:
            record["forward_mode"] = forward_mode
        record.update({k: v for k, v in extra.items() if v is not None})

        self._file.write(json.dumps(record, sort_keys=True) + "\n")
