"""Dormant local evidence adapter for a future telemetry-capable MFAAvalonia.

Activation checklist (do not enable until the packaged GUI is verified):
1. Configure interface.json telemetry.sentry with the project's real DSN and rates.
2. Set 密探采集诊断配置.attach.failure_diagnostics to true in BOTH resource bundles.
3. Verify the GUI's user-consent switch, debug-build exclusion and attachment delivery.

This module has no transport or Sentry SDK. It writes only after a failed scan;
successful scans with missing values are not converted into failures. The GUI
owns consent, sampling and upload. on_error images and .log files match upstream
MFAAvalonia TaskDiagnostics.cs (098fe848c5d599ff9a8ca1b2bcc6d27c1a19c72d).
Screenshot headers are masked, not guaranteed to be fully anonymized.
"""
import json
from pathlib import Path
import time
from uuid import uuid4

import cv2

from utils import logger


class ScanFailureDiagnostics:
    MAX_IMAGE_BYTES = 1024 * 1024

    def __init__(self, mode, game, root=None):
        self.root = root if root is not None else Path(__file__).resolve().parents[3] / "debug/on_error/operator_scan"
        self.mode = mode if mode in ("normal", "enhanced") else "unknown"
        self.game = game if game in ("代号鸢", "如鸢") else "unknown"
        self.stage = "initialize"
        self.started = time.monotonic()
        self.last_frame = None
        self.main_frame = None
        self.missing_fields = []
        self.identity_confirmed = None
        self.saved = False

    def observe_main(self, image, stats, identity_confirmed):
        self.main_frame = image
        self.missing_fields = [key for key in ("life", "attack", "level", "cultivation") if stats.get(key) is None]
        self.identity_confirmed = bool(identity_confirmed)

    def save_failure(self, reason):
        if self.saved:
            return
        self.saved = True
        # Fixed reason codes only; exception messages and runtime parameters may contain credentials.
        reason = reason if reason in ("task_failed", "exception") else "exception"
        metadata = {
            "schema_version": 1, "reason": reason, "scan_mode": self.mode, "game": self.game,
            "stage": self.stage if self.stage in ("initialize", "main", "detail", "huaji", "disc") else "unknown",
            "elapsed_ms": round((time.monotonic() - self.started) * 1000),
            "missing_fields": self.missing_fields, "identity_confirmed": self.identity_confirmed,
            "image_source": "previously_observed_frames_not_fresh_failure_capture", "images": [],
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            stem = f"operator-scan-{uuid4().hex}"
            seen = set()
            for label, frame in (("main", self.main_frame), ("last", self.last_frame)):
                if frame is None or id(frame) in seen:
                    continue
                seen.add(id(frame))
                if len(frame.shape) != 3 or frame.shape[0] < 2 or frame.shape[1] < 2:
                    continue
                # Preserve coordinates for offline replay without retaining the top toolbar.
                masked_top = round(frame.shape[0] * 200 / 1280)
                masked = frame.copy()
                masked[:masked_top] = 0
                ok, encoded = cv2.imencode(".jpg", masked, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok or encoded.nbytes > self.MAX_IMAGE_BYTES:
                    metadata.setdefault("omitted_images", []).append(label)
                    continue
                filename = f"{stem}-{label}.jpg"
                (self.root / filename).write_bytes(encoded.tobytes())
                metadata["images"].append({"file": filename, "masked_top": masked_top,
                                           "original_size": [frame.shape[1], frame.shape[0]]})
            (self.root / f"{stem}.log").write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
            logger.info(f"AgentInfoCollector: 已保存本地失败诊断 {stem}，上传由 GUI 授权与采样控制")
        except Exception as exc:
            # Diagnostics must never replace the original failure or trigger another screenshot/click.
            logger.warning(f"AgentInfoCollector: 本地失败诊断写入失败 ({type(exc).__name__})")
