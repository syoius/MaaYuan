"""Bounded, in-memory MaaFW notifications for the active diagnostic session."""
from __future__ import annotations

import math
import threading
import time

from maa.agent.agent_server import AgentServer
from maa.context import ContextEventSink

from custom.action.toolbox_diagnostics import (
    _ALGORITHM, _TRACE_TEXT, _box, _node, _text, boundary,
)

_ACTIVE = None
_EVENTS = {
    "Node.Recognition.Succeeded": ("reco", True),
    "Node.Recognition.Failed": ("reco", False),
    "Node.Action.Succeeded": ("action", True),
    "Node.Action.Failed": ("action", False),
}


class DiagnosticTrace:
    def __init__(self, session):
        global _ACTIVE
        self.session = session
        self.lock = threading.Lock()
        self.started = session.started
        self.closed = False
        self.text_enabled = True
        self.nodes = None
        self.steps = []
        self.failed_step = None
        self.failed_frame = None
        self.failed_context = None
        _ACTIVE = self

    def close(self):
        global _ACTIVE
        with self.lock:
            self.closed = True
            if _ACTIVE is self:
                _ACTIVE = None

    def observe_frame(self, frame):
        with self.lock:
            height, width = frame.shape[:2]
            self.session.client["resolution"] = f"{width}x{height}"
            if self.session.mode == "images":
                self.session.last_frame = frame

    def enable_text(self, enabled):
        with self.lock:
            self.text_enabled = enabled

    def last_failed_node(self, stage):
        with self.lock:
            node = next((step["node"] for step in reversed(self.steps)
                         if step["kind"] == "reco" and not step["hit"] and step["stage"] == stage), "")
            return _node(node, self.session.secrets) if node != "(direct)" else ""

    @boundary
    def record(self, context, msg, details):
        kind, hit = _EVENTS[msg]
        with self.lock:
            if self.closed:
                return
            session = self.session
            if self.nodes is None:
                # Cache even an unavailable resource; never query on every event.
                self.nodes = frozenset()
                try:
                    self.nodes = frozenset(context.tasker.resource.node_list)
                except Exception:
                    pass
            reco = details.get("reco_details")
            reco = reco if kind == "reco" and isinstance(reco, dict) else {}
            name = reco.get("name") or details.get("name")
            node = _node(name, session.secrets) if isinstance(name, str) and name in self.nodes else ""
            elapsed = min(86400000, max(0, int((time.monotonic() - self.started) * 1000)))
            step = {"kind": kind, "node": node or "(direct)", "stage": session.stage,
                    "hit": hit, "n": 1, "t": elapsed}
            if kind == "reco":
                algo = _text(reco.get("algorithm"), _ALGORITHM, session.secrets)
                if algo:
                    step["algo"] = algo
                detail = reco.get("detail")
                detail = detail if isinstance(detail, dict) else {}
                best = detail.get("best")
                if not isinstance(best, dict) or not best:
                    results = detail.get("filtered") or detail.get("all") or []
                    best = results[0] if isinstance(results, list) and results else {}
                best = best if isinstance(best, dict) else {}
                score = best.get("score")
                if type(score) in {int, float} and math.isfinite(score) and -1 <= score <= 100000:
                    step["score"] = round(score, 3)
                box = reco.get("box")
                if not _box(box, session.client["resolution"]):
                    box = best.get("box")
                if _box(box, session.client["resolution"]):
                    step["box"] = list(box)
                text = _text(best.get("text"), _TRACE_TEXT, session.secrets) if algo == "OCR" and self.text_enabled else ""
                if text:
                    step["text"] = text
            if self.steps and all(self.steps[-1][key] == step[key] for key in ("kind", "node", "stage", "hit")):
                previous = self.steps[-1]
                previous["n"] = min(10000, previous["n"] + 1)
                previous["dt"] = elapsed - previous["t"]
                if "score" in step:
                    previous["score"] = max(previous.get("score", -1), step["score"])
                for key in ("algo", "box", "text"):
                    if key in step:
                        previous[key] = step[key]
                step = previous
            else:
                self.steps.append(step)
                if len(self.steps) > 64:
                    self.steps.pop(0)
            if kind == "reco" and not hit and session.mode == "images":
                # Keep the frame reference, not the SDK's original or annotated images.
                self.failed_step = step
                self.failed_frame = session.last_frame
                self.failed_context = context

    def snapshot(self):
        with self.lock:
            result = []
            for step in self.steps:
                clean = dict(step)
                clean["node"] = _node(clean["node"], self.session.secrets) or "(direct)"
                for key, pattern in (("algo", _ALGORITHM), ("text", _TRACE_TEXT)):
                    if key in clean and not _text(clean[key], pattern, self.session.secrets):
                        clean.pop(key)
                if clean.get("algo") != "OCR":
                    clean.pop("text", None)
                if "box" in clean and not _box(clean["box"], self.session.client["resolution"]):
                    clean.pop("box")
                result.append(clean)
            return result

    @boundary
    def failure_image(self):
        with self.lock:
            if self.failed_frame is None:
                return None
            index = next((i for i, step in enumerate(self.steps) if step is self.failed_step), None)
            if index is None:
                # The failure frame remains useful after its trace step is evicted.
                return self.failed_frame, {}
            frame, context, node = self.failed_frame, self.failed_context, self.failed_step["node"]
        meta = {"trace_index": index}
        height, width = frame.shape[:2]
        try:
            if node != "(direct)" and _node(node, self.session.secrets):
                data = context.get_node_data(node)
                roi = data.get("recognition", {}).get("param", {}).get("roi")
                if _box(roi, f"{width}x{height}"):
                    meta["node_roi"] = list(roi)
        except Exception:
            # Missing node data must not discard an otherwise usable failure image.
            pass
        return frame, meta

    def release_frames(self):
        with self.lock:
            self.failed_frame = self.failed_context = None


@AgentServer.context_sink()
class ToolboxDiagnosticSink(ContextEventSink):
    def on_raw_notification(self, context, msg: str, details: dict):
        active = _ACTIVE
        if active is None:
            return
        if msg in _EVENTS:
            active.record(context, msg, details)
