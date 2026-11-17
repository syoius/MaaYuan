"""FIFO upload transport; the scan thread owns diagnostics and progress updates."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Iterator, Literal

from utils import logger
from custom.action.inventory_reporting import UploadSettings
from custom.action.operator_growth_exchange import commit_v3_document, preview_v3_document, summarize_preview

UploadStage = Literal["preview", "commit"]


@dataclass(frozen=True)
class UploadResponse:
    record_id: str
    stage: UploadStage
    payload: dict[str, Any]


@dataclass(frozen=True)
class UploadFailure:
    record_id: str
    stage: UploadStage
    error: Exception


class OperatorScanUploadQueue:
    def __init__(self, settings: UploadSettings, *, commit: bool = True):
        self._settings = settings
        self._commit = commit
        self._documents: Queue[dict[str, Any] | None] = Queue()
        self._results: Queue[UploadResponse | UploadFailure] = Queue()
        self._cancel = Event()
        self._closed = False
        self._skipped = 0
        self._worker = Thread(target=self._run, name="AgentInfoUpload", daemon=True)
        self._worker.start()

    def enqueue(self, document: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("密探后台上报队列已关闭")
        self._documents.put(copy.deepcopy(document))
        logger.bind(console=False).info(
            "AgentInfoCollector: 已加入后台上报队列 "
            f"record_id={document['records'][0]['record_id']!r}"
        )

    def take_results(self) -> Iterator[UploadResponse | UploadFailure]:
        while True:
            try:
                yield self._results.get_nowait()
            except Empty:
                return

    def finish(self, should_stop: Callable[[], bool]) -> None:
        if self._worker.is_alive():
            logger.bind(console=False).info("AgentInfoCollector: 正在等待后台上报队列结束")
        while True:
            if should_stop() and not self._cancel.is_set():
                self._cancel.set()
                logger.info("停止后台上报，跳过排队任务并等待当前请求返回")
            if not self._closed:
                self._closed = True
                self._documents.put(None)
            self._worker.join(timeout=0.1)
            if not self._worker.is_alive():
                break
        if self._skipped:
            logger.info(f"已跳过 {self._skipped} 个后台上报任务，本地记录已保存")

    def _run(self) -> None:
        while True:
            document = self._documents.get()
            if document is None:
                return
            record_id = document["records"][0]["record_id"]
            stages: tuple[UploadStage, ...] = ("preview", "commit") if self._commit else ("preview",)
            for stage in stages:
                if self._cancel.is_set():
                    self._skipped += 1
                    break
                try:
                    if stage == "preview":
                        logger.bind(console=False).info(f"AgentInfoCollector: 后台上报开始 record_id={record_id!r}")
                    send = preview_v3_document if stage == "preview" else commit_v3_document
                    payload = send(document, self._settings.base_url, self._settings.token)
                    detail = summarize_preview(payload) if stage == "preview" else "v3 自动上报 commit 完成"
                    response = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    response = response.replace(self._settings.token, "<redacted>")
                    logger.bind(console=False).info(
                        f"AgentInfoCollector: {detail}, record_id={record_id!r}, response={response[:4000]}"
                    )
                    self._results.put(UploadResponse(record_id, stage, payload))
                except Exception as exc:
                    message = str(exc).replace(self._settings.token, "<redacted>")
                    logger.bind(console=False).warning(
                        "AgentInfoCollector: v3 自动上报失败，文档已保存 "
                        f"record_id={record_id!r}: {message}"
                    )
                    self._results.put(UploadFailure(record_id, stage, exc))
                    break
