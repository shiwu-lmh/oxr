import asyncio
from typing import Dict, Any, Optional


def _build_task_payload(task_id: str) -> Dict[str, Any]:
    return {
        "record_id": None,
        "task_id": task_id,
        "status": "queuing",
        "file_name": None,
        "result": {},
        "error": None,
        "metadata": {},
        "progress": {
            "stage": "queued",
            "label": "等待处理",
            "percent": 0,
            "detail": "任务已提交，等待解析服务。",
        },
    }


class MemoryTaskStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create(self, task_id: str, file_name: Optional[str] = None) -> None:
        async with self._lock:
            self._store[task_id] = _build_task_payload(task_id)
            self._store[task_id]["file_name"] = file_name

    async def set_processing(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._store:
                self._store[task_id]["status"] = "processing"

    async def set_progress(self, task_id: str, progress: Dict[str, Any]) -> None:
        async with self._lock:
            if task_id in self._store:
                self._store[task_id]["status"] = "processing"
                self._store[task_id]["progress"] = dict(progress)

    async def set_success(self, task_id: str, result: Dict[str, Any]) -> None:
        async with self._lock:
            if task_id in self._store:
                self._store[task_id].update(
                    status="success",
                    file_name=result.get("file_name", self._store[task_id].get("file_name")),
                    result=result.get("result", {}),
                    metadata=result.get("metadata", {}),
                    error=None,
                    progress={
                        "stage": "complete",
                        "label": "解析完成",
                        "percent": 100,
                        "detail": "识别结果已生成。",
                    },
                )

    async def set_failed(self, task_id: str, error: str) -> None:
        async with self._lock:
            if task_id in self._store:
                self._store[task_id].update(
                    status="failed",
                    error=error,
                    progress={
                        **self._store[task_id].get("progress", {}),
                        "stage": "failed",
                        "label": "解析失败",
                    },
                )

    async def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._store.get(task_id)
