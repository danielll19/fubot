"""Persistent workflow and route-health storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fubot.orchestrator.models import (
    AssignmentRecord,
    ExecutionLogRecord,
    ProviderHealthState,
    TaskRecord,
    WorkflowRecord,
    utc_now,
)
from fubot.utils.helpers import ensure_dir


class WorkflowStore:
    """Stores workflow records under the workspace for restart recovery and audits."""

    def __init__(self, workspace: Path, workflow_dir_name: str = "workflows", health_file: str = "provider-health.json"):
        self.base_dir = ensure_dir(workspace / workflow_dir_name)
        self.health_path = self.base_dir / health_file

    def _workflow_path(self, workflow_id: str) -> Path:
        return self.base_dir / f"{workflow_id}.json"

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def save_workflow(
        self,
        workflow: WorkflowRecord,
        tasks: list[TaskRecord],
        assignments: list[AssignmentRecord],
    ) -> None:
        workflow.updated_at = utc_now()
        payload = {
            "workflow": workflow.to_dict(),
            "tasks": [task.to_dict() for task in tasks],
            "assignments": [assignment.to_dict() for assignment in assignments],
        }
        self._atomic_write_json(self._workflow_path(workflow.id), payload)

    def load_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        path = self._workflow_path(workflow_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_workflows(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            if path == self.health_path or path.name.endswith("-health.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            data["_path"] = str(path)
            items.append(data)
        return items

    def recover_incomplete(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.list_workflows()
            if (item.get("workflow") or {}).get("status") not in {"completed", "cancelled", "failed"}
        ]

    def append_log(
        self,
        workflow: WorkflowRecord,
        tasks: list[TaskRecord],
        assignments: list[AssignmentRecord],
        record: ExecutionLogRecord,
    ) -> None:
        workflow.execution_logs.append(record.to_dict())
        self.save_workflow(workflow, tasks, assignments)

    def update_shared_board(
        self,
        workflow: WorkflowRecord,
        tasks: list[TaskRecord],
        assignments: list[AssignmentRecord],
        key: str,
        value: Any,
    ) -> None:
        workflow.shared_board[key] = value
        self.save_workflow(workflow, tasks, assignments)

    def load_health(self) -> dict[str, Any]:
        if not self.health_path.exists():
            return {}
        try:
            payload = json.loads(self.health_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, Any] = {}
        for provider, state in payload.items():
            if not provider:
                continue
            normalized[provider] = ProviderHealthState.from_dict(provider, state).normalized().to_dict()
        return normalized

    def save_health(self, payload: dict[str, Any]) -> None:
        current = {
            provider: ProviderHealthState.from_dict(provider, state).normalized()
            for provider, state in self.load_health().items()
            if provider
        }
        incoming = {
            provider: ProviderHealthState.from_dict(provider, state).normalized()
            for provider, state in (payload or {}).items()
            if provider
        }
        merged: dict[str, Any] = {}
        for provider in sorted(set(current) | set(incoming)):
            current_state = current.get(provider)
            incoming_state = incoming.get(provider)
            if current_state is None and incoming_state is None:
                continue
            if current_state is None:
                final_state = incoming_state
            elif incoming_state is None:
                final_state = current_state
            else:
                final_state = current_state.merged_with(incoming_state)
            if final_state is None:
                continue
            merged[provider] = final_state.to_dict()
        self._atomic_write_json(self.health_path, merged)
