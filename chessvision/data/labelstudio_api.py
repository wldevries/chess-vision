"""Label Studio REST API client (read tasks, patch corner keypoints).

Lets a tool update the corner data *inside* a Label Studio project programmatically --
no manual clicking in the LS UI. The project (default `chess-vision-2`) already has the
capture images as tasks (its source storage is the `chess` bucket); we look up the task
for an image, replace just the `corners` keypoints on its annotation (leaving the piece
keypoints intact), and PATCH it back. LS then exports the corrected annotation to the
`chess-annotations` bucket, which `sync_captures.py annotations` folds into
`label-studio.json` -- the same path the training stack already reads.

Config (env / .env, gitignored):
    LABEL_STUDIO_URL     e.g. http://workstation:8080
    LABEL_STUDIO_TOKEN   personal access token (LS -> Account & Settings -> Access Token)
    LABEL_STUDIO_PROJECT project title or id (default "chess-vision-2")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv

DEFAULT_PROJECT = "chess-vision-2"


@dataclass(frozen=True)
class LabelStudioConfig:
    url: str
    token: str
    project: str = DEFAULT_PROJECT

    @classmethod
    def from_env(cls, *, load: bool = True) -> "LabelStudioConfig":
        if load:
            load_dotenv()
        url = os.environ.get("LABEL_STUDIO_URL")
        token = os.environ.get("LABEL_STUDIO_TOKEN")
        missing = [n for n, v in (("LABEL_STUDIO_URL", url), ("LABEL_STUDIO_TOKEN", token)) if not v]
        if missing:
            raise RuntimeError(f"missing env var(s): {', '.join(missing)} (see .env.example)")
        return cls(
            url=url.rstrip("/"),
            token=token,
            project=os.environ.get("LABEL_STUDIO_PROJECT", DEFAULT_PROJECT),
        )


class LabelStudioClient:
    """Thin httpx wrapper. `LABEL_STUDIO_TOKEN` is a Personal Access Token (a JWT
    *refresh* token); we exchange it at `/api/token/refresh` for a short-lived access
    token and send that as `Authorization: Bearer`. `_request` re-exchanges once on a
    401 so a long bulk run survives access-token expiry."""

    def __init__(self, config: LabelStudioConfig | None = None, timeout: float = 30.0):
        self.config = config or LabelStudioConfig.from_env()
        self._http = httpx.Client(base_url=self.config.url, timeout=timeout)
        self._refresh_access()

    def _refresh_access(self) -> None:
        # The refresh endpoint takes the PAT in the body; no auth header needed.
        r = self._http.post("/api/token/refresh", json={"refresh": self.config.token})
        r.raise_for_status()
        self._http.headers["Authorization"] = f"Bearer {r.json()['access']}"

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        r = self._http.request(method, path, **kw)
        if r.status_code == 401:
            self._refresh_access()
            r = self._http.request(method, path, **kw)
        return r

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- projects ----------------------------------------------------------
    def list_projects(self) -> list[dict]:
        r = self._request("GET", "/api/projects")
        r.raise_for_status()
        body = r.json()
        return body.get("results", body) if isinstance(body, dict) else body

    def resolve_project_id(self) -> int:
        """Project id from the configured title (or pass-through if it's already numeric)."""
        if str(self.config.project).isdigit():
            return int(self.config.project)
        for p in self.list_projects():
            if p.get("title") == self.config.project:
                return int(p["id"])
        raise RuntimeError(f"project {self.config.project!r} not found at {self.config.url}")

    # ---- tasks / annotations ----------------------------------------------
    def iter_tasks(self, project_id: int, page_size: int = 200) -> Iterator[dict]:
        """Yield every task dict for a project (paginated /api/tasks)."""
        page = 1
        while True:
            r = self._request(
                "GET",
                "/api/tasks",
                params={"project": project_id, "page": page, "page_size": page_size},
            )
            if r.status_code == 404:  # LS returns 404 past the last page
                return
            r.raise_for_status()
            body = r.json()
            tasks = body.get("tasks", body.get("results", [])) if isinstance(body, dict) else body
            if not tasks:
                return
            yield from tasks
            page += 1

    def get_task(self, task_id: int) -> dict:
        r = self._request("GET", f"/api/tasks/{task_id}")
        r.raise_for_status()
        return r.json()

    def create_annotation(self, task_id: int, result: list[dict]) -> dict:
        r = self._request("POST", f"/api/tasks/{task_id}/annotations/", json={"result": result})
        r.raise_for_status()
        return r.json()

    def update_annotation(self, annotation_id: int, result: list[dict]) -> dict:
        r = self._request("PATCH", f"/api/annotations/{annotation_id}/", json={"result": result})
        r.raise_for_status()
        return r.json()

    def export_storages(self, project_id: int) -> list[dict]:
        r = self._request("GET", "/api/storages/export", params={"project": project_id})
        r.raise_for_status()
        return r.json()

    def sync_export_storage(self, storage_id: int, storage_type: str = "s3") -> dict:
        """Force the target (export) cloud storage to push annotations to its bucket.
        Required because the chess-annotations target is not synced automatically.
        The sync path is type-scoped: /api/storages/export/<type>/<id>/sync."""
        r = self._request("POST", f"/api/storages/export/{storage_type}/{storage_id}/sync")
        r.raise_for_status()
        return r.json()


def task_image_ref(task: dict) -> str | None:
    """Best-effort extract the image reference from a task's `data` (field name varies)."""
    data = task.get("data") or {}
    for key in ("image", "img", "url", "image_url"):
        if key in data and isinstance(data[key], str):
            return data[key]
    # fall back: first string value that looks like a path/uri
    for v in data.values():
        if isinstance(v, str) and ("/" in v):
            return v
    return None
