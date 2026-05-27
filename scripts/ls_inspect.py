"""Inspect the Label Studio project: confirm connectivity and learn the real task /
annotation shapes before we patch anything. Read-only.

    uv run python scripts/ls_inspect.py
"""

from __future__ import annotations

import json

from chessvision.data.labelstudio_api import LabelStudioClient, task_image_ref


def _short(obj, n=600):
    s = json.dumps(obj, indent=2, default=str)
    return s if len(s) <= n else s[:n] + " ...(truncated)"


def main() -> int:
    client = LabelStudioClient()
    cfg = client.config
    print(f"LS @ {cfg.url}  project={cfg.project!r}")

    projects = client.list_projects()
    print(f"projects visible: {[(p.get('id'), p.get('title')) for p in projects]}")
    pid = client.resolve_project_id()
    print(f"resolved project id: {pid}")

    # pull a small first page to inspect shapes
    it = client.iter_tasks(pid, page_size=5)
    first = next(it, None)
    if first is None:
        print("no tasks in project")
        return 0

    print("\n=== sample TASK ===")
    print("task id:", first.get("id"))
    print("data keys:", list((first.get("data") or {}).keys()))
    print("image ref:", task_image_ref(first))
    print("annotation count:", len(first.get("annotations", []) or []))
    print("task (trimmed):", _short({k: first.get(k) for k in ("id", "data", "is_labeled")}))

    # full task (with annotations) to see the result/keypoint shape
    full = client.get_task(first["id"])
    anns = full.get("annotations", []) or []
    print("\n=== sample ANNOTATION ===")
    if anns:
        a = anns[0]
        print("annotation id:", a.get("id"), "| was_cancelled:", a.get("was_cancelled"))
        results = a.get("result", []) or []
        froms = {}
        for r in results:
            froms[r.get("from_name")] = froms.get(r.get("from_name"), 0) + 1
        print("result control -> count:", froms)
        corner_rs = [r for r in results if r.get("from_name") == "corners"]
        print("one corner result (trimmed):", _short(corner_rs[0] if corner_rs else "NONE"))
    else:
        print("task has no annotations")

    # rough total
    total = 1 + sum(1 for _ in it)
    print(f"\n(first page+ scanned; >= {total} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
