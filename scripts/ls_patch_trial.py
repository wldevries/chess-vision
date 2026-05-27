"""Single-task PATCH trial (safe: restores afterward).

Confirms the core mechanism before building the bulk UI:
  - we can PATCH the corner keypoints on an existing annotation,
  - it keeps the piece keypoints intact,
  - the task stays completed (is_labeled stays true -> no re-review queue),
  - and we can map a task back to its capture (session/filename) via the fileuri.
Then it restores the original corners so nothing is left changed.
"""

from __future__ import annotations

import base64
import copy

from chessvision.data.labelstudio_api import LabelStudioClient, task_image_ref


def decode_fileuri(image_ref: str) -> str:
    # /tasks/<id>/resolve/?fileuri=<base64 of s3://...>
    if "fileuri=" not in image_ref:
        return image_ref
    b64 = image_ref.split("fileuri=", 1)[1].split("&", 1)[0]
    pad = "=" * (-len(b64) % 4)
    return base64.urlsafe_b64decode(b64 + pad).decode("utf-8", "replace")


def main() -> int:
    c = LabelStudioClient()
    pid = c.resolve_project_id()

    # find the first task that has a non-cancelled annotation with corner keypoints
    target = None
    scanned = 0
    for t in c.iter_tasks(pid):
        scanned += 1
        full = c.get_task(t["id"])
        for a in full.get("annotations", []) or []:
            if a.get("was_cancelled"):
                continue
            if any(r.get("from_name") == "corners" for r in a.get("result", []) or []):
                target = (full, a)
                break
        if target:
            break
    if not target:
        print(f"scanned {scanned} tasks, none with a corner annotation")
        return 1

    task, ann = target
    image_ref = task_image_ref(task)
    print(f"task {task['id']}  is_labeled={task.get('is_labeled')}")
    print(f"  maps to: {decode_fileuri(image_ref)}")
    print(f"  annotation {ann['id']}  was_cancelled={ann.get('was_cancelled')}")
    original = ann["result"]
    corner_rs = [r for r in original if r.get("from_name") == "corners"]
    piece_n = sum(1 for r in original if r.get("from_name") == "pieces")
    print(f"  result: {len(corner_rs)} corners + {piece_n} pieces")
    print("  corner x before:", [round(r["value"]["x"], 3) for r in corner_rs])

    # PATCH: shift each corner x by +0.5%, keep everything else
    modified = copy.deepcopy(original)
    for r in modified:
        if r.get("from_name") == "corners":
            r["value"]["x"] = r["value"]["x"] + 0.5
    c.update_annotation(ann["id"], modified)

    after = c.get_task(task["id"])
    after_ann = next(a for a in after["annotations"] if a["id"] == ann["id"])
    after_corners = [r for r in after_ann["result"] if r.get("from_name") == "corners"]
    after_pieces = sum(1 for r in after_ann["result"] if r.get("from_name") == "pieces")
    print("\nAFTER PATCH:")
    print(f"  is_labeled={after.get('is_labeled')}  (should still be True)")
    print(f"  corners={len(after_corners)} pieces={after_pieces}  (pieces must be {piece_n})")
    print("  corner x after:", [round(r["value"]["x"], 3) for r in after_corners])

    # restore
    c.update_annotation(ann["id"], original)
    restored = c.get_task(task["id"])
    rest_ann = next(a for a in restored["annotations"] if a["id"] == ann["id"])
    rest_corners = [r for r in rest_ann["result"] if r.get("from_name") == "corners"]
    print("\nRESTORED corner x:", [round(r["value"]["x"], 3) for r in rest_corners])
    print(f"  is_labeled={restored.get('is_labeled')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
