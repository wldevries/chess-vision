"""Tests for publishing a finished session: photos+metadata upload and task
generation, exercised against an in-memory fake S3 client."""

from __future__ import annotations

import json
from pathlib import Path

from chessvision.data.publish import publish_session
from chessvision.data.storage import StorageConfig


class FakeS3:
    """Minimal stand-in for the boto3 S3 client used by storage/publish."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get_paginator(self, _name: str):
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket, Prefix):  # noqa: N803 (boto3 kwarg names)
                contents = [
                    {"Key": k, "Size": len(v)} for k, v in objects.items() if k.startswith(Prefix)
                ]
                yield {"Contents": contents}

        return _Paginator()

    def upload_file(self, filename, bucket, key, ExtraArgs=None):  # noqa: N803
        self.objects[key] = Path(filename).read_bytes()

    def put_object(self, *, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.objects[Key] = Body


CONFIG = StorageConfig(endpoint_url="http://fake", access_key="a", secret_key="b", bucket="chess")


def _session_dir(tmp_path: Path) -> tuple[Path, list[dict]]:
    out = tmp_path / "20260524-101010"
    out.mkdir()
    (out / "a.jpg").write_bytes(b"\xff\xd8jpegbytes")  # content doesn't need to decode for upload
    (out / "captures.jsonl").write_text("{}\n", encoding="utf-8")
    records = [
        {
            "filename": "a.jpg",
            "corners": {
                "top_left": [10, 10],
                "top_right": [90, 12],
                "bottom_right": [88, 88],
                "bottom_left": [12, 86],
            },
            "pieces": [
                {
                    "piece": "K",
                    "color": "w",
                    "base": [50, 50],
                    "quad": [[40, 40], [60, 40], [60, 60], [40, 60]],
                }
            ],
        }
    ]
    return out, records


def test_publish_uploads_photos_metadata_and_writes_tasks(tmp_path: Path, monkeypatch) -> None:
    out, records = _session_dir(tmp_path)
    # build_task reads image dimensions via Pillow; stub it so the fake jpeg works.
    monkeypatch.setattr("chessvision.data.publish.ls.image_size_from_path", lambda _p: (100, 100))
    client = FakeS3()

    result = publish_session(out, "20260524-101010", records, config=CONFIG, client=client)

    # Photos + metadata landed under captures/<session>/...
    assert "captures/20260524-101010/a.jpg" in client.objects
    assert "captures/20260524-101010/captures.jsonl" in client.objects
    # A task JSON landed under tasks/<session>/... and references the s3 image.
    task_key = "tasks/20260524-101010/a.json"
    assert task_key in client.objects
    task = json.loads(client.objects[task_key])
    assert task["data"]["image"] == "s3://chess/captures/20260524-101010/a.jpg"
    # 4 corner keypoints + 1 piece keypoint (points-only by default).
    assert len(task["predictions"][0]["result"]) == 5
    assert result["tasks"] == 1
    assert result["uploaded"] == 2  # a.jpg + captures.jsonl


def test_publish_skips_task_when_photo_missing(tmp_path: Path, monkeypatch) -> None:
    out, records = _session_dir(tmp_path)
    monkeypatch.setattr("chessvision.data.publish.ls.image_size_from_path", lambda _p: (100, 100))
    records.append({"filename": "gone.jpg", "corners": None, "pieces": None})  # never written
    client = FakeS3()

    result = publish_session(out, "20260524-101010", records, config=CONFIG, client=client)

    assert "tasks/20260524-101010/gone.json" not in client.objects
    assert result["tasks"] == 1
