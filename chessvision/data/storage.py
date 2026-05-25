"""S3-compatible object storage (MinIO) for the captured dataset.

Syncs the local `data/captures/` tree (per-session `captures.jsonl` label files
plus their `*.jpg` frames) to/from a bucket on the local-network MinIO server, so
the dataset lives off any single machine without bloating git.

Configuration comes from the environment (loaded from `.env` via python-dotenv):
    MINIO_ENDPOINT_URL   e.g. http://workstation.lan:9000  (the S3 API port, not 9001)
    MINIO_ACCESS_KEY     access key (created with `mc admin accesskey create`)
    MINIO_SECRET_KEY     secret key
    MINIO_BUCKET         bucket name, e.g. "chess"

Sync is one-directional per call and size-based: a file is (re)transferred only
when it is missing on the destination or its size differs. That is deliberately
simple — capture frames are write-once, so content never changes under a name.
"""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

# MinIO speaks path-style S3 (bucket in the path, not the host), so virtual-host
# addressing must be disabled.
_BOTO_CONFIG = Config(signature_version="s3v4", s3={"addressing_style": "path"})


@dataclass(frozen=True)
class StorageConfig:
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str

    @classmethod
    def from_env(cls, *, load: bool = True) -> StorageConfig:
        if load:
            load_dotenv()
        missing = [
            name
            for name in (
                "MINIO_ENDPOINT_URL",
                "MINIO_ACCESS_KEY",
                "MINIO_SECRET_KEY",
                "MINIO_BUCKET",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                f"missing storage config: {', '.join(missing)} "
                "(set them in .env — see .env.example)"
            )
        return cls(
            endpoint_url=os.environ["MINIO_ENDPOINT_URL"],
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            bucket=os.environ["MINIO_BUCKET"],
        )


def get_client(config: StorageConfig):
    """A boto3 S3 client bound to the MinIO endpoint."""
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        config=_BOTO_CONFIG,
    )


def put_bytes(client, bucket: str, key: str, data: bytes, content_type: str) -> None:
    """PUT an in-memory object (no local file) with an explicit Content-Type."""
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def get_bytes(client, bucket: str, key: str) -> bytes:
    """Read an object's full body into memory."""
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _remote_sizes(client, bucket: str, prefix: str) -> dict[str, int]:
    """Map of key -> size for every object under `prefix`."""
    sizes: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _content_type(path: Path) -> str:
    """MIME type for an object key. mimetypes covers .jpg/.json/.png; .jsonl is
    not registered, so map our newline-delimited label files explicitly."""
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


@dataclass
class SyncResult:
    transferred: list[str]
    skipped: int

    def summary(self) -> str:
        return f"{len(self.transferred)} transferred, {self.skipped} up-to-date"


def upload_dir(
    local_dir: str | Path,
    prefix: str,
    *,
    config: StorageConfig | None = None,
    client=None,
    dry_run: bool = False,
    force: bool = False,
    ignore: Collection[str] = (),
) -> SyncResult:
    """Upload `local_dir` to `<bucket>/<prefix>/...`, skipping same-size objects.

    Each object is PUT with an explicit `Content-Type` (boto3 otherwise defaults
    to application/octet-stream, which breaks browser preview / direct serving).
    Pass `force=True` to re-upload even same-size objects — needed to repair the
    content type of objects already in the bucket.

    `ignore` is a set of basenames to never transfer (e.g. derived top-level files
    that are regenerated locally and shouldn't live in the bucket).

    Relative paths under `local_dir` become keys under `prefix` (with forward
    slashes, as S3 requires regardless of host OS).
    """
    config = config or StorageConfig.from_env()
    client = client or get_client(config)
    local_dir = Path(local_dir)
    prefix = prefix.strip("/")
    ignore = set(ignore)

    remote = _remote_sizes(client, config.bucket, f"{prefix}/")
    transferred: list[str] = []
    skipped = 0
    for path in _iter_files(local_dir):
        if path.name in ignore:
            continue
        rel = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{rel}"
        if not force and remote.get(key) == path.stat().st_size:
            skipped += 1
            continue
        if not dry_run:
            client.upload_file(
                str(path),
                config.bucket,
                key,
                ExtraArgs={"ContentType": _content_type(path)},
            )
        transferred.append(key)
    return SyncResult(transferred=transferred, skipped=skipped)


def download_prefix(
    prefix: str,
    local_dir: str | Path,
    *,
    config: StorageConfig | None = None,
    client=None,
    dry_run: bool = False,
    ignore: Collection[str] = (),
) -> SyncResult:
    """Download everything under `<bucket>/<prefix>/` into `local_dir`, mirroring
    the key layout. Skips files already present at the same size.

    `ignore` is a set of basenames to never download (e.g. derived top-level files
    regenerated locally — pulling them would clobber the fresh local copy)."""
    config = config or StorageConfig.from_env()
    client = client or get_client(config)
    local_dir = Path(local_dir)
    prefix = prefix.strip("/")
    ignore = set(ignore)

    remote = _remote_sizes(client, config.bucket, f"{prefix}/")
    transferred: list[str] = []
    skipped = 0
    for key, size in sorted(remote.items()):
        rel = key[len(prefix) + 1 :]  # strip "prefix/"
        if Path(rel).name in ignore:
            continue
        dest = local_dir / rel
        if dest.exists() and dest.stat().st_size == size:
            skipped += 1
            continue
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(config.bucket, key, str(dest))
        transferred.append(key)
    return SyncResult(transferred=transferred, skipped=skipped)
