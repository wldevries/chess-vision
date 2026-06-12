"""Azure Blob backend for the bulk dataset sync.

The canonical off-machine home for the captured dataset is a single blob container
(default ``chess``). Unlike hive-vision's read-through ``LabelStore`` (which reads the
blob on every save), this module is a *bulk directory sync*: mirror a local tree up to
the container or back down, skipping objects whose size already matches — the same
size-based, resumable contract as the MinIO path in ``storage.py``.

Auth is the account connection string in ``.env`` (``STORAGE_CONNECTION_STRING``, shared
with the hive-vision project); callers ``load_dotenv()`` before :meth:`BlobConfig.from_env`.
The ``azure.storage.blob`` import is **lazy** (inside the methods) so geometry/test paths
never need the SDK or a network round-trip.

Keys mirror the local layout as flat POSIX paths (no container-side folders needed)::

    source/<relpath>      raw originals (incl. source/inbox/<photo>)
    store/<relpath>       EXIF-normalized JPEGs
    labels.jsonl          the label index (single blob, last write wins)
    models/<name>         model weights (future; same container, models/ prefix)
"""

from __future__ import annotations

import mimetypes
import os
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from azure.storage.blob import ContainerClient

DEFAULT_CONTAINER = "chess"
CONNECTION_STRING_ENV = "STORAGE_CONNECTION_STRING"
CONTAINER_ENV = "AZURE_BLOB_CONTAINER"


@dataclass(frozen=True)
class BlobConfig:
    """Azure Blob connection info: an account connection string plus a container."""

    connection_string: str
    container: str = DEFAULT_CONTAINER

    @classmethod
    def from_env(cls, *, load: bool = True) -> BlobConfig:
        """Build from ``STORAGE_CONNECTION_STRING`` and ``AZURE_BLOB_CONTAINER``
        (the latter defaults to ``chess``). Raises if the connection string is missing."""
        if load:
            from dotenv import load_dotenv

            load_dotenv()
        conn = os.environ.get(CONNECTION_STRING_ENV)
        if not conn:
            raise RuntimeError(
                f"{CONNECTION_STRING_ENV} is not set — put the storage account connection "
                "string in .env (see .env.example)."
            )
        return cls(
            connection_string=conn,
            container=os.environ.get(CONTAINER_ENV) or DEFAULT_CONTAINER,
        )

    def __repr__(self) -> str:  # never echo the connection string (it holds the key)
        return f"BlobConfig(connection_string=***, container={self.container!r})"


def get_container_client(config: BlobConfig) -> ContainerClient:
    """A ``ContainerClient`` bound to the configured account + container. Lazy SDK import."""
    from azure.storage.blob import BlobServiceClient

    svc = BlobServiceClient.from_connection_string(config.connection_string)
    return svc.get_container_client(config.container)


def _content_type(path: Path) -> str:
    """MIME type for a blob. mimetypes covers .jpg/.json/.png; .jsonl is unregistered,
    so map our newline-delimited label files explicitly."""
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _remote_sizes(container: ContainerClient, prefix: str) -> dict[str, int]:
    """Map of blob name -> size for every blob under ``prefix``."""
    return {b.name: b.size for b in container.list_blobs(name_starts_with=prefix)}


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
    config: BlobConfig | None = None,
    container: ContainerClient | None = None,
    dry_run: bool = False,
    force: bool = False,
    ignore: Collection[str] = (),
) -> SyncResult:
    """Upload ``local_dir`` to ``<container>/<prefix>/...``, skipping same-size blobs.

    Each blob is written with an explicit ``Content-Type`` so browser preview / direct
    serving works. ``force=True`` re-uploads even same-size blobs (e.g. to repair a
    content type). ``ignore`` is a set of basenames to never transfer.

    Relative paths under ``local_dir`` become keys under ``prefix`` (forward slashes).
    """
    from azure.storage.blob import ContentSettings

    config = config or BlobConfig.from_env()
    container = container or get_container_client(config)
    local_dir = Path(local_dir)
    prefix = prefix.strip("/")
    ignore = set(ignore)

    remote = _remote_sizes(container, f"{prefix}/")
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
            with path.open("rb") as fh:
                container.upload_blob(
                    key,
                    fh,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=_content_type(path)),
                )
        transferred.append(key)
    return SyncResult(transferred=transferred, skipped=skipped)


def upload_file(
    local_path: str | Path,
    key: str,
    *,
    config: BlobConfig | None = None,
    container: ContainerClient | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> SyncResult:
    """Upload a single file to ``<container>/<key>`` (used for the root ``labels.jsonl``)."""
    from azure.storage.blob import ContentSettings

    config = config or BlobConfig.from_env()
    container = container or get_container_client(config)
    local_path = Path(local_path)
    key = key.strip("/")

    remote = _remote_sizes(container, key)
    if not force and remote.get(key) == local_path.stat().st_size:
        return SyncResult(transferred=[], skipped=1)
    if not dry_run:
        with local_path.open("rb") as fh:
            container.upload_blob(
                key,
                fh,
                overwrite=True,
                content_settings=ContentSettings(content_type=_content_type(local_path)),
            )
    return SyncResult(transferred=[key], skipped=0)


def download_prefix(
    prefix: str,
    local_dir: str | Path,
    *,
    config: BlobConfig | None = None,
    container: ContainerClient | None = None,
    dry_run: bool = False,
    ignore: Collection[str] = (),
) -> SyncResult:
    """Download everything under ``<container>/<prefix>/`` into ``local_dir``, mirroring
    the key layout. Skips files already present at the same size. A bare-file ``prefix``
    (e.g. ``labels.jsonl``) downloads that single blob to ``local_dir/<prefix>``."""
    config = config or BlobConfig.from_env()
    container = container or get_container_client(config)
    local_dir = Path(local_dir)
    prefix = prefix.strip("/")
    ignore = set(ignore)

    remote = _remote_sizes(container, prefix)
    transferred: list[str] = []
    skipped = 0
    for key, size in sorted(remote.items()):
        # Strip the "prefix/" dir component; a single-file prefix keeps its basename.
        rel = key[len(prefix) + 1 :] if key.startswith(prefix + "/") else Path(key).name
        if Path(rel).name in ignore:
            continue
        dest = local_dir / rel
        if dest.exists() and dest.stat().st_size == size:
            skipped += 1
            continue
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                container.download_blob(key).readinto(fh)
        transferred.append(key)
    return SyncResult(transferred=transferred, skipped=skipped)
