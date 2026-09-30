"""Untrusted source artifacts from a checkout of an external corpus, bound to an exact commit's tree listing.

The system never shells out (see ``tests/test_claude_contract_security.py``), so git is run by the *operator*, outside
the system, to produce the tree listing of the commit being imported::

    git -C <checkout> checkout <commit>
    git -C <checkout> ls-tree -r -z <commit> > <manifest>

Every file is read from the checkout and its git blob id is re-derived from the bytes; it must equal the blob id the
manifest lists for that path at that commit, or the file is refused (``ExternalSourceError``). Edited, re-encoded
(e.g. line-ending conversion) or substituted files therefore cannot enter under the commit's name. The blob id is the
object id GitHub serves for ``<repository>/blob/<commit>/<path>``, so anyone can re-verify a recorded identity
independently. Limitation: the commit id itself is operator-declared — the system checks bytes against the listing,
not the listing against the commit object.

Identity is content: SHA-256 of the exact bytes plus the git blob id. A filename, strategy name or URL alone is never
an identity. Files are read as bytes and handed to text parsers; they are never executed, imported or evaluated.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ati.core.canonical import sha256_hex
from ati.core.errors import ExternalSourceError

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_BLOB = re.compile(r"^[0-9a-f]{40}$")
_UNSAFE_PATH = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class SourceArtifact:
    repository: str          # canonical "owner/name"
    commit: str              # full 40-hex commit id the bytes belong to
    path: str                # path inside the repository at that commit
    url: str                 # human link to the exact file at the exact commit
    git_blob: str            # git object id of the file content (re-derived from the bytes)
    source_hash: str         # SHA-256 of the exact bytes
    size: int
    raw: bytes

    @property
    def external_strategy_id(self) -> str:
        """Immutable identity: repository + commit + path + content hash. The same file re-imported at the same
        commit is the same identity; a changed file (or a later commit) is a new identity; the old one stays."""
        return "ext_" + sha256_hex({"repository": self.repository, "commit": self.commit, "path": self.path,
                                    "source_hash": self.source_hash})[:24]

    def identity(self) -> dict:
        return {"external_strategy_id": self.external_strategy_id, "source_repository": self.repository,
                "source_commit": self.commit, "source_path": self.path, "source_url": self.url,
                "source_git_blob": self.git_blob, "source_hash": self.source_hash, "source_size": self.size}


def git_blob_id(raw: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()  # noqa: S324 (git object id, not security)


def _check_path(path: str) -> None:
    if not path or _UNSAFE_PATH.search(path) or path.startswith("/") or ".." in path.split("/"):
        raise ExternalSourceError(f"unsafe source path {path!r}")


def parse_manifest(data: bytes) -> dict[str, str]:
    """``git ls-tree -r -z`` output → {path: blob id}. Only regular files (100644/100755) are importable."""
    out: dict[str, str] = {}
    for rec in data.split(b"\0"):
        if not rec:
            continue
        try:
            meta, path_b = rec.split(b"\t", 1)
            mode, kind, oid = meta.decode("ascii").split(" ")
            path = path_b.decode("utf-8")
        except ValueError as exc:
            raise ExternalSourceError("manifest is not `git ls-tree -r -z` output") from exc
        if kind == "blob" and mode in ("100644", "100755") and _BLOB.match(oid):
            out[path] = oid
    if not out:
        raise ExternalSourceError("manifest lists no files")
    return out


class GitCorpus:
    """Read-only view of one external repository at one commit: a checkout plus that commit's tree listing."""

    def __init__(self, checkout: Path | str, repository: str, commit: str, manifest: Path | str,
                 url_base: str | None = None):
        if not _COMMIT.match(commit):
            raise ExternalSourceError(f"source commit must be a full 40-hex commit id, got {commit!r}")
        self.checkout = Path(checkout).resolve()
        self.repository = repository
        self.commit = commit
        try:
            self.blobs = parse_manifest(Path(manifest).read_bytes())
        except OSError as exc:
            raise ExternalSourceError(f"manifest unreadable: {exc}") from exc
        self.url_base = url_base or f"https://github.com/{repository}"

    def list(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self.blobs if p.startswith(prefix))

    def read(self, path: str) -> SourceArtifact:
        _check_path(path)
        expected = self.blobs.get(path)
        if expected is None:
            raise ExternalSourceError(f"{path} is not a file of {self.repository}@{self.commit[:12]}")
        target = self.checkout / path
        if target.is_symlink() or self.checkout not in target.resolve().parents:
            raise ExternalSourceError(f"{path} escapes the checkout")
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise ExternalSourceError(f"{path} unreadable: {exc}") from exc
        if git_blob_id(raw) != expected:
            raise ExternalSourceError(f"{path}: bytes are not git object {expected[:12]} of commit {self.commit[:12]} "
                                      "(checkout modified, re-encoded, or at another commit)")
        return SourceArtifact(self.repository, self.commit, path, f"{self.url_base}/blob/{self.commit}/{path}",
                              expected, hashlib.sha256(raw).hexdigest(), len(raw), raw)

    def read_many(self, paths: list[str]) -> list[SourceArtifact]:
        return [self.read(p) for p in paths]
