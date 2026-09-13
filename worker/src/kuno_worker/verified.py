"""Verified mode inside the enclave: record every denoising step, keep what an audit opening
needs for the retention window, and forget it afterwards.

A backend in verified mode reports each latent state to a `StepRecorder`. The recorder
hashes it into a leaf and hands it to the `RetentionStore`, which keeps it encrypted with
a key that exists only in this process (it never leaves the confidential VM and dies with
it). `finish()` builds the Merkle commitment that goes into the receipt.

Retention strategy (see VERIFIED_MODE.md for the memory math): every latent is kept by
default (`checkpoint_every=1`). Profiles with large latents and many steps may keep every
k-th latent instead; an audit of a step between checkpoints then recomputes the missing
latents from the nearest checkpoint with the backend's registered replayer, and refuses to
serve anything whose recomputation does not reproduce the committed digest.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kuno_protocol.blobs import decrypt_blob, encrypt_blob
from kuno_protocol.canonical import canonical_json
from kuno_protocol.verified import (
    StepCommitment,
    StepLeaf,
    StepTranscript,
    Tensor,
    VerifiedModeError,
    build_commitment,
    expected_layout,
    f64_hex,
    latent_digest,
    new_salt,
    pack_tensors,
    tensor_from_array,
    unpack_tensors,
)

log = logging.getLogger("kuno.worker.verified")

DEFAULT_RETENTION_S = 3600.0
# A trajectory that never finished (crash, cancel) is dropped this long after it began.
ABANDONED_GRACE_S = 3600.0

# (transcript, backend context, target leaf index, latent state at index - 1) -> state at index
Replayer = Callable[[StepTranscript, bytes, int, list[Tensor]], list[Tensor]]


class RetentionError(RuntimeError):
    """An opening cannot be produced (expired, never retained, or recomputation diverged)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class RetainedJob:
    transcript: StepTranscript
    leaves: list[StepLeaf]
    salt: bytes
    commitment: StepCommitment
    audit_binding: str | None
    sealed_at: float


@dataclass
class _Entry:
    job_id: str
    begun_at: float
    checkpoint_every: int
    names: set[str] = field(default_factory=set)
    stored: set[int] = field(default_factory=set)
    record: RetainedJob | None = None


class RetentionStore:
    """Encrypted, expiring storage for per-step latents.

    `directory=None` keeps ciphertext in memory (TDX-encrypted RAM); a directory keeps it
    on the VM's disk, which the host can read, hence the encryption at rest either way.
    """

    def __init__(self, directory: Path | None = None, window_s: float = DEFAULT_RETENTION_S, clock: Callable[[], float] = time.time):
        self._key = os.urandom(32)  # enclave-held; never persisted, never logged
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            os.chmod(self.directory, 0o700)
        self.window_s = window_s
        self.clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._memory: dict[str, bytes] = {}
        self._replayers: dict[str, Replayer] = {}

    # ------------------------------------------------------------ storage

    def _path(self, name: str) -> Path:
        assert self.directory is not None
        return self.directory / hashlib.sha256(name.encode()).hexdigest()[:40]

    def _write(self, entry: _Entry, name: str, data: bytes) -> None:
        sealed = encrypt_blob(self._key, "kuno/v1/retain/" + name, data)
        if self.directory is None:
            self._memory[name] = sealed
        else:
            path = self._path(name)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(sealed)
            tmp.replace(path)
        entry.names.add(name)

    def _read(self, name: str) -> bytes:
        if self.directory is None:
            sealed = self._memory[name]
        else:
            sealed = self._path(name).read_bytes()
        return decrypt_blob(self._key, "kuno/v1/retain/" + name, sealed)

    def _delete(self, entry: _Entry) -> None:
        for name in entry.names:
            if self.directory is None:
                self._memory.pop(name, None)
            else:
                self._path(name).unlink(missing_ok=True)
        entry.names.clear()

    # ------------------------------------------------------------ writing

    def register_replayer(self, runtime: str, replayer: Replayer) -> None:
        self._replayers[runtime] = replayer

    def begin(self, job_id: str, checkpoint_every: int = 1, context: bytes = b"") -> None:
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        with self._lock:
            self.sweep()
            old = self._entries.pop(job_id, None)
            if old is not None:
                self._delete(old)
            entry = _Entry(job_id, self.clock(), checkpoint_every)
            self._entries[job_id] = entry
            self._write(entry, f"{job_id}/context", context)

    def put_latent(self, job_id: str, index: int, tensors: list[Tensor], force: bool = False) -> None:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                raise RetentionError("not_retained", "trajectory was not begun")
            if force or index % entry.checkpoint_every == 0:
                self._write(entry, f"{job_id}/latent/{index}", pack_tensors(tensors))
                entry.stored.add(index)

    def seal(self, job_id: str, record: RetainedJob) -> float:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                raise RetentionError("not_retained", "trajectory was not begun")
            entry.record = record
            return record.sealed_at + self.window_s

    def discard(self, job_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(job_id, None)
            if entry is not None:
                self._delete(entry)

    def sweep(self, now: float | None = None) -> int:
        """Deletes every trajectory past its window. Returns how many were removed."""
        now = self.clock() if now is None else now
        removed = 0
        with self._lock:
            for job_id, entry in list(self._entries.items()):
                if entry.record is not None:
                    expired = now >= entry.record.sealed_at + self.window_s
                else:
                    expired = now >= entry.begun_at + self.window_s + ABANDONED_GRACE_S
                if expired:
                    self._delete(entry)
                    del self._entries[job_id]
                    removed += 1
        return removed

    # ------------------------------------------------------------ reading

    def record(self, job_id: str) -> RetainedJob | None:
        with self._lock:
            self.sweep()
            entry = self._entries.get(job_id)
            return entry.record if entry is not None else None

    def expires_at(self, job_id: str) -> float | None:
        record = self.record(job_id)
        return record.sealed_at + self.window_s if record is not None else None

    def __contains__(self, job_id: str) -> bool:
        return self.record(job_id) is not None

    def latents(self, job_id: str, index: int) -> list[Tensor]:
        """The committed latent state at leaf `index`, recomputed from a checkpoint if needed."""
        with self._lock:
            self.sweep()
            entry = self._entries.get(job_id)
            if entry is None or entry.record is None:
                raise RetentionError("not_retained", "no retained trajectory for this job")
            record = entry.record
            if not 0 <= index < len(record.leaves):
                raise RetentionError("bad_step", "leaf index out of range")
            if index in entry.stored:
                return unpack_tensors(self._read(f"{job_id}/latent/{index}"))
            layout = expected_layout(record.transcript)
            stage = layout[index].stage
            start = max((i for i in entry.stored if i < index and layout[i].stage == stage), default=None)
            replayer = self._replayers.get(record.transcript.runtime)
            if start is None or replayer is None:
                raise RetentionError("not_retained", "latent was not retained and cannot be recomputed")
            context = self._read(f"{job_id}/context")
            state = unpack_tensors(self._read(f"{job_id}/latent/{start}"))
        for target in range(start + 1, index + 1):
            state = replayer(record.transcript, context, target, state)
            if latent_digest(state) != record.leaves[target].latent:
                log.error("recomputing leaf %d of job %s diverged from its commitment", target, job_id)
                raise RetentionError("nondeterministic", "recomputation did not reproduce the committed latent")
        return state


@dataclass
class OpeningsHandle:
    """What a VideoResult carries: where this job's openings can be produced from, and until when."""

    store: RetentionStore
    job_id: str

    @property
    def expires_at(self) -> float | None:
        return self.store.expires_at(self.job_id)

    def discard(self) -> None:
        self.store.discard(self.job_id)


class StepRecorder:
    """Receives each latent state from a backend's step hook, in order, with no gaps."""

    def __init__(
        self,
        store: RetentionStore,
        job_id: str,
        *,
        checkpoint_every: int = 1,
        context: bytes = b"",
        audit_binding: str | None = None,
    ):
        self.store = store
        self.job_id = job_id
        self.audit_binding = audit_binding
        self.leaves: list[StepLeaf] = []
        self._salt = new_salt()
        self._stage = -1
        store.begin(job_id, checkpoint_every, context)

    def report(self, index: int, stage: int, kind: str, sigma: float, tensors: list[Tensor]) -> None:
        if index != len(self.leaves):
            raise VerifiedModeError(f"step {index} reported after {len(self.leaves)} leaves")
        leaf = StepLeaf(index=index, stage=stage, kind=kind, sigma=f64_hex(sigma), latent=latent_digest(tensors))
        self.leaves.append(leaf)
        # A stage's first latent is always kept: recomputation never crosses a stage boundary.
        self.store.put_latent(self.job_id, index, tensors, force=kind == "init" or stage != self._stage)
        self._stage = stage

    def report_arrays(self, index: int, stage: int, kind: str, sigma: float, **arrays) -> None:
        self.report(index, stage, kind, sigma, [tensor_from_array(name, array) for name, array in arrays.items()])

    def finish(self, transcript: StepTranscript) -> tuple[StepCommitment, OpeningsHandle]:
        """Raises VerifiedModeError when the reported steps don't match the transcript's schedule."""
        if transcript.job_id != self.job_id:
            raise VerifiedModeError("transcript is for a different job")
        commitment, _ = build_commitment(transcript, self.leaves, self._salt)
        record = RetainedJob(transcript, list(self.leaves), self._salt, commitment, self.audit_binding, self.store.clock())
        self.store.seal(self.job_id, record)
        return commitment, OpeningsHandle(self.store, self.job_id)

    def abort(self) -> None:
        self.store.discard(self.job_id)


def context_bytes(**values) -> bytes:
    """Canonical encoding of what a replayer needs besides latents (e.g. the prompt)."""
    return canonical_json(values)


_shared: RetentionStore | None = None
_shared_lock = threading.Lock()


def shared_retention() -> RetentionStore:
    """The process-wide store that backends write to and the audit responder reads from.

    KUNO_VERIFIED_RETENTION_S sets the window (default 3600); KUNO_VERIFIED_RETENTION_DIR puts
    ciphertext on disk instead of in (encrypted) VM memory.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            directory = os.environ.get("KUNO_VERIFIED_RETENTION_DIR")
            _shared = RetentionStore(
                Path(directory) if directory else None, float(os.environ.get("KUNO_VERIFIED_RETENTION_S", DEFAULT_RETENTION_S))
            )
        return _shared
