"""The enclave seals every output video in the padded blob format (version 2). The receipt's `content_digest` still
covers the plaintext file, and `output_digest`/`output_bytes` describe the sealed, padded blob that was uploaded."""

from __future__ import annotations

import uuid

from kuno_protocol.blobs import HEADER_LEN, LENGTH_LEN, TAG_LEN, V2, blob_version, decrypt_blob, sealed_size
from kuno_protocol.canonical import b64e, sha256_hex
from kuno_protocol.crypto import SenderSession
from kuno_protocol.profiles import Mode
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad, output_label
from kuno_worker.backends.mock import MockBackend

from test_worker_provenance import make_worker


def test_the_worker_uploads_a_padded_output_and_the_receipt_describes_both_files(tmp_path):
    worker = make_worker(tmp_path, MockBackend())
    params = GenerationParams(profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    ciphertext = session.seal(
        SealedPayload(prompt="a lighthouse at dusk").model_dump_json().encode(), job_aad(job_id, worker.identity.enclave_id, params, [])
    )
    receipt = worker.process(MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[]))

    [sealed] = worker.client.uploads
    assert blob_version(sealed) == V2
    video = decrypt_blob(session.output_key, output_label(job_id), sealed)
    assert video[4:8] == b"ftyp"
    # Padded to the PADMÉ bucket of the file plus its authenticated length, never just the plaintext plus overhead.
    assert len(sealed) == sealed_size(len(video)) >= HEADER_LEN + LENGTH_LEN + len(video) + TAG_LEN
    body = receipt.body
    assert body.content_digest == sha256_hex(video)
    assert (body.output_digest, body.output_bytes) == (sha256_hex(sealed), len(sealed))
