"""Private jobs only ever reach confidential miners; standard jobs may run anywhere."""

from __future__ import annotations

from kuno_protocol.schemas import JobStatus
from kuno_protocol.tiers import CONFIDENTIAL, OPEN, tier_for_tee, tier_serves


def test_tee_kinds_map_to_tiers_and_anything_unknown_is_open():
    assert tier_for_tee("tdx") == CONFIDENTIAL
    assert tier_for_tee("mock") == CONFIDENTIAL
    assert tier_for_tee("open") == OPEN
    assert tier_for_tee("sev-snp-someday") == OPEN


def test_private_jobs_need_a_confidential_miner_and_standard_jobs_take_any():
    assert tier_serves(CONFIDENTIAL, "private") and not tier_serves(OPEN, "private")
    assert tier_serves(CONFIDENTIAL, "standard") and tier_serves(OPEN, "standard")
    assert not tier_serves(CONFIDENTIAL, "public") and not tier_serves(OPEN, "")


def test_job_status_defaults_to_private_so_existing_gateways_and_clients_keep_parsing():
    status = {
        "job_id": "j", "status": "queued", "params": {
            "profile_id": "ltx-2.5-fast", "mode": "text_to_video", "duration_s": 2, "resolution": "720p", "aspect_ratio": "16:9", "fps": 24,
        },
        "enclave_id": "e", "price_usd": 0.1, "created_at": 0, "updated_at": 0,
    }
    assert JobStatus.model_validate(status).privacy == "private"
    assert JobStatus.model_validate({**status, "privacy": "standard"}).privacy == "standard"
