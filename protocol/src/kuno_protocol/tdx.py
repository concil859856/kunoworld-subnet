"""Intel TDX quote verification with Intel DCAP, via Phala's dcap-qvl.

dcap-qvl (MIT; Rust with Python bindings, `kuno-protocol[tdx]`) checks the PCK certificate
chain up to Intel's SGX Root CA, the CRLs, the quoting enclave's report and identity, the
quote signature, and the platform's TCB level against Intel-signed TCB info. We then apply
our own TCB-status and advisory policy to its result.

Collateral comes from Intel PCS or any PCCS that speaks the v4 API. It changes at most
every few weeks per platform family, so it is cached per (PCCS, FMSPC, CA) until the
earliest `nextUpdate` it carries or a configured TTL, whichever comes first; a
verification that fails because cached collateral expired is retried once with a fresh
copy.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

INTEL_PCS_URL = "https://api.trustedservices.intel.com"
DEFAULT_ALLOWED_TCB_STATUSES = ("UpToDate",)
TCB_STATUSES = (
    "UpToDate",
    "SWHardeningNeeded",
    "ConfigurationNeeded",
    "ConfigurationAndSWHardeningNeeded",
    "OutOfDate",
    "OutOfDateConfigurationNeeded",
    "Revoked",
)


class TdxVerifierUnavailable(RuntimeError):
    """The DCAP verification library is not installed."""


def dcap_module():
    try:
        import dcap_qvl
    except ImportError as exc:
        raise TdxVerifierUnavailable("TDX quote verification needs dcap-qvl: install kuno-protocol[tdx]") from exc
    return dcap_qvl


@dataclass
class TdxQuoteResult:
    ok: bool
    detail: str
    status: str | None = None
    advisory_ids: list[str] = field(default_factory=list)
    # The platform's PPID from the PCK certificate dcap-qvl verified to Intel's root (dcap-qvl >= 0.6).
    ppid: bytes | None = None


def _timestamp(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def collateral_expiry(collateral: Any) -> float | None:
    """Earliest `nextUpdate` of the signed TCB info and QE identity, as a Unix time."""
    dates = []
    for document in (collateral.tcb_info, collateral.qe_identity):
        try:
            dates.append(_timestamp(json.loads(document)["nextUpdate"]))
        except (ValueError, KeyError, TypeError):
            continue
    return min(dates) if dates else None


def fetch_collateral(pccs_url: str, quote: bytes, timeout_s: float = 30.0) -> Any:
    """dcap-qvl's fetch is async; run it on its own loop so callers inside an event loop can use it too."""
    dcap = dcap_module()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(lambda: asyncio.run(dcap.get_collateral(pccs_url, quote))).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


class CollateralCache:
    def __init__(
        self,
        fetch: Callable[[str, bytes], Any] = fetch_collateral,
        ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ):
        self._fetch, self._ttl, self._clock = fetch, ttl_s, clock
        self._entries: dict[tuple[str, str, str], tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, pccs_url: str, quote: bytes, refresh: bool = False) -> Any:
        parsed = dcap_module().Quote.parse(quote)
        key = (pccs_url, parsed.fmspc(), parsed.ca())
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
        if entry is not None and not refresh and entry[1] > now:
            return entry[0]
        collateral = self._fetch(pccs_url, quote)
        expiry = now + self._ttl
        next_update = collateral_expiry(collateral)
        if next_update is not None:
            expiry = min(expiry, next_update)
        with self._lock:
            self._entries[key] = (collateral, expiry)
        return collateral


def verify_tdx_quote(
    quote: bytes,
    *,
    collateral: Any = None,
    cache: CollateralCache | None = None,
    pccs_url: str = INTEL_PCS_URL,
    now: float | None = None,
    allowed_statuses: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_TCB_STATUSES,
    rejected_advisories: tuple[str, ...] | list[str] = (),
    root_ca_der: bytes | None = None,
) -> TdxQuoteResult:
    """Full DCAP verification of a TD quote: Intel root, CRLs, QE identity, signature, TCB status.

    Pass `collateral` to verify offline; otherwise it is fetched from `pccs_url` through `cache`.
    Measurements and REPORTDATA are not judged here — `verify_evidence` does that.
    """
    dcap = dcap_module()
    now_s = int(time.time() if now is None else now)
    try:
        parsed = dcap.Quote.parse(quote)
    except ValueError as exc:
        return TdxQuoteResult(False, f"unparseable quote: {exc}")
    if not parsed.is_tdx():
        return TdxQuoteResult(False, "not a TDX quote")

    supplied = collateral is not None
    cache = cache or CollateralCache()
    report, error = None, None
    for refresh in (False,) if supplied else (False, True):
        if not supplied:
            try:
                collateral = cache.get(pccs_url, quote, refresh=refresh)
            except Exception as exc:  # network, TLS, PCCS errors all mean "cannot verify now"
                return TdxQuoteResult(False, f"could not fetch collateral from {pccs_url}: {type(exc).__name__}: {exc}")
        try:
            if root_ca_der is None:
                report = dcap.verify(quote, collateral, now_s)
            else:
                report = dcap.verify_with_root_ca(quote, collateral, root_ca_der, now_s)
            break
        except ValueError as exc:
            error = exc
            if "expired" not in str(exc).lower():
                break
    if report is None:
        return TdxQuoteResult(False, f"DCAP verification failed: {error}")

    status, advisories = report.status, list(report.advisory_ids)
    if status not in allowed_statuses:
        return TdxQuoteResult(
            False, f"TCB status {status} is not accepted (allowed: {', '.join(allowed_statuses)})", status, advisories
        )
    blocked = sorted(set(advisories) & set(rejected_advisories))
    if blocked:
        return TdxQuoteResult(False, f"platform is affected by rejected advisories {', '.join(blocked)}", status, advisories)
    detail = f"TCB status {status}" + (f", advisories {', '.join(advisories)}" if advisories else "")
    ppid = getattr(report, "ppid", None)
    return TdxQuoteResult(True, detail, status, advisories, bytes(ppid) if ppid else None)


class DcapQuoteVerifier:
    """The `QuoteVerifier` that `verify_evidence` expects, backed by dcap-qvl and a collateral cache."""

    def __init__(
        self,
        pccs_url: str = INTEL_PCS_URL,
        allowed_statuses: tuple[str, ...] | list[str] = DEFAULT_ALLOWED_TCB_STATUSES,
        rejected_advisories: tuple[str, ...] | list[str] = (),
        collateral_ttl_s: float = 3600.0,
        root_ca_der: bytes | None = None,
        fetch: Callable[[str, bytes], Any] | None = None,
        clock: Callable[[], float] = time.time,
        timeout_s: float = 30.0,
    ):
        dcap_module()  # a production verifier must fail at startup, not at the first registration
        unknown = [s for s in allowed_statuses if s not in TCB_STATUSES]
        if unknown:
            raise ValueError(f"unknown TCB statuses: {', '.join(unknown)}")
        self.pccs_url = pccs_url.rstrip("/")
        self.allowed_statuses = tuple(allowed_statuses)
        self.rejected_advisories = tuple(rejected_advisories)
        self.root_ca_der = root_ca_der
        self._clock = clock
        fetcher = fetch or (lambda url, quote: fetch_collateral(url, quote, timeout_s))
        self.cache = CollateralCache(fetcher, collateral_ttl_s, clock)

    def verify(self, quote: bytes) -> tuple[bool, str]:
        result = self.verify_quote(quote)
        return result.ok, result.detail

    def verify_quote(self, quote: bytes) -> TdxQuoteResult:
        """The full result, including the verified PPID `verify_evidence` turns into a hardware identity."""
        return verify_tdx_quote(
            quote,
            cache=self.cache,
            pccs_url=self.pccs_url,
            now=self._clock(),
            allowed_statuses=self.allowed_statuses,
            rejected_advisories=self.rejected_advisories,
            root_ca_der=self.root_ca_der,
        )
