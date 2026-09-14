"""Capacity pay: how long each ready, attested confidential-tier GPU has been verified (VALIDATING.md, "Capacity pay").

Chutes (SN64) pays miners for the time their attested GPUs serve rather than per request, and an instance that lives
under an hour earns nothing there. This module keeps that uptime, from this validator's own challenges only; scoring.py
applies the gates and the switch's targets and blends it into the scores, and usd_pay.py prices it instead.

  check   a successful challenge verdict of this validator's own on the confidential tier, once for each GPU `ueid`
          identity it proved. GPUs counted without an identity, and open-tier GPUs (self-reported), are never checked.
  run     consecutive checks of one (hotkey, GPU) at most `max_gap_s` apart. A longer gap (a missed or failed
          challenge) ends the run, and the next check starts a new one.
  credit  a run counts once it spans the switch's `capacity_min_uptime_s`, and then all of it counts: the time
          between its checks that falls inside the scoring window, each stretch split equally over the families the
          enclave served at the later check.

`max_gap_s` is `KUNO_CAPACITY_MAX_GAP_S`, two round intervals by default: a round that runs long keeps the run, a round
in which the GPU wasn't verified breaks it. Runs are kept in the validator state file and pruned once they end before
the window.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

# `kuno-validator run --interval` default: one tempo.
DEFAULT_ROUND_INTERVAL_S = 4320.0
MAX_GAP_ROUNDS = 2.0


class CapacityTracker:
    """Verified runs per (hotkey, GPU identity token)."""

    def __init__(self, max_gap_s: float = MAX_GAP_ROUNDS * DEFAULT_ROUND_INTERVAL_S):
        if not max_gap_s > 0:
            raise ValueError("KUNO_CAPACITY_MAX_GAP_S must be positive")
        self.max_gap_s = float(max_gap_s)
        # hotkey -> GPU token -> runs, oldest first: [start, last, spans]. A span [from, to, profile ids] is the time
        # between two checks, with the profiles the enclave served at the later one.
        self.runs: dict[str, dict[str, list[list]]] = {}

    @classmethod
    def from_env(cls, env: Mapping[str, str], interval_s: float = DEFAULT_ROUND_INTERVAL_S) -> CapacityTracker:
        """KUNO_CAPACITY_MAX_GAP_S (seconds; default two round intervals)."""
        text = (env.get("KUNO_CAPACITY_MAX_GAP_S") or "").strip()
        return cls(float(text) if text else MAX_GAP_ROUNDS * interval_s)

    def record(self, checks: Iterable[tuple[str, str, Iterable[str]]], now: float, window_s: float) -> None:
        """One round's checks: (hotkey, GPU token, profile ids its enclave serves). A GPU that several of a hotkey's
        enclaves show is one check."""
        served: dict[tuple[str, str], set[str]] = {}
        for hotkey, token, profiles in checks:
            served.setdefault((hotkey, token), set()).update(profiles)
        for (hotkey, token), profiles in sorted(served.items()):
            runs = self.runs.setdefault(hotkey, {}).setdefault(token, [])
            run = runs[-1] if runs else None
            if run is not None and now <= run[1]:
                continue  # already checked at this time
            if run is None or now - run[1] > self.max_gap_s:
                runs.append([now, now, []])
                continue
            spans, ids = run[2], sorted(profiles)
            if spans and spans[-1][2] == ids:
                spans[-1][1] = now
            else:
                spans.append([run[1], now, ids])
            run[1] = now
        self.prune(now, window_s)

    def prune(self, now: float, window_s: float) -> None:
        """Drops runs that ended before the window and can't be extended any more, and stretches before the window.
        A run keeps its start, so the uptime rule still sees how long it has lasted."""
        start, horizon = now - window_s, now - max(window_s, self.max_gap_s)
        for hotkey in list(self.runs):
            gpus = self.runs[hotkey]
            for token in list(gpus):
                kept = [run for run in gpus[token] if run[1] >= horizon]
                for run in kept:
                    run[2] = [span for span in run[2] if span[1] > start]
                if kept:
                    gpus[token] = kept
                else:
                    del gpus[token]
            if not gpus:
                del self.runs[hotkey]

    def gpu_seconds(
        self, now: float, window_s: float, min_uptime_s: float, families: Callable[[list[str]], list[str]]
    ) -> dict[str, dict[str, float]]:
        """Verified GPU-seconds inside the window per hotkey and family, from runs that lasted at least `min_uptime_s`.
        `families` maps an enclave's profile ids to the families its time is split over."""
        start = now - window_s
        credit: dict[str, dict[str, float]] = {}
        for hotkey, gpus in sorted(self.runs.items()):
            for runs in gpus.values():
                for first, last, spans in runs:
                    if last - first < min_uptime_s:
                        continue
                    for begin, end, profiles in spans:
                        seconds, served = min(end, now) - max(begin, start), families(profiles)
                        if seconds <= 0 or not served:
                            continue
                        by_family = credit.setdefault(hotkey, {})
                        for family in served:
                            by_family[family] = by_family.get(family, 0.0) + seconds / len(served)
        return credit

    def dump(self) -> dict:
        return {"runs": self.runs}

    def load(self, data: Mapping | None) -> None:
        """Restores runs from the state file, skipping anything malformed."""
        if not isinstance(data, Mapping):
            return
        for hotkey, gpus in (data.get("runs") or {}).items():
            if not isinstance(gpus, Mapping):
                continue
            for token, runs in gpus.items():
                try:
                    parsed = [
                        [float(first), float(last), [[float(begin), float(end), [str(p) for p in ids]] for begin, end, ids in spans]]
                        for first, last, spans in runs
                    ]
                except (TypeError, ValueError):
                    continue
                if parsed:
                    self.runs.setdefault(hotkey, {})[token] = parsed
