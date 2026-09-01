"""Verify the config-signature lineage manifest.

For every run listed in ``data/provenance/config_signature_lineage.json`` this
script checks that

1. the recorded ``reconstructed_legacy_payload`` reproduces the
   ``stored_signature`` (SHA-256 16-hex prefix over the canonical JSON
   encoding),
2. the recorded ``changed_fields`` exactly describe the difference between the
   legacy payload and the bundled-runner payload, and
3. the actual bundled ``status.json`` at ``status_path`` exists and carries the
   same ``config_signature`` and ``case_id`` as the manifest entry — so a
   manifest edited independently of the data cannot pass.

With ``--with-runners`` it additionally rebuilds each case configuration with
the bundled experiment runners and checks that

4. the resulting payload matches ``bundled_runner_payload`` for every manifest
   entry, and
5. the coverage accounting holds: the set of runner-covered runs whose stored
   signature does not recompute equals exactly the manifest's run set.

Usage (from the repository root):
    python3 scripts/analysis/verify_config_signatures.py
    python3 scripts/analysis/verify_config_signatures.py --with-runners
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data" / "provenance" / "config_signature_lineage.json"
ABSENT = "<absent>"


def canonical_signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def check_manifest(manifest: dict) -> int:
    failures = 0
    for case_id, entry in manifest["runs"].items():
        legacy = entry["reconstructed_legacy_payload"]
        current = entry["bundled_runner_payload"]

        sig = canonical_signature(legacy)
        if sig != entry["stored_signature"]:
            print(f"[FAIL] {case_id}: legacy payload -> {sig}, "
                  f"manifest stored_signature {entry['stored_signature']}")
            failures += 1

        recorded = {k: tuple(v) for k, v in entry["changed_fields"].items()}
        actual = {
            k: (legacy.get(k, ABSENT), current.get(k, ABSENT))
            for k in set(legacy) | set(current)
            if legacy.get(k, ABSENT) != current.get(k, ABSENT)
        }
        if recorded != actual:
            print(f"[FAIL] {case_id}: changed_fields mismatch "
                  f"(recorded {sorted(recorded)}, actual {sorted(actual)})")
            failures += 1

        status_file = REPO_ROOT / entry["status_path"]
        if not status_file.is_file():
            print(f"[FAIL] {case_id}: status file missing — {entry['status_path']}")
            failures += 1
            continue
        with open(status_file) as f:
            status = json.load(f)
        if status.get("config_signature") != entry["stored_signature"]:
            print(f"[FAIL] {case_id}: actual status config_signature "
                  f"{status.get('config_signature')} != manifest "
                  f"{entry['stored_signature']} — {entry['status_path']}")
            failures += 1
        if status.get("case_id") != case_id:
            print(f"[FAIL] {case_id}: actual status case_id "
                  f"{status.get('case_id')!r} differs — {entry['status_path']}")
            failures += 1
    return failures


def check_runners(manifest: dict) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))
    import importlib

    failures = 0
    covered = 0
    mismatched = set()
    for modname, listname in [
        ("run_single_particle_sedimentation_experiments", "SINGLE_PARTICLE_CASES"),
        ("run_two_particle_sedimentation_experiments", "TWO_PARTICLE_CASES"),
    ]:
        mod = importlib.import_module(modname)
        captured = {}
        real_make_signature = mod.make_signature

        def capture(payload, _c=captured, _real=real_make_signature):
            _c["payload"] = payload
            return _real(payload)

        mod.make_signature = capture
        for case in getattr(mod, listname):
            case_id = case["case_id"]
            status_file = case["output_dir"] / "status.json"
            if not status_file.exists():
                continue
            with open(status_file) as f:
                stored = json.load(f).get("config_signature")
            if not stored:
                continue
            covered += 1
            current = mod._config_signature(case, mod._build_config(case))
            if current != stored:
                mismatched.add(case_id)
            if case_id in manifest["runs"]:
                expected = manifest["runs"][case_id]["bundled_runner_payload"]
                # 서명은 JSON 인코딩 위에서 정의되므로 tuple/list 차이는 JSON 왕복으로 정규화해 비교
                live = json.loads(json.dumps(captured["payload"]))
                if live != expected:
                    print(f"[FAIL] {case_id}: bundled runner payload differs from manifest")
                    failures += 1
        mod.make_signature = real_make_signature

    manifest_set = set(manifest["runs"])
    print(f"runner coverage: {covered} runs with stored signatures, "
          f"{len(mismatched)} mismatched")
    if mismatched != manifest_set:
        extra = sorted(mismatched - manifest_set)
        missing = sorted(manifest_set - mismatched)
        if extra:
            print(f"[FAIL] mismatched runs absent from manifest: {extra}")
        if missing:
            print(f"[FAIL] manifest runs that recompute cleanly now: {missing}")
        failures += len(extra) + len(missing)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-runners", action="store_true",
                        help="also rebuild each case with the bundled runners "
                             "and check payloads and coverage accounting")
    args = parser.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)

    failures = check_manifest(manifest)
    n = len(manifest["runs"])
    print(f"manifest + status-file check: {n} entries, {failures} failure(s)")

    if args.with_runners:
        failures += check_runners(manifest)

    if failures:
        print(f"FAILED: {failures} check(s) did not pass")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
