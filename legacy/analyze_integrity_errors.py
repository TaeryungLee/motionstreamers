from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
TRUMANS_ROOT = PROJECT_ROOT / "data" / "raw" / "trumans"
LINGO_ROOT = PROJECT_ROOT / "data" / "raw" / "lingo" / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose integrity report errors per sample.")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "data" / "preprocessed" / "_integrity_report.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "preprocessed" / "_integrity_diagnosis.json")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def trumans_action_path(sequence_id: str) -> Path:
    base = sequence_id.split("_augment", 1)[0]
    primary = TRUMANS_ROOT / "Actions" / f"{base}.txt"
    if primary.exists():
        return primary
    return TRUMANS_ROOT / "Actions2" / f"{base}.txt"


def diagnose_trumans_error(err: dict) -> dict:
    msg = err.get("msg", "")
    seq_id = err.get("sequence_id")
    out = {"reason": None}
    if not seq_id:
        return out

    if "segment_list missing or empty" in msg:
        action_path = trumans_action_path(seq_id)
        if not action_path.exists():
            out["reason"] = "missing Actions txt"
        else:
            lines = [line for line in action_path.read_text().splitlines() if line.strip()]
            if len(lines) == 0:
                out["reason"] = "empty Actions txt"
            else:
                out["reason"] = "segment list empty despite Actions txt (parse mismatch)"
        out["action_path"] = str(action_path)
        return out

    if "plot_path missing" in msg:
        out["reason"] = "plot image missing"
        return out

    if "human_motion_ref" in msg:
        out["reason"] = "human_motion_ref invalid"
        return out

    if "segment range invalid" in msg or "interaction outside segment" in msg:
        out["reason"] = "segment frame indices invalid"
        return out

    if "object_id" in msg:
        out["reason"] = "object reference missing from object_list"
        return out

    if "sequence_ids" in msg:
        out["reason"] = "scene/sequence list mismatch"
        return out

    return out


def diagnose_lingo_error(err: dict) -> dict:
    msg = err.get("msg", "")
    out = {"reason": None}
    if "segment_list missing or empty" in msg:
        out["reason"] = "segment list empty (lingo sequence)"
    elif "human_motion_ref" in msg:
        out["reason"] = "human_motion_ref invalid"
    elif "plot_path missing" in msg:
        out["reason"] = "plot image missing"
    elif "segment range invalid" in msg or "interaction outside segment" in msg:
        out["reason"] = "segment frame indices invalid"
    elif "sequence_ids" in msg:
        out["reason"] = "scene/sequence list mismatch"
    return out


def main() -> None:
    args = parse_args()
    report = load_json(args.report)
    diagnosis: dict[str, Any] = {"report": str(args.report), "datasets": []}

    for dataset_report in report.get("datasets", []):
        dataset = dataset_report.get("dataset")
        errors = dataset_report.get("errors", [])
        enriched = []
        reason_counter = Counter()
        per_sequence = defaultdict(list)
        for err in errors:
            err_copy = dict(err)
            if dataset == "trumans":
                extra = diagnose_trumans_error(err_copy)
            else:
                extra = diagnose_lingo_error(err_copy)
            err_copy.update(extra)
            if extra.get("reason"):
                reason_counter[extra["reason"]] += 1
            key = err_copy.get("sequence_id") or err_copy.get("scene_id") or "unknown"
            per_sequence[key].append(err_copy)
            enriched.append(err_copy)

        diagnosis["datasets"].append(
            {
                "dataset": dataset,
                "stats": dataset_report.get("stats", {}),
                "reason_counts": reason_counter.most_common(),
                "errors": enriched,
                "per_sequence": per_sequence,
            }
        )

    args.out.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(f"Diagnosis written to: {args.out}")
    for d in diagnosis["datasets"]:
        print(d["dataset"], "top reasons:", d["reason_counts"][:5])


if __name__ == "__main__":
    main()
