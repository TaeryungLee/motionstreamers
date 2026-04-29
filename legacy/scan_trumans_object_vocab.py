from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


ACTION_OBJECT_PATTERNS = (
    r"\bthe ([a-z0-9 _/-]+)$",
    r"\bthe ([a-z0-9 _/-]+)\b",
    r"\bon the ([a-z0-9 _/-]+)$",
    r"\bwith the ([a-z0-9 _/-]+)$",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scan TRUMANS object names across actions, Object_all, object_list, and obj_list.txt.")
    parser.add_argument("--trumans-root", type=Path, default=Path("data/raw/trumans"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trumans_object_vocab_scan.json"))
    return parser.parse_args()


def extract_action_object(text: str) -> str | None:
    lower = text.strip().lower()
    if lower in {"sit down", "stand up", "lie down", "squat down", "kneel down", "crouch down"}:
        return None
    for pattern in ACTION_OBJECT_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            return match.group(1).strip(" .")
    return None


def canonicalize_name(raw: str) -> str:
    text = raw.lower().strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = (
        ("fridge", "refrigerator"),
        ("water bottle", "bottle"),
        ("bottle root", "bottle"),
        ("cup root", "cup"),
        ("vase root", "vase"),
        ("chair base", "chair"),
        ("chair seat", "chair"),
        ("movable chair", "chair"),
        ("sofa couch", "sofa"),
    )
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def scan_actions(actions_dir: Path) -> tuple[Counter, Counter]:
    text_counter = Counter()
    object_counter = Counter()
    for path in actions_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            text = parts[2].strip()
            text_counter[text] += 1
            obj = extract_action_object(text)
            if obj:
                object_counter[canonicalize_name(obj)] += 1
    return text_counter, object_counter


def scan_object_pose_names(object_pose_dir: Path) -> Counter:
    counter = Counter()
    for path in object_pose_dir.glob("*.npy"):
        data = np.load(path, allow_pickle=True).item()
        for name in data.keys():
            counter[canonicalize_name(name)] += 1
    return counter


def scan_object_list_npy(path: Path) -> Counter:
    counter = Counter()
    values = np.load(path, allow_pickle=True)
    for value in values.tolist():
        counter[canonicalize_name(str(value))] += 1
    return counter


def scan_obj_list_txt(recordings_blend_dir: Path) -> Counter:
    counter = Counter()
    for path in recordings_blend_dir.glob("*/obj_list.txt"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            counter[canonicalize_name(line)] += 1
    return counter


def top_items(counter: Counter, k: int = 200) -> list[dict]:
    return [{"name": name, "count": count} for name, count in counter.most_common(k)]


def main():
    args = parse_args()
    root = args.trumans_root

    action_text_counter, action_object_counter = scan_actions(root / "Actions")
    object_pose_counter = scan_object_pose_names(root / "Object_all" / "Object_pose")
    object_list_counter = scan_object_list_npy(root / "object_list.npy")
    obj_list_counter = scan_obj_list_txt(root / "Recordings_blend")

    merged = Counter()
    for counter in (action_object_counter, object_pose_counter, object_list_counter, obj_list_counter):
        merged.update(counter)

    payload = {
        "actions_total": sum(action_text_counter.values()),
        "unique_action_texts": len(action_text_counter),
        "top_action_texts": top_items(action_text_counter, 100),
        "top_action_objects": top_items(action_object_counter, 200),
        "top_object_all_names": top_items(object_pose_counter, 300),
        "top_object_list_names": top_items(object_list_counter, 200),
        "top_obj_list_txt_names": top_items(obj_list_counter, 300),
        "top_merged_names": top_items(merged, 400),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
