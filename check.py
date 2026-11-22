#!/usr/bin/env python3
"""Compare a results file against a reference made with --sample-fps 0 (every frame).

    python check.py reference.jsonl results.jsonl

Uses the thresholds from the brief: objects F1 >= 0.95, people exact >= 85%,
people within 1 >= 95%. Only videos present in the reference are compared.
"""
import gzip
import json
import sys


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    secs, errors = {}, set()
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
            else:
                secs[(d["video"], d["second"])] = (set(d["objects"]), d["people"])
    return secs, errors


def main(ref_path, res_path):
    ref, ref_err = load(ref_path)
    res, res_err = load(res_path)
    ref_videos = {v for v, _ in ref} | ref_err
    tp = fp = fn = 0
    exact = within1 = total = 0
    missing = 0
    for key, (r_obj, r_ppl) in ref.items():
        total += 1
        if key not in res:
            missing += 1
            fn += len(r_obj)
            continue
        o_obj, o_ppl = res[key]
        tp += len(r_obj & o_obj)
        fp += len(o_obj - r_obj)
        fn += len(r_obj - o_obj)
        exact += o_ppl == r_ppl
        within1 += abs(o_ppl - r_ppl) <= 1
    extra = sum(1 for (v, s) in res if v in ref_videos and (v, s) not in ref)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ex = exact / total if total else 0
    w1 = within1 / total if total else 0
    print(f"videos compared:      {len(ref_videos)}")
    print(f"seconds in reference: {total}  missing in results: {missing}  extra: {extra}")
    print(f"error lines  ref: {len(ref_err)}  results: {len(res_err & ref_videos)}")
    print(f"objects  P={prec:.4f} R={rec:.4f} F1={f1:.4f}   (need >= 0.95)")
    print(f"people exact     {ex:.4f}   (need >= 0.85)")
    print(f"people within 1  {w1:.4f}   (need >= 0.95)")
    ok = f1 >= 0.95 and ex >= 0.85 and w1 >= 0.95 and missing == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
