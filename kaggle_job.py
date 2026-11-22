#!/usr/bin/env python3
"""Process the 50 FineVideo parquet files one at a time on a machine with little disk
(Kaggle / Colab): download a file, run run.py on it, delete it, move on.
The next file is downloaded while the current one is processed.

    HF_TOKEN=... python kaggle_job.py --gpu 0 --shard 0 --nshards 2 --out results_parts

Already finished files (a .done marker exists) are skipped, so you can re-run after a
disconnect and it continues where it stopped.
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

from huggingface_hub import hf_hub_download

REPO = "HuggingFaceFV/finevideo"
HERE = os.path.dirname(os.path.abspath(__file__))


def fname(i):
    return f"train-{i:05d}-of-01357.parquet"


def download(i, tmp, token):
    d = os.path.join(tmp, f"f{i:05d}")
    os.makedirs(d, exist_ok=True)
    for attempt in range(5):
        try:
            p = hf_hub_download(REPO, f"data/{fname(i)}", repo_type="dataset",
                                local_dir=d, token=token)
            return d, p
        except Exception as e:
            print(f"download {i} failed (attempt {attempt + 1}): {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download file {i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=49)
    ap.add_argument("--out", default="results_parts")
    ap.add_argument("--tmp", default="/tmp/finevideo")
    ap.add_argument("--sample-fps", default="5")
    ap.add_argument("--batch", default="32")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.out, exist_ok=True)
    files = [i for i in range(args.first, args.last + 1) if i % args.nshards == args.shard]
    todo = [i for i in files if not os.path.exists(os.path.join(args.out, fname(i) + ".done"))]
    print(f"[gpu {args.gpu}] {len(todo)} files to do: {todo}", flush=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    t_start = time.time()
    prefetch = {}

    def start_prefetch(i):
        box = {}

        def run():
            try:
                box["res"] = download(i, args.tmp, token)
            except Exception as e:
                box["err"] = e
        th = threading.Thread(target=run, daemon=True)
        th.start()
        prefetch[i] = (th, box)

    if todo:
        start_prefetch(todo[0])
    for k, i in enumerate(todo):
        th, box = prefetch.pop(i)
        th.join()
        if k + 1 < len(todo):
            start_prefetch(todo[k + 1])
        if "err" in box:
            print(f"[gpu {args.gpu}] SKIP {fname(i)}: {box['err']}", flush=True)
            continue
        d, _ = box["res"]
        in_dir = os.path.join(d, "data")
        part = os.path.join(args.out, fname(i) + ".jsonl")
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--input", in_dir,
                            "--output", part, "--sample-fps", args.sample_fps,
                            "--batch", args.batch], env=env)
        dt = time.time() - t0
        if r.returncode == 0:
            with open(os.path.join(args.out, fname(i) + ".done"), "w") as f:
                f.write(f"{dt:.1f}\n")
        shutil.rmtree(d, ignore_errors=True)
        el = time.time() - t_start
        print(f"[gpu {args.gpu}] {fname(i)} done in {dt / 60:.1f} min "
              f"({k + 1}/{len(todo)}, total {el / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
