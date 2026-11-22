#!/usr/bin/env python3
"""Video indexing with YOLO11l.

For each second of each video (one video per row of each .parquet file) record which
of five objects are present and how many people are visible.

    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline:
  reader thread  -> reads mp4 bytes row by row from the parquet files
  decoder threads -> decode with PyAV, rotate, keep `--sample-fps` frames per second,
                     resize to the letterbox size YOLO would use anyway
  main thread    -> batches frames of the same shape, runs YOLO11l on the GPU,
                    aggregates per second (majority vote / median) and writes JSON lines
"""
import argparse
import glob
import io
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from fractions import Fraction

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

# COCO class ids used by yolo11l.pt
CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
CLASS_IDS = list(CLASSES)
OBJ_ORDER = ["person", "car", "dog", "laptop", "cell phone"]
IMGSZ = 640
CONF = 0.4

_END = object()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Reading parquet rows
# ---------------------------------------------------------------------------
def iter_videos(paths, limit=0):
    """Yield (video_id, mp4_bytes or None, error or None) for every row."""
    for item in _iter_videos(paths):
        yield item
        limit -= 1
        if limit == 0:
            return


def _iter_videos(paths):
    for path in paths:
        name = os.path.basename(path)
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:  # unreadable file: we cannot even count rows
            yield f"{name}:0", None, f"could not open parquet file: {type(e).__name__}"
            continue
        row = 0
        for rg in range(pf.num_row_groups):
            n_rows = pf.metadata.row_group(rg).num_rows
            try:
                table = pf.read_row_group(rg, columns=["mp4"])
                col = table.column("mp4")
            except Exception as e:
                for i in range(n_rows):
                    yield f"{name}:{row + i}", None, f"could not read row: {type(e).__name__}"
                row += n_rows
                continue
            for i in range(len(col)):
                try:
                    data = col[i].as_py()
                    if isinstance(data, dict):  # some writers store {"bytes": ...}
                        data = data.get("bytes")
                    if not data:
                        yield f"{name}:{row}", None, "empty mp4 column"
                    else:
                        yield f"{name}:{row}", data, None
                except Exception as e:
                    yield f"{name}:{row}", None, f"could not read row: {type(e).__name__}"
                row += 1
            del table, col


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def letterbox_resize(img):
    """Resize exactly like ultralytics LetterBox does before padding.

    YOLO then sees an image whose long side is already 640, so its own letterbox only
    pads, and the network input is identical to running on the full-size frame.
    Doing it here (in a decoder thread) keeps the main thread free and makes the
    frames we queue much smaller.
    """
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def decode_video(vid, data, sample_fps, out_q, chunk=16):
    """Decode one video and push frames to out_q.

    Messages: ("frames", vid, [(second, img), ...]) and finally
              ("end", vid, last_second, error_or_None)
    If sample_fps <= 0 every frame is kept (reference mode).
    """
    last_second = -1
    buf = []
    error = None
    container = None
    try:
        container = av.open(io.BytesIO(data), mode="r")
        if not container.streams.video:
            raise RuntimeError("no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        rate = stream.average_rate or stream.guessed_rate or 25
        first_pts = None
        idx = 0
        last_slot = None
        for frame in container.decode(stream):
            # timestamp relative to the first decoded frame, as an exact fraction
            if frame.pts is not None and tb is not None:
                if first_pts is None:
                    first_pts = frame.pts
                t = Fraction(frame.pts - first_pts) * Fraction(tb)
            else:
                t = Fraction(idx) / Fraction(rate)
            idx += 1
            if t < 0:
                t = Fraction(0)
            sec = math.floor(t)
            if sec > last_second:
                last_second = sec

            if sample_fps > 0:
                slot = math.floor(t * sample_fps)  # keep first frame of each 1/sample_fps slot
                if slot == last_slot:
                    continue
                last_slot = slot

            img = frame.to_ndarray(format="bgr24")
            rot = frame.rotation or 0
            k = int(round(rot / 90.0)) % 4
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
            buf.append((sec, letterbox_resize(img)))
            if len(buf) >= chunk:
                out_q.put(("frames", vid, buf))
                buf = []
        if idx == 0:
            raise RuntimeError("no frames decoded")
    except Exception as e:
        error = f"could not decode video: {type(e).__name__}"
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
    if buf:
        out_q.put(("frames", vid, buf))
    out_q.put(("end", vid, last_second, error))


def reader_thread(paths, task_q, n_workers, limit):
    try:
        for item in iter_videos(paths, limit):
            task_q.put(item)
    finally:
        for _ in range(n_workers):
            task_q.put(_END)


def decoder_worker(task_q, out_q, sample_fps):
    while True:
        item = task_q.get()
        if item is _END:
            out_q.put(("worker_done",))
            return
        vid, data, err = item
        if err is not None:
            out_q.put(("end", vid, -1, err))
            continue
        decode_video(vid, data, sample_fps, out_q)
        del data


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def median_round_down(counts):
    s = sorted(counts)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def video_lines(vid, per_sec, last_second, error):
    """per_sec: second -> list of (people_count, frozenset(object names)) per frame."""
    lines = []
    # On a mid-video failure keep only the seconds that were fully decoded.
    upto = last_second if error is None else last_second - 1
    for s in range(0, upto + 1):
        frames = per_sec.get(s, [])
        n = len(frames)
        if n == 0:
            objs, people = [], 0
        else:
            cnt = defaultdict(int)
            for _, names in frames:
                for nm in names:
                    cnt[nm] += 1
            objs = [nm for nm in OBJ_ORDER if 2 * cnt[nm] >= n]
            people = median_round_down([p for p, _ in frames])
        lines.append(json.dumps({"video": vid, "second": s, "objects": objs, "people": people}))
    if error is not None:
        lines.append(json.dumps({"video": vid, "error": error}))
    return lines


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="output .jsonl file")
    ap.add_argument("--sample-fps", type=float, default=float(os.environ.get("SAMPLE_FPS", 5)),
                    help="frames per second to run the model on (0 = every frame)")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 32)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 0)),
                    help="decoder threads (0 = auto)")
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS", "yolo11l.pt"))
    ap.add_argument("--device", default=os.environ.get("DEVICE", None))
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (for testing)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    log(f"{len(paths)} parquet files, sample_fps={args.sample_fps}, batch={args.batch}")

    import torch
    from ultralytics import YOLO

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    half = torch.cuda.is_available() and str(device) != "cpu"
    model = YOLO(args.weights)
    pred_kw = dict(imgsz=IMGSZ, conf=CONF, iou=0.7, classes=CLASS_IDS, half=half,
                   device=device, verbose=False)
    # warm-up
    model.predict([np.zeros((360, 640, 3), np.uint8)], **pred_kw)

    n_workers = args.workers or max(2, min(8, (os.cpu_count() or 4) - 1))
    task_q = queue.Queue(maxsize=n_workers * 2)
    out_q = queue.Queue(maxsize=n_workers * 8)
    threading.Thread(target=reader_thread, args=(paths, task_q, n_workers, args.limit), daemon=True).start()
    for _ in range(n_workers):
        threading.Thread(target=decoder_worker, args=(task_q, out_q, args.sample_fps),
                         daemon=True).start()

    per_video = defaultdict(lambda: defaultdict(list))  # vid -> sec -> [(people, names)]
    pending = defaultdict(int)  # vid -> frames queued but not yet inferred
    ended = {}  # vid -> (last_second, error)
    buffers = defaultdict(list)  # image shape -> [(vid, sec, img)]
    n_buffered = 0
    workers_left = n_workers
    stats = dict(videos=0, frames=0, errors=0, lines=0)
    t0 = time.time()

    out = open(args.output, "w")

    def infer(items):
        nonlocal n_buffered
        imgs = [it[2] for it in items]
        results = model.predict(imgs, **pred_kw)
        for (vid, sec, _), r in zip(items, results):
            cls = r.boxes.cls.int().tolist() if r.boxes is not None else []
            people = sum(1 for c in cls if c == 0)
            names = frozenset(CLASSES[c] for c in cls)
            per_video[vid][sec].append((people, names))
            pending[vid] -= 1
        n_buffered -= len(items)
        stats["frames"] += len(items)

    def flush(force=False):
        for shape in list(buffers):
            b = buffers[shape]
            while len(b) >= args.batch or (force and b):
                take, buffers[shape] = b[:args.batch], b[args.batch:]
                b = buffers[shape]
                infer(take)
            if not b:
                del buffers[shape]

    def finalize_ready():
        for vid in [v for v in ended if pending[v] == 0]:
            last_second, error = ended.pop(vid)
            lines = video_lines(vid, per_video.pop(vid, {}), last_second, error)
            if lines:
                out.write("\n".join(lines) + "\n")
            stats["videos"] += 1
            stats["lines"] += len(lines)
            if error:
                stats["errors"] += 1
                log(f"ERROR {vid}: {error}")
            pending.pop(vid, None)
            if stats["videos"] % 25 == 0:
                el = time.time() - t0
                log(f"{stats['videos']} videos, {stats['frames']} frames, "
                    f"{stats['frames'] / max(el, 1e-6):.1f} frames/s, {el / 60:.1f} min")
                out.flush()

    while workers_left > 0:
        try:
            msg = out_q.get(timeout=0.5)
        except queue.Empty:
            flush(force=True)  # GPU idle: run whatever we have
            finalize_ready()
            continue
        kind = msg[0]
        if kind == "frames":
            _, vid, frames = msg
            for sec, img in frames:
                buffers[img.shape].append((vid, sec, img))
            pending[vid] += len(frames)
            n_buffered += len(frames)
            flush()
        elif kind == "end":
            _, vid, last_second, error = msg
            ended[vid] = (last_second, error)  # written once all its frames are inferred
        elif kind == "worker_done":
            workers_left -= 1
        finalize_ready()

    flush(force=True)
    finalize_ready()
    out.close()
    el = time.time() - t0
    log(f"done: {stats['videos']} videos, {stats['errors']} errors, {stats['lines']} lines, "
        f"{stats['frames']} frames in {el / 60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
