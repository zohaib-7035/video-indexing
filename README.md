# Video indexing with YOLO11l

For each second of each video in a folder of FineVideo parquet files, `run.py` records which
of person, car, dog, laptop and cell phone are present and how many people are visible.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12)

System packages (needed by OpenCV):

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

Python packages:

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (run once, needs internet)

Downloads the official `yolo11l.pt` weights into the repository folder:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

Optional flags: `--sample-fps` (frames per second sent to the model, default 5; 0 = every frame),
`--batch` (default 32), `--workers` (decoder threads, default auto), `--limit N` (first N videos only).

## How it works

- A reader thread reads the `mp4` bytes from each parquet row.
- Decoder threads decode each video with PyAV, apply the rotation tag, compute each frame's
  time from the first decoded frame, keep `--sample-fps` evenly spaced frames per second and
  resize them to the letterbox size YOLO uses (identical model input, smaller frames).
- The main thread batches frames of the same shape and runs YOLO11l (imgsz 640, conf 0.4,
  IoU 0.7, FP16 on GPU), then per second marks an object present if it is in at least half the
  sampled frames and takes the median people count (rounded down on a tie).
- Damaged videos or files produce an error line; the run continues.

## Checking correctness

`check.py` compares results with an every-frame reference using the brief's thresholds:

```bash
python run.py --input data --output ref.jsonl --sample-fps 0 --limit 10
python run.py --input data --output fast.jsonl
python check.py ref.jsonl fast.jsonl
```

## Running the 50 files on Kaggle

`kaggle_notebook.ipynb` + `kaggle_job.py` download, process and delete one parquet file at a
time on each GPU. Results are in `results.jsonl.gz`.
