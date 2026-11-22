# Hardware used for the full 50-file run

- Where: Kaggle notebook
- GPU: 2 × NVIDIA Tesla T4, 16 GB each   <!-- check run_info.txt -->
- CPU: FILL IN (model name from run_info.txt), 4 vCPUs
- RAM: FILL IN GB
- Disk: FILL IN (SSD)
- OS: FILL IN (from run_info.txt)
- Full run time (start to finish, 50 files): FILL IN hours, with SAMPLE_FPS = FILL IN

The files were downloaded and processed one at a time (`kaggle_job.py`), because the
Kaggle disk cannot hold all 25 GB at once. Download time overlapped with processing.
