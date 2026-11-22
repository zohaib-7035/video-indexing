# Hardware used for the full 50-file run

- Where: Kaggle notebook (free GPU)
- GPU: 2 × NVIDIA Tesla T4, 15 GB memory each (15360 MiB)
- CPU: Intel(R) Xeon(R) CPU @ 2.00GHz, 4 vCPUs
- RAM: 31 GB
- Disk: Kaggle cloud disk (SSD)
- OS: Ubuntu 24.04.4 LTS
- Full run time (start to finish, 50 files): 3.67 hours (about 3 h 40 min)

## Run details

- Files finished: 50/50
- Videos processed: 1650
- Error lines: 0
- Result lines: 461,757
- Settings: 5 frames per second sent to the model, batch size 32
- Both GPUs used at the same time, 25 files each (one process per GPU)
- Speed: about 92–95 frames per second per GPU

The 50 files were downloaded and processed one at a time with kaggle_job.py,
because the Kaggle disk cannot hold all 25 GB at once. The next file was
downloaded while the current one was being processed.
