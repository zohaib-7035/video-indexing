# Write-up -- I didn't use AI for this 

## 1. Your timings (100 words)

First version (every frame): 6 videos took 8.2 minutes for 44,465 frames, about 90 frames per second on one T4. I never ran every frame on all 50 files, so this is an estimate: about 11.5–14 million frames, so about 18–21 hours on 2 GPUs.

Final version (5 frames per second): one file (33 videos) took 8.2 minutes. All 50 files took 3.67 hours on 2 T4 GPUs, with 0 errors.

What helped most: sending only 5 frames per second to YOLO, so about 5–6 times fewer frames. Using both GPUs at the same time helped second.

## 2. Your pipeline (150 words)

1. Read parquet: a reader thread takes the mp4 bytes from each row.
2. Decode: decoder threads open each video with PyAV.
3. Rotate: if the video has a rotation tag, I turn the frame upright.
4. Pick frames: I keep the first frame in every 0.2 seconds, so 5 frames per second. I also resize them to YOLO's size (640) here, so frames are smaller.
5. YOLO: the main thread puts frames of the same size into batches of 32 and runs YOLO11l on the GPU in FP16.
6. Combine per second: an object (person, car, dog, laptop, cell phone) is present if it is in at least half of that second's frames. The people count is the median.
7. Write a line: one JSON line per second. A broken video gets an error line and the run continues.

## 3. Where the time goes (150 words)

The slowest part is YOLO on the GPU. I know this from the logs. The log prints the speed every 25 videos, and it stayed at about 88–96 frames per second (average 91.7 over all 50 files). The every-frame test ran at the same speed, about 90 frames per second. So the speed depends on how many frames go to the GPU, not on reading or decoding.

The time per file also follows the number of frames: file 2 had 58,128 frames and took 10.2 minutes, and file 7 had 37,831 frames and took 7.2 minutes. When I cut the frames by about 5 times, the time also dropped by about 5 times.

## 4. Checking correctness (100 words)

I made a reference by running every frame on 6 videos (--sample-fps 0). Then I ran the fast 5-fps version on the same file and compared both with check.py, second by second.

- Objects F1: 0.968 (need 0.95)
- People count exact: 0.928 (need 0.85)
- People count within 1: 0.996 (need 0.95)
- No missing seconds

Result: PASS. Skipping frames gives almost the same answers as using every frame.

## 5. Working with AI (100 words)

The AI first gave me timing estimates for the full run, but they were only rough guesses. I did not trust them. I added a test mode that times one real file on the real Kaggle GPUs. One file took 8.2 minutes, so I predicted 8.2 × 50 ÷ 2 ≈ 3.4 hours for the full run. The real run took 3.67 hours, so the measured estimate was close. I learned to measure a small real piece first instead of trusting a guess.

## 6. Next steps (100 words)

1. TensorRT: the GPU is the slowest part, so making YOLO faster gives the biggest gain.
2. Decode video on the GPU (NVDEC): when YOLO gets faster, decoding on the CPU could become the slow part.
3. Try 3 frames per second and check with check.py how much accuracy drops.
4. Test correctness on more videos (30–50 instead of 6), including dark, fast and crowded videos.
