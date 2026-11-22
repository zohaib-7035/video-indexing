# Write-up -- I didn't use AI for this

## 1. Timings
- First version: 8.2 minutes for 6 videos.
- Final version: 5 FPS processing.
- All 50 files completed in 3.67 hours using 2 T4 GPUs.
- No errors.

## 2. Pipeline
1. Read videos from Parquet.
2. Decode videos using PyAV.
3. Correct video rotation.
4. Extract 5 frames per second.
5. Resize frames to 640.
6. Run YOLO11l in batches of 32 using FP16.
7. Combine detections for each second.
8. Save results as JSON.

## 3. Performance
- YOLO GPU processing was the main bottleneck.
- Average speed was 91.7 FPS.
- Processing fewer frames reduced execution time.
- Using 2 GPUs improved total processing time.

## 4. Accuracy
Compared every-frame processing with 5 FPS sampling.

- Object F1: 0.968
- Exact people count: 0.928
- People count within 1: 0.996
- Missing seconds: 0

**Result: PASS.** All accuracy requirements were met.

## 5. AI Usage
- AI helped with initial estimates and implementation ideas.
- I tested one real file on Kaggle GPUs.
- Based on that test, I estimated the full runtime.
- Actual runtime was 3.67 hours, close to my estimate.

## 6. Next Steps
1. Use TensorRT to speed up YOLO.
2. Try GPU video decoding with NVDEC.
3. Test 3 FPS sampling.
4. Check accuracy on 30–50 videos.
