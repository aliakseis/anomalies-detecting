# anomalies-detecting

![alt text](demo.jpg)

```
./anomalies-detecting ./source.jpg 20,100,280,200,20,400,600,200,600,20,100,600,850,600,900,20 ./output.jpg
```
A fast, KD-tree–based image anomaly-detection tool implemented in C++ using **OpenCV** and **nanoflann**.

The algorithm searches for unusual local patterns (patches) in an image.  
Each image patch is embedded into a high-dimensional feature space and compared against all other patches using efficient nearest-neighbor search.  
Patches that significantly deviate from their neighbors are marked as anomalies.

---

## Example Usage

```
./anomalies-detecting ./source.jpg 20,100,280,200,20,400,600,200,600,20,100,600,850,600,900,20 ./output.jpg
```

---

## 1. Overview

This project implements **local anomaly detection using self-similarity**:

1. Convert image to grayscale  
2. Extract sliding-window patches  
3. Normalize each patch vector  
4. Build KD-tree (nanoflann)  
5. For each ROI patch:  
   - Query nearest neighbors  
   - Compute anomaly score  
6. Generate anomaly mask  

---

## 2. Building

### Requirements

- C++17  
- CMake ≥ 3.10  
- OpenCV  
- nanoflann (header-only)

### Build

```
mkdir build
cd build
cmake ..
cmake --build .
```

---

## 3. Full Algorithm Description

### 3.1 Patch Extraction

Sliding window of `DIMENSION × DIMENSION` extracts all patches.

### 3.2 Patch Feature Vector

Each patch → normalized feature vector.  
Includes brightness coefficient.

### 3.3 KD-Tree Construction

All patches inserted into nanoflann KD-tree for fast nearest-neighbor search.

### 3.4 Nearest Neighbor Search

KNN search with filtering of nearby spatial neighbors.

### 3.5 Anomaly Scoring

`N-th nearest neighbor distance` used as anomaly score.  
Higher distance = more anomalous.

### 3.6 Multithreading

Image is split into vertical strips processed by multiple threads.

---

## 4. Output

`getAnomalies()` returns a grayscale anomaly mask.

---

## 5. Files

| File | Description |
|------|-------------|
| main.cpp | CLI runner |
| anomalies-detecting.cpp/.h | Core algorithm |
| CMakeLists.txt | Build config |

---

## 6. License

MIT License
