# Music Recommendation System using K-Means Clustering

A **Machine Learning** project that builds a content-based music recommendation system using **K-Means clustering** on audio features from the Spotify dataset.

---

## Overview

This system clusters songs based on 9 audio features and recommends similar songs by finding nearest neighbors **within the same cluster**. Songs in the same cluster share similar musical characteristics (energy, mood, acousticness, etc.), making in-cluster recommendations more accurate than brute-force search across the full dataset.

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | Spotify Dataset (Kaggle) |
| Features | 9 audio features per track |
| Target | No label — unsupervised clustering |

### Audio Features Used

| Feature | Description |
|---------|-------------|
| `danceability` | How suitable a track is for dancing (0.0–1.0) |
| `energy` | Intensity and activity level (0.0–1.0) |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words (0.0–1.0) |
| `acousticness` | Confidence the track is acoustic (0.0–1.0) |
| `instrumentalness` | Predicts whether a track has no vocals (0.0–1.0) |
| `liveness` | Presence of a live audience (0.0–1.0) |
| `valence` | Musical positiveness — happy vs. sad (0.0–1.0) |
| `tempo` | Beats per minute (BPM) |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| Pandas & NumPy | Data loading and preprocessing |
| Scikit-learn | K-Means, PCA, StandardScaler, NearestNeighbors, Silhouette Score |
| Matplotlib & Seaborn | Visualizations |
| Google Colab | Cloud execution environment |

---

## Project Pipeline

```
Spotify Dataset (CSV)
    ↓
Data Loading & Cleaning (dropna, dropDuplicates)
    ↓
Feature Scaling (StandardScaler — zero mean, unit variance)
    ↓
Exploratory Data Analysis (distributions, correlation heatmap)
    ↓
Dimensionality Reduction (PCA: 9D → 2D for visualization)
    ↓
Optimal k Selection (Elbow Method + Silhouette Score)
    ↓
K-Means Clustering (k=3, k-means++ initialization)
    ↓
Cluster Profiling (musical characteristics per cluster)
    ↓
Recommendation (NearestNeighbors within same cluster)
```

---

## Clusters Discovered

| Cluster | Label | Characteristics |
|---------|-------|-----------------|
| 0 | Energetic / Electronic | High energy, loud, danceable, low acousticness |
| 1 | Acoustic / Calm | High acousticness, low energy, softer tempo |
| 2 | Balanced / Mixed | Mid-range across all features, versatile mood |

---

## How Recommendations Work

1. **Scale** the input song's features using the trained StandardScaler
2. **Predict** which cluster the song belongs to using K-Means
3. **Find** the k nearest neighbors **within that cluster only** using Euclidean distance (NearestNeighbors)
4. **Return** the top-N most similar songs

---

## What's Inside the Notebook

| Section | Details |
|---------|---------|
| Data Loading | Upload CSV on Colab or local path fallback |
| Preprocessing | Drop nulls/duplicates, scale 9 features |
| EDA | Feature distribution histograms, correlation heatmap |
| PCA Visualization | 2D scatter of all songs before clustering |
| Elbow Method | WCSS plot for k=1 to 10 — optimal at k=3 |
| K-Means | Fit with k=3, k-means++ init, label assignment |
| Silhouette Score | Quantitative clustering quality metric |
| Cluster Profiles | Bar charts comparing avg feature values per cluster |
| Recommendation | `get_recommendations(track_name, n=5)` function |
| Output | Song name, artist, cluster, and similarity score |

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File → Upload Notebook** and upload `music_recommendation_kmeans.ipynb`
3. Download the Spotify dataset from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
4. Upload the CSV when prompted by the file upload cell
5. Click **Runtime → Run All**

### Option 2 — Jupyter Notebook (Local)

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Run:**
```bash
jupyter notebook music_recommendation_kmeans.ipynb
```

> **Note:** Replace the Colab upload cell with:
> ```python
> df = pd.read_csv("your_spotify_dataset.csv")
> ```

---

## Sample Output

```
Recommendations for: "Blinding Lights"
Cluster: 0 (Energetic / Electronic)

  Track Name              Artist          Cluster  Similarity Score
  Starboy                 The Weeknd      0        0.9821
  Save Your Tears         The Weeknd      0        0.9764
  In Your Eyes            The Weeknd      0        0.9703
  Levitating              Dua Lipa        0        0.9681
  Don't Start Now         Dua Lipa        0        0.9650
```

---

## Results

| Metric | Value |
|--------|-------|
| Optimal Clusters (k) | 3 |
| Silhouette Score | ~0.18 |
| Recommendation Method | NearestNeighbors (Euclidean) within cluster |
| Features Used | 9 Spotify audio features |

---

## Academic Details

- **Institution:** REVA University, Bengaluru
- **Course:** Machine Learning Applications (MLA)
- **Academic Year:** 2024–25
- **Semester:** 4th

---

## Author

**Thomson Sunny**  
GitHub: [@Thomson-4](https://github.com/Thomson-4)
