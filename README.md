# Color Theme Extractor

Generate a usable UI theme from any image using color clustering + scoring.

This project has:

- A reusable extraction engine in `theme_extractor.py`
- A Streamlit frontend in `app.py`
- A CLI smoke entrypoint in `main.py`

## Features

- Extracts dominant image palette with K-Means clustering in LAB space
- Scores each cluster using population, saturation, and lightness penalty
- Selects the best accent color automatically
- Synthesizes a UI-friendly theme from the accent color
- Streamlit UI with:
  - image upload and preview
  - palette + accent priority rendering
  - detailed algorithm breakdown
  - JSON preview popover
  - copy/download theme JSON

## Project Structure

```text
color-theme-extractor/
├── app.py               # Streamlit frontend
├── theme_extractor.py   # Core algorithm engine
├── main.py              # CLI smoke test
├── requirements.txt
├── data/
│   └── test.png
└── .gitignore
```

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Then upload an image in the browser UI.

## Run the CLI Example

```bash
python main.py
```

This uses `data/test.png` and prints:

- generated theme
- selected accent color
- dominance-sorted palette

## How the Algorithm Works

### 1) Preprocess

- Load image as RGB
- Resize to `64x64` by default for stable speed and signal quality

### 2) Convert Color Spaces

- Convert RGB -> LAB for clustering (better perceptual distance)
- Convert RGB -> HSV for saturation measurement

### 3) Cluster Colors

- Run K-Means on LAB pixels with configurable `k` (default `6`)
- Get cluster centers and pixel labels

### 4) Sort by Dominance

- Count pixels per cluster
- Sort clusters by count descending so palette starts with dominant colors

### 5) Compute Cluster Metrics

For each cluster:

- Population: `count_i / total_pixels`
- Saturation: mean HSV saturation of that cluster
- Lightness: LAB `L` from cluster center

### 6) Score Clusters

If penalty is enabled:

- apply `0.5` penalty when lightness is too dark/light (`L < 20` or `L > 85`)

Score formula:

```text
score_i = (population_weight * population_i + saturation_weight * saturation_i) * penalty_i
```

### 7) Pick Accent + Generate Theme

- Accent = highest score cluster center
- Derive `primary`, `secondary`, `background_light`, `background_dark`, `surface`
- Choose `text_on_primary` using luminance contrast rule

## Engine API

Create extractor:

```python
from theme_extractor import ThemeExtractor

extractor = ThemeExtractor(
    k=6,
    resize=(64, 64),
    population_weight=0.5,
    saturation_weight=0.8,
    penalty_enabled=True,
)
```

Extract from path/file-like object:

```python
result = extractor.extract("data/test.png")
```

Returns a dictionary with keys:

- `config`
- `images` (`original`, `resized`, `clustered` arrays)
- `distribution` (`lab_ab`, `rgb_norm`)
- `palette_rgb`
- `accent_rgb`
- `cluster_details`
- `theme`

Theme shape:

```json
{
  "primary": [r, g, b],
  "secondary": [r, g, b],
  "background_light": [r, g, b],
  "background_dark": [r, g, b],
  "surface": [r, g, b],
  "text_on_primary": [r, g, b]
}
```

## Frontend Controls

In the Streamlit sidebar:

- `Clusters (K)`
- `Population Weight`
- `Saturation Weight`
- `Penalize too dark/light colors`

## Notes

- The frontend is intentionally presentation-only; algorithm logic lives in `theme_extractor.py`.
- The UI footer credits: `Made by Raunak | GitHub: RaunakDiesFromCode`.

## License

Add a license file if you plan to publish/distribute this repository.
