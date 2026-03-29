import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2hsv, lab2rgb, hsv2rgb
from sklearn.cluster import KMeans


class ThemeExtractor:
    def __init__(
        self,
        k=6,
        resize=(64, 64),
        population_weight=0.5,
        saturation_weight=0.8,
        penalty_enabled=True,
    ):
        self.k = k
        self.resize = resize
        self.population_weight = population_weight
        self.saturation_weight = saturation_weight
        self.penalty_enabled = penalty_enabled

    @staticmethod
    def _to_uint8(rgb):
        return np.clip(np.round(rgb * 255), 0, 255).astype(int)

    def _generate_theme(self, accent_rgb):
        accent_hsv = rgb2hsv(accent_rgb.reshape(1, 1, 3)).reshape(3)
        h, s, v = accent_hsv

        primary = accent_rgb
        secondary = hsv2rgb(np.array([h, max(0.3, s * 0.6), v]))
        bg_light = hsv2rgb(np.array([h, 0.1, 0.95]))
        bg_dark = hsv2rgb(np.array([h, 0.2, 0.15]))
        surface = hsv2rgb(np.array([h, 0.15, 0.9]))

        def text_color(rgb):
            r, g, b = rgb
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            return np.array([0, 0, 0]) if lum > 0.6 else np.array([1, 1, 1])

        return {
            "primary": self._to_uint8(primary).tolist(),
            "secondary": self._to_uint8(secondary).tolist(),
            "background_light": self._to_uint8(bg_light).tolist(),
            "background_dark": self._to_uint8(bg_dark).tolist(),
            "surface": self._to_uint8(surface).tolist(),
            "text_on_primary": self._to_uint8(text_color(primary)).tolist(),
        }

    def extract(self, image_path):
        # =========================
        # LOAD + PREPROCESS
        # =========================
        img = Image.open(image_path).convert("RGB")
        img_small = img.resize(self.resize)
        img_np = np.array(img_small)

        pixels = img_np.reshape(-1, 3)
        img_norm = img_np / 255.0

        # =========================
        # COLOR SPACE
        # =========================
        img_lab = rgb2lab(img_norm)
        img_hsv = rgb2hsv(img_norm)

        pixels_lab = img_lab.reshape(-1, 3)
        pixels_hsv = img_hsv.reshape(-1, 3)

        # =========================
        # CLUSTERING
        # =========================
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(pixels_lab)

        labels = kmeans.labels_
        centers_lab = kmeans.cluster_centers_

        centers_rgb_unsorted = lab2rgb(
            centers_lab.reshape(1, self.k, 3)).reshape(self.k, 3)

        # =========================
        # SORT BY POPULATION
        # =========================
        counts = np.bincount(labels, minlength=self.k)
        idx = np.argsort(counts)[::-1]

        centers_rgb = centers_rgb_unsorted[idx]
        centers_lab = centers_lab[idx]
        counts = counts[idx]

        # =========================
        # SATURATION + LIGHTNESS
        # =========================
        cluster_saturation = []

        for i in range(self.k):
            cluster_pixels = pixels_hsv[labels == i]
            if len(cluster_pixels) == 0:
                cluster_saturation.append(0)
            else:
                cluster_saturation.append(np.mean(cluster_pixels[:, 1]))

        cluster_saturation = np.array(cluster_saturation)[idx]
        lightness = centers_lab[:, 0]

        # =========================
        # SCORING
        # =========================
        scores = []
        total_count = np.sum(counts)

        for i in range(self.k):
            pop_norm = counts[i] / total_count
            sat = cluster_saturation[i]
            L = lightness[i]

            penalty = 0.5 if (self.penalty_enabled and (
                L < 20 or L > 85)) else 1.0

            score = (
                self.population_weight * pop_norm +
                self.saturation_weight * sat
            ) * penalty
            scores.append(score)

        scores = np.array(scores)
        best_idx = np.argmax(scores)

        accent = centers_rgb[best_idx]
        accent_rgb = self._to_uint8(accent).tolist()

        # =========================
        # THEME GENERATION
        # =========================
        theme = self._generate_theme(accent)

        clustered_preview = centers_rgb_unsorted[labels].reshape(
            self.resize[1], self.resize[0], 3)
        palette_rgb = [self._to_uint8(color).tolist() for color in centers_rgb]

        cluster_details = []
        for i in range(self.k):
            cluster_details.append({
                "rank": i + 1,
                "rgb": palette_rgb[i],
                "count": int(counts[i]),
                "population": float(counts[i] / total_count),
                "saturation": float(cluster_saturation[i]),
                "lightness": float(lightness[i]),
                "score": float(scores[i]),
            })

        return {
            "config": {
                "k": int(self.k),
                "resize": [int(self.resize[0]), int(self.resize[1])],
                "population_weight": float(self.population_weight),
                "saturation_weight": float(self.saturation_weight),
                "penalty_enabled": bool(self.penalty_enabled),
            },
            "images": {
                "original": np.array(img),
                "resized": img_np,
                "clustered": clustered_preview,
            },
            "distribution": {
                "lab_ab": pixels_lab[:, 1:3],
                "rgb_norm": img_norm.reshape(-1, 3),
            },
            "palette_rgb": palette_rgb,
            "accent_rgb": accent_rgb,
            "cluster_details": cluster_details,
            "theme": theme,
        }
