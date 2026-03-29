import numpy as np
from PIL import Image
from skimage.color import rgb2lab, rgb2hsv, lab2rgb, hsv2rgb
from sklearn.cluster import KMeans


class ThemeExtractor:
    def __init__(self, k=6, resize=(64, 64)):
        self.k = k
        self.resize = resize

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

        centers_rgb = lab2rgb(centers_lab.reshape(
            1, self.k, 3)).reshape(self.k, 3)

        # =========================
        # SORT BY POPULATION
        # =========================
        counts = np.bincount(labels)
        idx = np.argsort(counts)[::-1]

        centers_rgb = centers_rgb[idx]
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

        for i in range(self.k):
            pop_norm = counts[i] / np.sum(counts)
            sat = cluster_saturation[i]
            L = lightness[i]

            penalty = 0.5 if (L < 20 or L > 85) else 1.0

            score = (0.5 * pop_norm + 0.8 * sat) * penalty
            scores.append(score)

        scores = np.array(scores)
        best_idx = np.argmax(scores)

        accent = centers_rgb[best_idx]

        # =========================
        # THEME GENERATION
        # =========================
        accent_hsv = rgb2hsv(accent.reshape(1, 1, 3)).reshape(3)
        h, s, v = accent_hsv

        primary = accent
        secondary = hsv2rgb(np.array([h, max(0.3, s * 0.6), v]))
        bg_light = hsv2rgb(np.array([h, 0.1, 0.95]))
        bg_dark = hsv2rgb(np.array([h, 0.2, 0.15]))
        surface = hsv2rgb(np.array([h, 0.15, 0.9]))

        def text_color(rgb):
            r, g, b = rgb
            lum = 0.299*r + 0.587*g + 0.114*b
            return np.array([0, 0, 0]) if lum > 0.6 else np.array([1, 1, 1])

        theme = {
            "primary": (primary * 255).astype(int).tolist(),
            "secondary": (secondary * 255).astype(int).tolist(),
            "background_light": (bg_light * 255).astype(int).tolist(),
            "background_dark": (bg_dark * 255).astype(int).tolist(),
            "surface": (surface * 255).astype(int).tolist(),
            "text_on_primary": (text_color(primary) * 255).astype(int).tolist()
        }

        return theme
