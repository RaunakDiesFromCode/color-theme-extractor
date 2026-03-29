from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2hsv, lab2rgb, hsv2rgb
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
IMAGE_PATH = "data/test.png"
RESIZE_DIM = (64, 64)
K = 6

# =========================
# LOAD + PREPROCESS
# =========================
img = Image.open(IMAGE_PATH).convert("RGB")
img_small = img.resize(RESIZE_DIM)
img_small_np = np.array(img_small)

pixels = img_small_np.reshape(-1, 3)
img_small_norm = img_small_np / 255.0

# =========================
# COLOR SPACE
# =========================
img_lab = rgb2lab(img_small_norm)
img_hsv = rgb2hsv(img_small_norm)

pixels_lab = img_lab.reshape(-1, 3)
pixels_hsv = img_hsv.reshape(-1, 3)

# =========================
# KMEANS
# =========================
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(pixels_lab)

labels = kmeans.labels_
centers_lab = kmeans.cluster_centers_

centers_rgb = lab2rgb(centers_lab.reshape(1, K, 3)).reshape(K, 3)

# =========================
# SORT BY DOMINANCE
# =========================
counts = np.bincount(labels)
sorted_idx = np.argsort(counts)[::-1]

centers_rgb = centers_rgb[sorted_idx]
centers_lab = centers_lab[sorted_idx]
counts = counts[sorted_idx]

# =========================
# SATURATION + LIGHTNESS
# =========================
cluster_saturation = []

for i in range(K):
    cluster_pixels = pixels_hsv[labels == i]
    if len(cluster_pixels) == 0:
        cluster_saturation.append(0)
    else:
        cluster_saturation.append(np.mean(cluster_pixels[:, 1]))

cluster_saturation = np.array(cluster_saturation)[sorted_idx]
lightness = centers_lab[:, 0]

# =========================
# SCORING
# =========================
scores = []

for i in range(K):
    pop_norm = counts[i] / np.sum(counts)
    sat = cluster_saturation[i]
    L = lightness[i]

    light_penalty = 0.5 if (L < 20 or L > 85) else 1.0

    score = (0.5 * pop_norm + 0.8 * sat) * light_penalty
    scores.append(score)

scores = np.array(scores)

best_idx = np.argmax(scores)
accent_rgb = centers_rgb[best_idx]
accent_rgb_255 = (accent_rgb * 255).astype(int)

print("\nAccent Color:", accent_rgb_255)

# =========================
# THEME GENERATION (HSV)
# =========================
accent_hsv = rgb2hsv(accent_rgb.reshape(1, 1, 3)).reshape(3)

h, s, v = accent_hsv

# Primary (main accent)
primary = accent_rgb

# Secondary (slightly less saturated)
secondary = hsv2rgb(np.array([h, max(0.3, s * 0.6), v]))

# Background light
bg_light = hsv2rgb(np.array([h, 0.1, 0.95]))

# Background dark
bg_dark = hsv2rgb(np.array([h, 0.2, 0.15]))

# Surface (cards)
surface = hsv2rgb(np.array([h, 0.15, 0.9]))

# =========================
# TEXT COLOR (contrast)
# =========================


def get_text_color(rgb):
    r, g, b = rgb
    luminance = 0.299*r + 0.587*g + 0.114*b
    return np.array([0, 0, 0]) if luminance > 0.6 else np.array([1, 1, 1])


text_on_primary = get_text_color(primary)
text_on_bg = get_text_color(bg_light)

# =========================
# VISUALIZE THEME
# =========================


def show_color(label, color):
    patch = np.ones((100, 100, 3)) * color
    plt.imshow(patch)
    plt.title(label)
    plt.axis("off")
    plt.show()


show_color("Primary", primary)
show_color("Secondary", secondary)
show_color("Background Light", bg_light)
show_color("Background Dark", bg_dark)
show_color("Surface", surface)
show_color("Text on Primary", text_on_primary)

# =========================
# THEME STRIP VIEW
# =========================
theme = np.zeros((100, 600, 3))

colors = [
    primary,
    secondary,
    bg_light,
    bg_dark,
    surface,
    text_on_primary
]

labels = ["Primary", "Secondary", "BG Light", "BG Dark", "Surface", "Text"]

step = 600 // len(colors)

for i, c in enumerate(colors):
    theme[:, i*step:(i+1)*step, :] = c

plt.figure(figsize=(10, 2))
plt.imshow(theme)
plt.title("Generated Theme")
plt.axis("off")
plt.show()
