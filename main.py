from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2hsv, lab2rgb
from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
IMAGE_PATH = "data/test.png"
RESIZE_DIM = (64, 64)
BLUR_RADIUS = 10
K = 6  # number of clusters

# =========================
# LOAD IMAGE
# =========================
img = Image.open(IMAGE_PATH).convert("RGB")

# =========================
# RESIZE (PRIMARY METHOD)
# =========================
img_small = img.resize(RESIZE_DIM)
img_small_np = np.array(img_small)

# =========================
# OPTIONAL BLUR (COMPARISON)
# =========================
img_blur = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
img_blur_np = np.array(img_blur)

# =========================
# ORIGINAL NUMPY
# =========================
img_np = np.array(img)

# =========================
# SHOW IMAGE COMPARISON
# =========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_small_np)
plt.title(f"Resized {RESIZE_DIM}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_blur_np)
plt.title(f"Blurred (r={BLUR_RADIUS})")
plt.axis("off")

plt.tight_layout()
plt.show()

# =========================
# FLATTEN PIXELS
# =========================
pixels = img_small_np.reshape(-1, 3)
print("Pixel array shape:", pixels.shape)

# =========================
# COLOR SPACE CONVERSION
# =========================
img_small_norm = img_small_np / 255.0

img_lab = rgb2lab(img_small_norm)
img_hsv = rgb2hsv(img_small_norm)

pixels_lab = img_lab.reshape(-1, 3)
pixels_hsv = img_hsv.reshape(-1, 3)

print("LAB shape:", pixels_lab.shape)
print("HSV shape:", pixels_hsv.shape)

# =========================
# K-MEANS CLUSTERING (LAB)
# =========================
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(pixels_lab)

labels = kmeans.labels_
centers_lab = kmeans.cluster_centers_

print("\nCluster centers (LAB):")
print(centers_lab)

# =========================
# CONVERT LAB → RGB
# =========================
centers_lab_reshaped = centers_lab.reshape(1, K, 3)
centers_rgb = lab2rgb(centers_lab_reshaped).reshape(K, 3)

# =========================
# SORT BY DOMINANCE
# =========================
counts = np.bincount(labels)
sorted_idx = np.argsort(counts)[::-1]

centers_rgb = centers_rgb[sorted_idx]
centers_lab = centers_lab[sorted_idx]
counts = counts[sorted_idx]

# =========================
# PRINT DOMINANT COLORS
# =========================
print("\nDominant Colors (RGB 0-255):")
for i, color in enumerate(centers_rgb):
    rgb_255 = (color * 255).astype(int)
    print(f"{i+1}: {rgb_255} | count = {counts[i]}")

# =========================
# SHOW COLOR PALETTE
# =========================
palette = np.zeros((50, 300, 3))
step = 300 // K

for i in range(K):
    palette[:, i*step:(i+1)*step, :] = centers_rgb[i]

plt.figure(figsize=(6, 2))
plt.imshow(palette)
plt.title("Extracted Color Palette")
plt.axis("off")
plt.show()

# =========================
# RGB DISTRIBUTION
# =========================
r = pixels[:, 0]
g = pixels[:, 1]
b_rgb = pixels[:, 2]

plt.figure(figsize=(6, 6))
plt.scatter(r, g, c=pixels / 255.0, s=3, alpha=0.6)
plt.xlabel("Red")
plt.ylabel("Green")
plt.title("Color Distribution (RGB space)")
plt.tight_layout()
plt.show()

# =========================
# LAB DISTRIBUTION
# =========================
a_lab = pixels_lab[:, 1]
b_lab = pixels_lab[:, 2]

plt.figure(figsize=(6, 6))
plt.scatter(a_lab, b_lab, c=img_small_norm.reshape(-1, 3), s=3, alpha=0.6)
plt.xlabel("A (Green-Red)")
plt.ylabel("B (Blue-Yellow)")
plt.title("LAB Color Distribution (A-B space)")
plt.tight_layout()
plt.show()
