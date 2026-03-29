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
K = 6

# =========================
# LOAD IMAGE
# =========================
img = Image.open(IMAGE_PATH).convert("RGB")

# =========================
# RESIZE
# =========================
img_small = img.resize(RESIZE_DIM)
img_small_np = np.array(img_small)

# =========================
# OPTIONAL BLUR
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
# K-MEANS CLUSTERING
# =========================
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(pixels_lab)

labels = kmeans.labels_
centers_lab = kmeans.cluster_centers_

# =========================
# LAB → RGB
# =========================
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
# CLUSTER SATURATION (HSV)
# =========================
cluster_saturation = []

for i in range(K):
    cluster_pixels = pixels_hsv[labels == i]

    if len(cluster_pixels) == 0:
        cluster_saturation.append(0)
    else:
        cluster_saturation.append(np.mean(cluster_pixels[:, 1]))

cluster_saturation = np.array(cluster_saturation)
cluster_saturation = cluster_saturation[sorted_idx]

# =========================
# LIGHTNESS (LAB)
# =========================
lightness = centers_lab[:, 0]

# =========================
# COLOR SCORING
# =========================
scores = []

for i in range(K):
    pop = counts[i]
    sat = cluster_saturation[i]
    L = lightness[i]

    pop_norm = pop / np.sum(counts)

    # Penalize unusable tones
    if L < 20 or L > 85:
        light_penalty = 0.5
    else:
        light_penalty = 1.0

    score = (0.5 * pop_norm + 0.8 * sat) * light_penalty
    scores.append(score)

scores = np.array(scores)

# =========================
# PICK ACCENT COLOR
# =========================
best_idx = np.argmax(scores)
accent_color = centers_rgb[best_idx]
accent_rgb = (accent_color * 255).astype(int)

# =========================
# PRINT DEBUG INFO
# =========================
print("\nDetailed Cluster Info:")

for i in range(K):
    rgb_255 = (centers_rgb[i] * 255).astype(int)
    print(f"""
Cluster {i+1}
RGB: {rgb_255}
Count: {counts[i]}
Saturation: {cluster_saturation[i]:.3f}
Lightness: {lightness[i]:.2f}
Score: {scores[i]:.3f}
""")

print("\nAccent Color:", accent_rgb)

# =========================
# SHOW PALETTE
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
# SHOW ACCENT COLOR
# =========================
accent_patch = np.ones((100, 100, 3)) * accent_color

plt.figure(figsize=(3, 3))
plt.imshow(accent_patch)
plt.title(f"Accent {accent_rgb}")
plt.axis("off")
plt.show()

# =========================
# RGB DISTRIBUTION
# =========================
r = pixels[:, 0]
g = pixels[:, 1]

plt.figure(figsize=(6, 6))
plt.scatter(r, g, c=pixels / 255.0, s=3, alpha=0.6)
plt.xlabel("Red")
plt.ylabel("Green")
plt.title("RGB Distribution")
plt.tight_layout()
plt.show()

# =========================
# LAB DISTRIBUTION
# =========================
a_lab = pixels_lab[:, 1]
b_lab = pixels_lab[:, 2]

plt.figure(figsize=(6, 6))
plt.scatter(a_lab, b_lab, c=img_small_norm.reshape(-1, 3), s=3, alpha=0.6)
plt.xlabel("A")
plt.ylabel("B")
plt.title("LAB Distribution")
plt.tight_layout()
plt.show()
