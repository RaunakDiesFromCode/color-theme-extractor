from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2hsv

# =========================
# CONFIG
# =========================
IMAGE_PATH = "data/test.png"
RESIZE_DIM = (64, 64)
BLUR_RADIUS = 10  # set 0 if you want to disable blur

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
# OPTIONAL BLUR (FOR COMPARISON ONLY)
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
# FLATTEN PIXELS (FOR ML)
# =========================
pixels = img_small_np.reshape(-1, 3)

print("Pixel array shape:", pixels.shape)

# =========================
# COLOR SPACE CONVERSION
# =========================

# Normalize to [0,1] for skimage
img_small_norm = img_small_np / 255.0

# Convert
img_lab = rgb2lab(img_small_norm)
img_hsv = rgb2hsv(img_small_norm)

# Flatten
pixels_lab = img_lab.reshape(-1, 3)
pixels_hsv = img_hsv.reshape(-1, 3)

print("LAB shape:", pixels_lab.shape)
print("HSV shape:", pixels_hsv.shape)

# =========================
# RGB COLOR DISTRIBUTION
# =========================
r = pixels[:, 0]
g = pixels[:, 1]
b = pixels[:, 2]

plt.figure(figsize=(6, 6))
plt.scatter(r, g, c=pixels / 255.0, s=5)
plt.xlabel("Red")
plt.ylabel("Green")
plt.title("Color Distribution (RGB space)")
plt.tight_layout()
plt.show()

# =========================
# LAB COLOR DISTRIBUTION
# =========================

a = pixels_lab[:, 1]
b = pixels_lab[:, 2]

plt.figure(figsize=(6, 6))
plt.scatter(a, b, c=img_small_norm.reshape(-1, 3), s=5)
plt.xlabel("A (Green-Red)")
plt.ylabel("B (Blue-Yellow)")
plt.title("LAB Color Distribution (A-B space)")
plt.tight_layout()
plt.show()
