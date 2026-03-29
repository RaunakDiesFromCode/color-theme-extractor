from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("data/test.png")
img = np.array(img)

print("Shape:", img.shape)

plt.imshow(img)
plt.title("Loaded Image")
plt.axis("off")
plt.show()
