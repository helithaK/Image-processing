import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# White matter enhancement function
def enhance_white_matter(image):
    x_mid, y_mid = 170, 75
    white_transform = np.arange(0, 256).astype(np.uint8)
    white_transform[:x_mid + 1] = np.linspace(0, y_mid, x_mid + 1, dtype=np.uint8)
    white_transform[x_mid:] = np.linspace(y_mid, 255, 256 - x_mid, dtype=np.uint8)
    return white_transform[image]

# Gray matter enhancement function
def enhance_gray_matter(image):
    x1, x2 = 130, 188
    y1, y2 = 50, 200
    grey_transform = np.linspace(0, 50, 256)
    grey_transform = np.round(grey_transform).astype(np.uint8)
    grey_transform[x1:x2 + 1] = np.linspace(y1, y2, x2 + 1 - x1, dtype=np.uint8)
    return grey_transform[image]

# Load the image
image = cv.imread("BrainProtonDensitySlice9.png", cv.IMREAD_GRAYSCALE)

# Enhance white matter and gray matter
white_matter_enhanced = enhance_white_matter(image)
gray_matter_enhanced = enhance_gray_matter(image)

# Plot the enhancement functions
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(enhance_white_matter(np.arange(256)))
plt.title("White Matter Accentuating Transform")
plt.xlabel("Input intensity")
plt.ylabel("Output intensity")

plt.subplot(122)
plt.plot(enhance_gray_matter(np.arange(256)))
plt.title("Gray Matter Accentuating Transform")
plt.xlabel("Input intensity")
plt.ylabel("Output intensity")

plt.tight_layout()

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(131)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(132)
plt.imshow(cv.cvtColor(white_matter_enhanced, cv.COLOR_BGR2RGB))
plt.title("White Matter Accentuated")
plt.axis('off')

plt.subplot(133)
plt.imshow(cv.cvtColor(gray_matter_enhanced, cv.COLOR_BGR2RGB))
plt.title("Gray Matter Accentuated")
plt.axis('off')

plt.show()
