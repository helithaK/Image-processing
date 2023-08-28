
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('spider.png', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Define a custom transformation function (e.g., invert the image)
transformed_image = cv2.bitwise_not(image)

# Create histograms for the original and equalized images
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

# Display the original, equalized, and transformed images using Matplotlib
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Equalized image
plt.subplot(132)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Transformed image
plt.subplot(133)
plt.imshow(transformed_image, cmap='gray')
plt.title('Transformed Image')
plt.axis('off')

# Create histograms subplot
plt.figure(figsize=(15, 5))

# Histogram of the original image
plt.subplot(121)
plt.plot(hist_original)
plt.title('Histogram of Original Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# Histogram of the equalized image
plt.subplot(122)
plt.plot(hist_equalized)
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()
