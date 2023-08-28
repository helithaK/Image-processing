import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image in the LAB color space
image = cv2.imread('highlights_and_shadows.jpg')
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Extract the L* channel
l_channel = lab_image[:,:,0]

# Define the gamma value
gamma = 0.2  # You can change this value as needed

# Apply gamma correction to the L* channel
l_channel_corrected = np.power(l_channel / 255.0, gamma) * 255.0

# Clip the values to ensure they are within the valid range [0, 255]
l_channel_corrected = np.clip(l_channel_corrected, 0, 255).astype(np.uint8)

# Replace the original L* channel with the corrected one
lab_image[:,:,0] = l_channel_corrected

# Convert LAB image back to BGR
output_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)

# Calculate histograms for the original and corrected L* channels
hist_original = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
hist_corrected = cv2.calcHist([l_channel_corrected], [0], None, [256], [0, 256])

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image Histogram')
plt.plot(hist_original)
plt.xlim([0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

plt.subplot(1, 2, 2)
plt.title('Corrected Image Histogram')
plt.plot(hist_corrected)
plt.xlim([0, 256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

plt.tight_layout()

# Display the gamma value
print(f'Gamma value used: {gamma}')

# Show the original and corrected images
cv2.imshow('Original Image', image)
cv2.imshow('Corrected Image', output_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()