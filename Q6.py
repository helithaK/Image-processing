import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the input image
input_image = cv2.imread("jeniffer.jpg", cv2.IMREAD_COLOR)

# Convert the input image to HSV
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)# Separate the HSV planes
hue_plane = hsv_image[:, :, 0]
saturation_plane = hsv_image[:, :, 1]
value_plane = hsv_image[:, :, 2]

# Plot the Hue, Saturation, and Value planes
plt.figure(figsize=(15, 10))
plt.rc("axes", titlesize=18)
plt.subplot(131)
plt.imshow(hue_plane, cmap='gray')
plt.title("Hue Plane Image ")
plt.axis('off')
plt.subplot(133)
plt.imshow(saturation_plane, cmap='gray')
plt.title("Saturation Plane Image")
plt.axis('off')
plt.subplot(132)
plt.imshow(value_plane, cmap='gray')
plt.title("Value Plane Image")
plt.axis('off')
plt.tight_layout()

# Threshold the Saturation plane to obtain a mask for the foreground
threshold = 13
foreground_mask = (saturation_plane > threshold).astype(np.uint8) * 255
foreground_mask_3d = np.repeat(foreground_mask[:, :, None], 3, axis=2)

# Extract the foreground using the mask
foreground_hsv = np.bitwise_and(hsv_image, foreground_mask_3d)
foreground_rgb = cv2.cvtColor(foreground_hsv, cv2.COLOR_HSV2RGB)

# Plot the original image, mask, and the extracted foreground
plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.subplot(133)
plt.imshow(foreground_mask_3d)
plt.title('Foreground Mask')
plt.axis('off')
plt.subplot(132)
plt.imshow(foreground_rgb)
plt.title('Extracted Foreground')
plt.axis('off')
plt.tight_layout()



# Create histograms for the extracted foreground and equalize them
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plt.rc("axes", titlesize=15)

equalized_foreground = foreground_rgb.copy()
channels = ('r', 'g', 'b')
total_pixels = foreground_mask.sum() // 255

# Loop over color channels, calculate, and plot histograms
for i, channel in enumerate(channels):
    hist = cv2.calcHist([foreground_rgb], [i], foreground_mask, [256], [0, 256])
    ax[0].plot(hist, color=channel)
    ax[0].set_xlim([0, 256])

    cumulative_hist = np.cumsum(hist)
    ax[1].plot(cumulative_hist, color=channel)
    ax[1].set_xlim([0, 256])

    transformation = cumulative_hist * 255 / cumulative_hist[-1]
    equalized_foreground[:, :, i] = transformation[foreground_rgb[:, :, i]]

# Apply the mask again after equalization
equalized_foreground = np.bitwise_and(equalized_foreground, foreground_mask_3d)

ax[0].set_title("Histogram of Foreground")
ax[1].set_title("Cumulative Histogram of Foreground")

# Extract the background and combine it with the equalized foreground
background_mask_3d = 255 - foreground_mask_3d
background_hsv = np.bitwise_and(hsv_image, background_mask_3d)
background_rgb = cv2.cvtColor(background_hsv, cv2.COLOR_HSV2RGB)
final_result = background_rgb + equalized_foreground

# Plot the equalized foreground and the final result with the original background
plt.figure(figsize=(10, 10))
plt.subplot(122)
plt.imshow(equalized_foreground)
plt.title('Equalized Foreground')
plt.axis('off')
plt.subplot(121)
plt.imshow(final_result)
plt.title('Final Result with Original Background')
plt.axis('off')
plt.tight_layout()

plt.show()
