import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the input image
image = cv2.imread("flower.png", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the segmentation mask
segmentation_mask = np.zeros(image.shape[:2], np.uint8)
segmentation_mask[150:550, 50:550] = cv2.GC_PR_FGD
segmentation_mask[250:410, 220:380] = cv2.GC_FGD

# Define models for background and foreground
bg_model = np.zeros((1, 65), np.float64)
fg_model = np.zeros((1, 65), np.float64)

# Define the region of interest
roi = (50, 100, 500, 500)

# Number of iterations for GrabCut
num_iterations = 5

# Apply GrabCut algorithm
cv2.grabCut(image, segmentation_mask, roi, bg_model, fg_model, num_iterations, cv2.GC_INIT_WITH_MASK)

# Create masks for foreground and background
foreground_mask = np.where((segmentation_mask == 2) | (segmentation_mask == 0), 0, 1).astype('uint8')
background_mask = 1 - foreground_mask

# Create foreground and background images
foreground_image = image * foreground_mask[:, :, np.newaxis]
background_image = image * background_mask[:, :, np.newaxis]

# Apply Gaussian blur to the background
kernel_size = 51
sigma = 5
blurred_background = cv2.GaussianBlur(background_image, (kernel_size, kernel_size), sigma)

# Combine blurred background and foreground for final enhancement
enhanced_image = blurred_background + foreground_image

# Display the results
plt.figure(figsize=(18, 6))
plt.rc('axes', titlesize=15)

plt.subplot(151)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(152)
plt.imshow(foreground_mask, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.subplot(153)
plt.imshow(foreground_image)
plt.title('Foreground Image')
plt.axis('off')

plt.subplot(154)
plt.imshow(background_image)
plt.title('Background Image')
plt.axis('off')

plt.subplot(155)
plt.imshow(enhanced_image)
plt.title('Enhanced Image')
plt.axis('off')

plt.tight_layout()
plt.show()
