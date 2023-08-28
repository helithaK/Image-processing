import cv2
import numpy as np

# Load the image
image = cv2.imread('spider.png')

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the image into its Hue, Saturation, and Value planes
hue, saturation, value = cv2.split(hsv_image)

# Define the parameters 'a' and 'b' for the transformation function
a = 0.7  # Increase 'a' for a stronger effect
b = 70  # Increase 'b' for a stronger effect

# Scale 'a' to be in the range [0, 1]
a = max(0, min(1, a))

# Apply the intensity transformation function to saturation
transformed_saturation = np.clip(
    saturation + (a * 255 / 128) * np.exp(-((saturation - 128) ** 2) / (2 * b ** 2)), 0, 255
).astype(np.uint8)

print(saturation)
print(transformed_saturation)

# Merge the modified planes back into an HSV image
modified_hsv_image = cv2.merge([hue, transformed_saturation, value])

# Convert the modified HSV image back to BGR color space
modified_image = cv2.cvtColor(modified_hsv_image, cv2.COLOR_HSV2BGR)

# Display the original and modified images
cv2.imshow('Original Image', image)
cv2.imshow('Modified Image', modified_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
