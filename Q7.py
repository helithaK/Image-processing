import cv2
import matplotlib.pyplot as plt
import numpy as np

def custom_filter(image, kernel):
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    k_hh, k_hw = kernel.shape[0] // 2, kernel.shape[1] // 2
    h, w = image.shape
    image_float = cv2.normalize(image.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    result = np.zeros(image.shape, 'float')

    for m in range(k_hh, h - k_hh):
        for n in range(k_hw, w - k_hw):
            result[m, n] = np.dot(image_float[m-k_hh: m+k_hh+1, n-k_hw: n+k_hw+1].flatten(), kernel.flatten())

    result = result * 255
    result = np.minimum(255, np.maximum(0, result)).astype(np.uint8)
    return result

def custom_filter_step(image, kernel):
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    k_hh, k_hw = kernel.shape[0] // 2, kernel.shape[1] // 2
    h, w = image.shape
    result = np.zeros(image.shape, 'float')
    for m in range(k_hh, h - k_hh):
        for n in range(k_hw, w - k_hw):
            result[m, n] = np.dot(image[m - k_hh: m + k_hh + 1, n - k_hw: n + k_hw + 1].flatten(), kernel.flatten())
    return result

def custom_filter_in_steps(image, kernel1, kernel2):
    image_float = cv2.normalize(image.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    result = custom_filter_step(custom_filter_step(image_float, kernel1), kernel2)
    result = result * 255
    result = np.minimum(255, np.maximum(0, result)).astype(np.uint8)
    return result

# Load the input image
input_img = cv2.imread("einstein.png", cv2.IMREAD_GRAYSCALE)

# Define a Sobel-like vertical kernel
sobel_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# Apply the Sobel filter using different methods
result_a = cv2.filter2D(input_img, -1, sobel_kernel)  # Using filter2D
result_b = custom_filter(input_img, sobel_kernel)    # Using custom function
kernel1 = np.array([1, 2, 1]).reshape((3, 1))
kernel2 = np.array([1, 0, -1]).reshape((1, 3))
result_c = custom_filter_in_steps(input_img, kernel1, kernel2)  # Using property of convolution

# Plot the results
plt.figure(figsize=(18, 6))
plt.rc('axes', titlesize=15)

plt.subplot(141)
plt.imshow(input_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(142)
plt.imshow(result_a, cmap='gray')
plt.title('Sobel Filter(filter2D)')
plt.axis('off')

plt.subplot(143)
plt.imshow(result_b, cmap='gray')
plt.title('Sobel Filter (Custom Funct)')
plt.axis('off')

plt.subplot(144)
plt.imshow(result_c, cmap='gray')
plt.title('Sobel Filter (Convolution Prop)')
plt.axis('off')

plt.tight_layout()
plt.show()
