import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# For a colour image

def interpolate(image, indices, type):
    if type == 'nn':
        indices[0] = np.minimum(np.round(indices[0]), image.shape[0] - 1)
        indices[1] = np.minimum(np.round(indices[1]), image.shape[1] - 1)
        indices = indices.astype(np.uint64)
        return image[indices[0], indices[1]]

    elif type == 'bi':
        floors = np.floor(indices).astype(np.uint64)
        ceils = floors + 1

        ceils_limited = [np.minimum(ceils[0], image.shape[0] - 1), np.minimum(ceils[1], image.shape[1] - 1)]

        p1 = image[floors[0], floors[1]]
        p2 = image[floors[0], ceils_limited[1]]
        p3 = image[ceils_limited[0], floors[1]]
        p4 = image[ceils_limited[0], ceils_limited[1]]

        # Repeat indices for the 3 color planes
        indices = np.repeat(indices[:, :, :, None], 3, axis=3)
        ceils = np.repeat(ceils[:, :, :, None], 3, axis=3)
        floors = np.repeat(floors[:, :, :, None], 3, axis=3)

        # Find the horizontal midpoints
        m1 = p1 * (ceils[1] - indices[1]) + p2 * (indices[1] - floors[1])
        m2 = p3 * (ceils[1] - indices[1]) + p4 * (indices[1] - floors[1])
        # Find the vertical midpoint of horizontal midpoints
        m = m1 * (ceils[0] - indices[0]) + m2 * (indices[0] - floors[0])
        return m.astype(np.uint8)


def zoom(image, factor, interpolation='nn'):
    h, w, _ = image.shape
    zoom_h, zoom_w = round(h * factor), round(w * factor)
    zoomed_image = np.zeros((zoom_h, zoom_w, 3)).astype(np.uint8)

    zoomed_indices = np.indices((zoom_h, zoom_w)) / factor
    zoomed_image = interpolate(image, zoomed_indices, interpolation)

    return zoomed_image


def normalized_ssd(image1, image2):
    ssd = np.sum((image1 - image2) ** 2)
    return ssd / (image1.size * 255 * 255)



img8 = cv.imread( "im10small.png", cv.IMREAD_COLOR)
original_img8 = cv.imread( "im10.png", cv.IMREAD_COLOR)

zoomed_nn = zoom(img8, 4, 'nn')
#print(normalized_ssd(zoomed_nn, original_img8))

zoomed_bi = zoom(img8, 4, 'bi')
#print(normalized_ssd(zoomed_bi, original_img8))


plt.figure(figsize = (20, 10))
plt.rc("axes", titlesize = 18)
plt.subplot(131)
plt.imshow(cv.cvtColor(original_img8, cv.COLOR_BGR2RGB))
plt.title('Original Larger Image')
plt.axis('off')
plt.subplot(132)
plt.imshow(cv.cvtColor(zoomed_nn, cv.COLOR_BGR2RGB))
plt.title('Nearest neighbor interpolation')
plt.axis('off')
plt.subplot(133)
plt.imshow(cv.cvtColor(zoomed_bi, cv.COLOR_BGR2RGB))
plt.title('Bilinear interpolation')
plt.axis('off')
plt.tight_layout()

plt.show()


print("Normalized sum of squared differences")
for i in range(1, 12):
    num = str(i).zfill(2)
    small_image = cv.imread(f"images/zooming/im{num}small.png", cv.IMREAD_COLOR)
    large_image = cv.imread(f"images/zooming/im{num}.png", cv.IMREAD_COLOR)

    zoomed_nn = zoom(small_image, 4, "nn")
    zoomed_bi = zoom(small_image, 4, "bi")
    try:
        print(f"Image {num}: Nearest neighbors = {normalized_ssd(zoomed_nn, large_image)},\tBilinear = {normalized_ssd(zoomed_bi, large_image)}")
    except: pass