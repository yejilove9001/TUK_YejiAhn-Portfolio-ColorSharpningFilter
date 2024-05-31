import cv2
import numpy as np

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image

def sharpen_intensity(image):
    hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, i = cv2.split(hsi_image)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened_i = cv2.filter2D(i, -1, kernel)
    sharpened_i = np.clip(sharpened_i, 0, 255).astype(np.uint8)
    sharpened_hsi_image = cv2.merge([h, s, sharpened_i])
    sharpened_image = cv2.cvtColor(sharpened_hsi_image, cv2.COLOR_HSV2BGR)
    return sharpened_image

# Read an image from file
original_image = cv2.imread('images/yeji.jpg')

# Apply sharpening to RGB channels
sharpened_image = sharpen_image(original_image)

# Apply sharpening to the intensity channel in HSI
sharpened_image_hsi = sharpen_intensity(original_image)

# Compute the absolute difference between the two sharpened images
difference = cv2.absdiff(sharpened_image, sharpened_image_hsi)

# Display the original and sharpened images
cv2.imshow('Original Image', original_image)
cv2.imshow('Sharpened Image (RGB)', sharpened_image)
cv2.imshow('Sharpened Image (HSI)', sharpened_image_hsi)
cv2.imshow('Difference', difference)

cv2.waitKey(0)
cv2.destroyAllWindows()