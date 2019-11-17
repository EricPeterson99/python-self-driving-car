import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # produce gradient lines
    canny = cv2.Canny(blur, 50, 150)
    return canny


image = cv2.imread('Image/test_image.jpg')
lane_image = np.copy(image)

canny = canny(lane_image)
# cropped_image = region_of_interest(canny)


cv2.imshow('result', canny)
cv2.waitKey(0)
