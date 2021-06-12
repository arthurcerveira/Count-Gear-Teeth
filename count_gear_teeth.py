import sys
from math import floor

import cv2
import matplotlib.pyplot as plt

# Get image from command line argument
image_path = sys.argv[1]
gear = cv2.imread(image_path)
precision_threshold = 0.005

# Filter edges and find contours
gear_bilateral_filter = cv2.bilateralFilter(gear, 5, 175, 175)
gear_median_blur = cv2.medianBlur(gear_bilateral_filter, 5)
gear_edges = cv2.Canny(gear_median_blur, 75, 200)
contours, _ = cv2.findContours(gear_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the contour with the biggest area
max_contour = max(contours, key=cv2.contourArea)

# Approximate the contour
arclen = cv2.arcLength(max_contour, True)
approx = cv2.approxPolyDP(max_contour, 
                          precision_threshold * arclen, 
                          True)

# Overlay the image with the results
for point in approx:
    cv2.circle(gear, point[0], 3, (0,255,0), -1, cv2.LINE_AA)

# The number of teeth is half the number of angles found on the gear
result = floor(len(approx) / 2)

# Display the results
print(f"Number of teeth: {result}")

plt.imshow(gear)
plt.show()