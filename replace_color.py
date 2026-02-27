import cv2
import numpy as np

# Load the image
img = cv2.imread('red.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for RED
# Note: Red wraps around the 0-180 hue scale, so we combine two masks
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = mask1 + mask2

# Replace red pixels with Magenta (Hue ~150 in OpenCV's 0-180 range)
# We only modify the Hue channel (index 0)
hsv[red_mask > 0, 0] = 150

# Convert back to BGR and save
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('output_magenta.jpg', result)