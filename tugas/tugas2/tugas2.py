import cv2
import numpy as np

# image
img = cv2.imread("/Users/dans/Library/CloudStorage/OneDrive-InstitutTeknologiSepuluhNopember/InternBayucaraka/task-opencv/MagangBayu2024-OpenCV/tugas/tugas2/tugas2.jpg")

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create a mask to isolate the blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Remove noise from the mask
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Sort the contours by size in descending order
contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[::-3]

for cnt in contours:
    # Approximate the contour shape to polygonal form with the specified precision in percent
    epsilon = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * epsilon, True)

x, y, w, h = cv2.boundingRect(approx)

# Draw the contours on the original image with pink color
result = cv2.drawContours(img, contours, -1, (255, 0, 255), 3)

# Count the number of contours and add it to the image
font = cv2.FONT_HERSHEY_SIMPLEX
result = cv2.putText(result, f"Number of lines : {len(approx)}", (x + w + 20, y + 20), font, 1, (255, 0, 255), 2, cv2.LINE_AA)

# Display the result
cv2.imshow("Task-2", result)
cv2.waitKey(0)
cv2.destroyAllWindows()