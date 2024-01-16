import cv2
import numpy as np

#membaca gambar dalam file
img = cv2.imread("/Users/dans/Library/CloudStorage/OneDrive-InstitutTeknologiSepuluhNopember/InternBayucaraka/task-opencv/MagangBayu2024-OpenCV/tugas/tugas1/tugas1.png")

#define kernel size  
kernel = np.ones((7,7),np.uint8)

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower bound and upper bound for Green color 
lower_bound = np.array([50, 20, 20])     
upper_bound = np.array([100, 255, 255])

# Threshold the HSV image to get only green colors
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Apply morphological operations to remove noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Bounding box
if len(contours) > 0:
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# dengan gambar yang telah dibaca
cv2.imshow("Gambar", img)

cv2.waitKey(0) #menunggu key-press
cv2.destroyAllWindows() #menutup semua window