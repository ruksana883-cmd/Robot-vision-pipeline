import cv2
import numpy as np

# ------------------------------
# Robot Vision Simulation Pipeline
# ------------------------------

# Load the image (simulate a robot camera frame)
img = cv2.imread("test_image.jpg")

# Resize image for faster processing
img = cv2.resize(img, (640, 480))

# Convert to grayscale (robot vision works better in single channel)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection (detect obstacles and boundaries)
edges = cv2.Canny(blur, 50, 150)

# Find contours from edges (detect object shapes)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # green contours

# Compute the centroid of each object (useful for robot grasping or navigation)
for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Mark centroid on image
        cv2.circle(contour_img, (cx, cy), 5, (0, 0, 255), -1)  # red dot

# Overlay edges on the original image (robot sees obstacles)
overlay = img.copy()
overlay[edges != 0] = [255, 0, 0]  # red edges

# Stack results horizontally for visualization
result = np.hstack((img, contour_img, overlay))

# Show the result
cv2.imshow("Robot Vision Pipeline - Original | Contours | Edge Overlay", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
