import cv2
import numpy as np

img = cv2.imread("image.jpg")

pts1 = np.float32([[50, 50], [300, 50],
                   [50, 300], [300, 300]])

pts2 = np.float32([[0, 0], [250, 0],
                   [0, 250], [250, 250]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

result = cv2.warpPerspective(img, matrix, (250, 250))

cv2.imshow("Original Image", img)
cv2.imshow("Transformed Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()