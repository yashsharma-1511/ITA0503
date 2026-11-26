import cv2

img = cv2.imread("image.jpg")

# Select ROI
roi = img[50:200, 50:200]

# Paste ROI at another location
img[250:400, 250:400] = roi

cv2.imshow("Modified Image", img)

cv2.imwrite("result.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()