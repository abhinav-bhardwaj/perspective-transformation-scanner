import numpy as np
import cv2
import argparse # For Command Line Arguments
import imutils 
from transform.transform import four_point_transform # perspective transorm
#from skimage.filters import threshold_local 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0 # Ratio of the original image
orig = image.copy() # Copy of Original Image
image = imutils.resize(image, height = 500) # Resizing image

imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting image from RGB to Grayscale
imgray = cv2.GaussianBlur(imgray, (5, 5), 0) # Performing Gaussian Blur to reduce noises
edge = cv2.Canny(imgray, 75, 200) # Canny Edge Detection

print("STEP 1: Edge Detection")
cv2.imshow("Original Image with Resizing", image)
cv2.imshow("Canny Edged Image", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding and storing contours of the Edged Image
cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts) # grabbing contours tuples based on OpenCV version
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5] #Sorting for the largest contours area

# Document usually has 4 edges, so checking if our contour grabbing is right
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
		break

print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Applying 4-Point Perspective Transformation on the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# converting warped image to grayscale
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#T = threshold_local(warped, 11, offset = 10, method = "gaussian")
#warped = (warped > T).astype("uint8") * 255

print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
