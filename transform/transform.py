import numpy as np
import cv2

def order_points(pts):
    #ordering of points is ncessary because we don't know the position of edges in contour list 
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    #top-left of rectangle will have the smallest sum of edge points 
    rect[0] = pts[np.argmin(s)] 
    #top-right of rectangle will have the smallest difference of edge points
    rect[1] = pts[np.argmin(diff)]  
    #bottom-right of rectangle will have the largest sum of edge points
    rect[2] = pts[np.argmax(s)] 
    #bottom-left of rectangle will have the largest difference of edge points
    rect[3] = pts[np.argmax(diff)] 

    return rect


def four_point_transform(image, pts):

    # here 'image' corresponds to original image
    # and 'pts' are the contour edge points 
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # computing width of new image which will be the
    # distance between bottom-right and bottom-left
    # or top-right and top-left x-coord
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # computing height of new image which will be the
    # distance between top-right and bottom-right
    # or top-left and bottom-left y-coord
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now creating a np-array with our new points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')

    # passing original 4 edge points and transformed points
    # to get transormation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # applying perpective transformation of openCV
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
