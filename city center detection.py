from cv2 import cv2
import numpy as np
import imutils
import math

city = cv2.imread('test.png')
city_hsv = cv2.cvtColor(city, cv2.COLOR_BGR2HSV)

lower_blue, upper_blue = np.array([94, 80, 2]), np.array([130, 255, 255])
lower_red, upper_red = np.array([155, 25, 0]), np.array([179, 255, 255])
# lower_red, upper_red = np.array([255, 0, 120]), np.array([255, 0, 120])
lower_yellow, upper_yellow = np.array([22, 93, 0]), np.array([45, 255, 255])

mask0 = cv2.inRange(city_hsv, lower_red, upper_red)
mask1 = cv2.inRange(city_hsv, lower_blue, upper_blue)
mask2 = cv2.inRange(city_hsv, lower_yellow, upper_yellow)
output = cv2.bitwise_and(city, city, mask=mask0)
output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(output, 30, 200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
for j, contour in enumerate(contours):
    if cv2.contourArea(contour) > -1:
        bounding_box = cv2.boundingRect(contour)
        # print(bounding_box)
        top_left, bottom_right = (bounding_box[0], bounding_box[1]),\
                                 (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
        cv2.rectangle(city, top_left, bottom_right, (255, 255, 255), 2)
        center = (bounding_box[0] + int(bounding_box[2] / 2), bounding_box[1] + int(bounding_box[3] / 2))
        cv2.circle(city, center, 3, (255, 0, 0), -1)

cv2.imshow('f1', city)
cv2.imshow('f2', mask0)
cv2.waitKey()
