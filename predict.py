import os
import numpy as np
import math
import random
import imutils
import utils
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from collections import Counter
import pickle
from shapely.ops import split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from shapely.geometry import Polygon, Point, LineString, MultiPolygon

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def predict(data):
    model = pickle.load(open('rf_63_acc.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))
    
    scaled = MinMaxScaler().fit_transform(data)
    test = pca.transform(scaled)
    test = pd.DataFrame(test)
    y_pred = model.predict(test)
    return y_pred

def _quadrat_cut_geometry(geometry, quadrat_width, min_num=3): 
    # create n evenly spaced points between the min and max x and y bounds
    west, south, east, north = geometry.bounds
    x_num = math.ceil((east - west) / quadrat_width) + 1
    y_num = math.ceil((north - south) / quadrat_width) + 1
    x_points = np.linspace(west, east, num=max(x_num, min_num))
    y_points = np.linspace(south, north, num=max(y_num, min_num))
    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points]
    horizont_lines = [LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points]
    lines = vertical_lines + horizont_lines
    # recursively split the geometry by each quadrat line
    for line in lines:
        geometry = MultiPolygon(split(geometry, line))
    return geometry

def gron_image(data, gdf, min_coord, max_coord):
    p1 = min_coord
    p2 = [min_coord[0], max_coord[1]]
    p3 = max_coord
    p4 = [max_coord[0], min_coord[1]]

    width = max_coord[1] - min_coord[1]
    height = max_coord[0] - min_coord[0]

    pointList = [p1, p2, p3, p4]

    bbPoly = Polygon(pointList)
    cityPoly = gdf['geometry'][0]

    bbBlocks = _quadrat_cut_geometry(bbPoly, 0.2, 7)
    cityBlocks = _quadrat_cut_geometry(cityPoly, 0.2, 7)
    st.write('No of blocks:', len(bbBlocks))

    layout = predict(data)
    layout = [random.randint(0, 1) for i in range(len(layout))]
    st.write('No of layout:', len(layout))
    st.write(layout)

    normalized_geoms = []
    for idx, block in enumerate(bbBlocks.geoms):
        temp, coords = [], []
        for i, j in zip(block.boundary.xy[0],block.boundary.xy[1]):
            coords.append([i, j])
            temp.append(utils.minmax(point=[i, j], min=min_coord, max=max_coord, h=height, w=width))
        temp.append(Polygon(coords).centroid)
        normalized_geoms.append(temp)
        if layout[idx] == 0 or layout[idx] == 1:
            normalized_geoms.append(0)
        else:
            normalized_geoms.append(1)

    # cnt, total = 0, len(layout)
    # for x in range(len(bbBlocks.geoms)):
    #     temp = []
    #     coords = []
    #     x_c, y_c = [], []
    #     for i,j in zip(bbBlocks[x].boundary.xy[0],bbBlocks[x].boundary.xy[1]):
    #         coords.append([i,j])
    #         x_c.append(i)
    #         y_c.append(j)

    #     max_x, max_y = max(x_c), max(y_c)
    #     st.write(x, max_x, max_y, max_coord)
    #     geom = Polygon(coords)
    #     cent = geom.centroid

    #     if cityPoly.contains(cent):
    #         # grid -> black & radial -> white
    #         if layout[cnt] == 0 or layout[cnt] == 1:
    #             temp.append(0)
    #         else:
    #             temp.append(1)
    #         cnt += 1
    #     else:
    #         temp.append(0)

    #     if max_x >= max_coord[0] or max_y >= max_coord[1]:
    #         st.write('Condn applied', x)
    #         img_array.append(temp)
    #         temp = []

    # st.write(img_array)

    img_grid, img_radial = utils.generate_image(normalized_blocks=normalized_geoms, height=height, width=width)

    cv2.imwrite('img_grid.png', np.array(img_grid))
    cv2.imwrite('img_rad.png', np.array(img_radial))

    # for i in img_array:
    #     st.write(i)
    # cv2.imwrite('img99.png', np.array(img_array))

    # city_hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    # lower_blue, upper_blue = np.array([94, 80, 2]), np.array([130, 255, 255])
    # lower_red, upper_red = np.array([155, 25, 0]), np.array([179, 255, 255])
    # # lower_red, upper_red = np.array([255, 0, 120]), np.array([255, 0, 120])
    # lower_yellow, upper_yellow = np.array([22, 93, 0]), np.array([45, 255, 255])

    # mask0 = cv2.inRange(city_hsv, lower_red, upper_red)
    # mask1 = cv2.inRange(city_hsv, lower_blue, upper_blue)
    # mask2 = cv2.inRange(city_hsv, lower_yellow, upper_yellow)
    # output = cv2.bitwise_and(img_array, img_array, mask=mask0)
    # output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(output, 30, 200)
    # contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # for j, contour in enumerate(contours):
    #     if cv2.contourArea(contour) > -1:
    #         bounding_box = cv2.boundingRect(contour)
    #         # print(bounding_box)
    #         top_left, bottom_right = (bounding_box[0], bounding_box[1]),\
    #                                 (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3])
    #         cv2.rectangle(img_array, top_left, bottom_right, (255, 255, 255), 2)
    #         center = (bounding_box[0] + int(bounding_box[2] / 2), bounding_box[1] + int(bounding_box[3] / 2))
    #         cv2.circle(img_array, center, 3, (255, 0, 0), -1)

    # return True

