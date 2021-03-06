import numpy as np
import math
import cv2
import streamlit as st

def minmax(point, min, max, h, w):
    if h > w:  # latitude is greater
        ratio = w / h
        return [(point[0] - min[0]) / (max[0] - min[0]), ratio * ((point[1] - min[1]) / (max[1] - min[1])) ]
    else:
        ratio = h / w
        return [ratio * ((point[0] - min[0]) / (max[0] - min[0])), (point[1] - min[1]) / (max[1] - min[1]) ]

def get_area_polygon(polygon):
    polygon = polygon.to_crs({'init': 'epsg:3857'})
    return (polygon.area) / 10**6

def get_bounding_box(coord):
    maxLatVal = -1000
    maxLongVal = -1000
    minLatVal = 1000
    minLongVal = 1000
    for i in range(len(coord)):
        if coord[i][0]>=maxLatVal:
            maxLatVal = coord[i][0]
            maxLat = i
        if coord[i][1]>=maxLongVal:
            maxLongVal = coord[i][1]
            maxLong = i
        if coord[i][0]<=minLatVal:
            minLatVal = coord[i][0]
            minLat = i
        if coord[i][1]<=minLongVal:
            minLongVal = coord[i][1]
            minLong = i
    bounds = [maxLatVal, maxLongVal, minLatVal, minLongVal]
    diffLat = maxLatVal - minLatVal
    diffLong = maxLongVal - minLongVal
    return ([minLatVal, minLongVal], [maxLatVal, maxLongVal], diffLat, diffLong)  # h, w

def generate_image(normalized_blocks, height, width):
    mult_factor = 10000
    if height > width:
        ratio = width / height
        init_mat_grid = np.zeros((mult_factor, int(ratio * mult_factor)))
        init_mat_radial = np.zeros((mult_factor, int(ratio * mult_factor)))
    else:
        ratio = height / width
        init_mat_grid = np.zeros((int(ratio * mult_factor), mult_factor))
        init_mat_radial = np.zeros((int(ratio * mult_factor), mult_factor))

    idx = 0
    for block in normalized_blocks:
        label = block[-1]
        if height > width:
            ratio = width / height
            min_val = [int(block[0][0] * mult_factor), int(block[0][1] * mult_factor * ratio)]
            max_val = [int(block[2][0] * mult_factor), int(block[2][1] * mult_factor * ratio)]
        else:
            ratio = height / width
            min_val = [int(block[0][0] * mult_factor * ratio), int(block[0][1] * mult_factor)]
            max_val = [int(block[2][0] * mult_factor * ratio), int(block[2][1] * mult_factor)]

        idx += 1
        st.write(idx)
        st.write(min_val, max_val)
        
        if label == 0:
            init_mat_grid[min_val[0]:max_val[0] + 1, min_val[1]:max_val[1] + 1] = 255
        elif label == 1:
            init_mat_radial[min_val[0]:max_val[0] + 1, min_val[1]:max_val[1] + 1] = 255

    # st.image(init_mat_grid, clamp=True)
    # st.image(init_mat_radial, clamp=True)

    if height > width:
        ratio = width / height
        init_mat_grid_resized = cv2.resize(init_mat_grid, (int(ratio * 300), 300), interpolation=cv2.INTER_AREA)
        init_mat_radial_resized = cv2.resize(init_mat_radial, (int(ratio * 300), 300), interpolation=cv2.INTER_AREA)
    else:
        ratio = height / width
        init_mat_grid_resized = cv2.resize(init_mat_grid, (300, int(ratio * 300)), interpolation=cv2.INTER_AREA)
        init_mat_radial_resized = cv2.resize(init_mat_radial, (300, int(ratio * 300)), interpolation=cv2.INTER_AREA)

    return init_mat_grid_resized, init_mat_radial_resized
