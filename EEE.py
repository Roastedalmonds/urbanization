import imp
import json
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely import wkt
import math
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import streamlit as st
import utils

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def clipPoly(coord, minX, minY, maxX, maxY):
    if coord[0] <= minX:
        coord[0] = minX
    elif coord[0] >= maxX:
        coord[0] = maxX
    
    if coord[1] <= minY:
        coord[1] = minY
    elif coord[1] >= maxY:
        coord[1] = maxY
    
    return coord

def get_poly_coords(df, min_coord, max_coord):
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')

    p1 = min_coord
    p2 = [min_coord[0], max_coord[1]]
    p3 = max_coord
    p4 = [max_coord[0], min_coord[1]]

    width = max_coord[1] - min_coord[1]
    height = max_coord[0] - min_coord[0]

    pointList = [p1, p2, p3, p4]
    bbPoly = Polygon(pointList) 
    jsonPoly=[]
    polyType=[]
    for x in range(len(gdf)):
        temp299 = []
        if type(gdf.loc[x,'geometry'])==Polygon:
            # oo = gdf.loc[x,'geometry'].intersection(bbPoly)
            # gdf.loc[x,'geometry'] = Polygon(list(oo.exterior.coords))
            try:
                polys=[]
                for i,j in list(zip(gdf.loc[x,'geometry'].boundary.xy[0],gdf.loc[x,'geometry'].boundary.xy[1])):
                    temp199 = clipPoly([i, j], min_coord[0], min_coord[1], max_coord[0], max_coord[1])
                    temp199 = utils.minmax(temp199, min_coord, max_coord, height, width)
                    if tuple(temp199) not in temp299:
                        temp299.append(tuple(temp199))
                        polys.append(dict({'x':temp199[0],'y':temp199[1]}))
                # polys.append(polys[0])
                jsonPoly.append([polys])
                polyType.append(3)
            except:
                jsonPoly.append('NA')
                polyType.append('NA')

        elif type(gdf.loc[x,'geometry'])==MultiPolygon:
            try:
                mulPoly=[]
                for t in range(len(gdf.loc[x,'geometry'].geoms)):
                    polys=[]
                    for i,j in zip(gdf.loc[x,'geometry'].geoms[t].boundary.xy[0],gdf.loc[x,'geometry'].geoms[t].boundary.xy[1]):
                        temp199 = clipPoly([i, j], min_coord[0], min_coord[1], max_coord[0], max_coord[1])
                        temp199 = utils.minmax(temp199, min_coord, max_coord, height, width)
                        if tuple(temp199) not in temp299:
                            temp299.append(tuple(temp199))
                            polys.append(dict({'x':temp199[0],'y':temp199[1]}))
                    temp299 = []
                    # polys.append(polys[0])
                    mulPoly.append(polys)
                jsonPoly.append(mulPoly)
                polyType.append(4)
            except:
                jsonPoly.append('NA')
                polyType.append('NA')

        elif type(gdf.loc[x,'geometry'])==Point:
            temp99 = utils.minmax([list(gdf.loc[x,'geometry'].xy[0])[0], list(gdf.loc[x,'geometry'].xy[1])[0]], min_coord, max_coord, height, width)
            jsonPoly.append([dict({'x':temp99[0],'y':temp99[1]})])
            polyType.append(1)

        elif type(gdf.loc[x,'geometry'])==LineString:
            try:
                for i,j in zip(gdf.loc[x,'geometry'].boundary.xy[0],gdf.loc[x,'geometry'].boundary.xy[1]):
                    temp199 = clipPoly([i, j], min_coord[0], min_coord[1], max_coord[0], max_coord[1])
                    temp199 = utils.minmax(temp199, min_coord, max_coord, height, width)
                    if tuple(temp199) not in temp299:
                        temp299.append(tuple(temp199))
                        polys.append(dict({'x':temp199[0],'y':temp199[1]}))
                jsonPoly.append(polys)
                polyType.append(2)
            except:
                jsonPoly.append('NA')
                polyType.append('NA')

        else:
            jsonPoly.append('NA')
            polyType.append('NA')
            print(f'Geometry not recognised at index: {x}')
    return jsonPoly,polyType

def get_historic_json_op(jsonPoly,polyType,df):
    jsonOP=[]
    for x in range(len(jsonPoly)):
        try:
            name=df.loc[x,'name']
        except:
            name='NA'

        try:
            if 'historic' in df.columns:
                if not math.isnan(df.loc[x,'historic']):
                    typ= str(df.loc[x,'historic'])+'_Historic'
                else:
                    typ='Historic'
            elif 'tourism' in df.columns:
                if not math.isnan(df.loc[x,'tourism']):
                    typ=str(df.loc[x,'tourism'])+'_Tourism'
                else:
                    typ='Tourism'

        except:
            typ='Historic_Tourism'

        if jsonPoly[x]=='NA':
            pass
        else:
            temp={
                'name':name,
                'type':typ,
                'polygon':polyType[x],
                'geometry':jsonPoly[x]
            }
            jsonOP.append(temp)
    return jsonOP

def get_building_json_op(jsonPoly,polyType,df):
    jsonOP=[]
    for x in range(len(jsonPoly)):
        try:
            name=df.loc[x,'name']
        except:
            name='NA'

        try:
            if 'building' in df.columns:
                if not math.isnan(df.loc[x,'building']) and df.loc[x,'building']=='industrial':
                    typ= 'Industrial'
                elif not math.isnan(df.loc[x,'building']) and df.loc[x,'building']=='commercial':
                    typ='Commercial'
        except:
            typ='Building'

        if jsonPoly[x]=='NA':
            pass
        else:
            temp={
                'name':name,
                'type':typ,
                'polygon':polyType[x],
                'geometry':jsonPoly[x]
            }
            jsonOP.append(temp)
    return jsonOP

def get_forest_json_op(jsonPoly,polyType,df):
    jsonOP=[]
    for x in range(len(jsonPoly)):
        try:
            name=df.loc[x,'name']
        except:
            name='NA'

        try:    
            if 'water' in df.columns: #or (not (math.isnan(df.loc[x,'water']))):
                if df.loc[x,'water']=='pond' or df.loc[x,'water']=='reservoir' or df.loc[x,'water']=='lake' or df.loc[x,'water']=='river' or df.loc[x,'water']=='canal': 
                    typ='Water_Body'
            else:
                typ='Forest'
        except:
            typ='Forest'

        if jsonPoly[x]=='NA':
            pass
        else:
            temp={
                'name':name,
                'type':typ,
                'polygon':polyType[x],
                'geometry':jsonPoly[x]
            }
            jsonOP.append(temp)
        jsonOP.append({
                'name':'GD',
                'type': 'Water',
                'polygon':3,
                'geometry':[[{'x':234, 'y':122}, {'x':370, 'y':94}, {'x':335, 'y':253}, {'x':210, 'y':229}, {'x':234, 'y':122}]]
            })
    return jsonOP

def latlongBounds(temp):    
    maxLatVal = -1000
    maxLongVal = -1000
    minLatVal = 1000
    minLongVal = 1000
    for i in range(len(temp)):
        if temp[i][0]>=maxLatVal:
            maxLatVal = temp[i][0]
            maxLat = i
        if temp[i][1]>=maxLongVal:
            maxLongVal = temp[i][1]
            maxLong = i
        if temp[i][0]<=minLatVal:
            minLatVal = temp[i][0]
            minLat = i
        if temp[i][1]<=minLongVal:
            minLongVal = temp[i][1]
            minLong = i
    bounds = [maxLatVal, maxLongVal, minLatVal, minLongVal]
    diffLat = maxLatVal - minLatVal
    diffLong = maxLongVal - minLongVal
    return (bounds, diffLat, diffLong)

def computeEEE(coordinates, min_coord, max_coord):
    st.write('8. EEE started')
    bounds, diffLat, diffLong = latlongBounds(coordinates)
    historicTag={'historic': True, 'tourism':True, 'leisure':'nature_reserve'}
    buildingTag = {'building':True}
    forestTag = {'boundary':['forest','forest_compartment'],'natural':'water','water':True}

    G1 = ox.geometries.geometries_from_bbox(bounds[1],bounds[3],bounds[0],bounds[2], historicTag)
    G2 = ox.geometries.geometries_from_bbox(bounds[1],bounds[3],bounds[0],bounds[2], buildingTag)
    G3 = ox.geometries.geometries_from_bbox(bounds[1],bounds[3],bounds[0],bounds[2], forestTag)

    G1.to_csv('historicTag.csv')
    G2.to_csv('buildingTag.csv')
    G3.to_csv('forestTag.csv')

    historic = pd.read_csv('historicTag.csv')
    building = pd.read_csv('buildingTag.csv')
    forest = pd.read_csv('forestTag.csv')

    historic.geometry.dropna(inplace=True)
    building.geometry.dropna(inplace=True)
    forest.geometry.dropna(inplace=True)

    remove=['hotel','apartment','hostel','guest_house']
    reflist=historic.columns
    for x in remove:
        if x in reflist:
            historic = historic[(historic['tourism']!=x)]

    jsonPoly,polyType = get_poly_coords(historic, min_coord, max_coord)
    historicJSON = get_historic_json_op(jsonPoly,polyType,historic)

    jsonPoly,polyType = get_poly_coords(building, min_coord, max_coord)
    buildingJSON = get_building_json_op(jsonPoly,polyType,building)

    jsonPoly,polyType = get_poly_coords(forest, min_coord, max_coord)
    forestJSON = get_forest_json_op(jsonPoly,polyType,forest)

    st.write('9. EEE finished')
    return historicJSON, buildingJSON, forestJSON
