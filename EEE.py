import imp
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely import wkt
import math
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import streamlit as st

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def minmax(point, min, max):
    return [(point[0] - min[0]) / (max[0] - min[0]), (point[1] - min[1]) / (max[1] - min[1])]

def get_poly_coords(df, min_coord, max_coord):
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326')
    jsonPoly=[]
    polyType=[]
    for x in range(len(gdf)):
        if type(gdf.loc[x,'geometry'])==Polygon:
            try:
                polys=[]
                for i,j in list(zip(gdf.loc[x,'geometry'].boundary.xy[0],gdf.loc[x,'geometry'].boundary.xy[1])):
                    temp99 = minmax((i, j), min_coord, max_coord)
                    polys.append(dict({'x':temp99[0],'y':temp99[1]}))
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
                        temp99 = minmax((i, j), min_coord, max_coord)
                        polys.append(dict({'x':temp99[0],'y':temp99[1]}))
                    # polys.append(polys[0])
                    mulPoly.append(polys)
                jsonPoly.append(mulPoly)
                polyType.append(4)
            except:
                jsonPoly.append('NA')
                polyType.append('NA')

        elif type(gdf.loc[x,'geometry'])==Point:
            temp99 = minmax((gdf.loc[x,'geometry'].xy[0], gdf.loc[x,'geometry'].xy[1]), min_coord, max_coord)
            jsonPoly.append([dict({'x':temp99[0],'y':temp99[1]})])
            polyType.append(1)

        elif type(gdf.loc[x,'geometry'])==LineString:
            try:
                for i,j in zip(gdf.loc[x,'geometry'].boundary.xy[0],gdf.loc[x,'geometry'].boundary.xy[1]):
                    temp99 = minmax((i, j), min_coord, max_coord)
                    polys.append(dict({'x':temp99[0],'y':temp99[1]}))
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
