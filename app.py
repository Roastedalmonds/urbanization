from matplotlib.pyplot import axis
import streamlit as st
import streamlit.components.v1 as components
import streamlit_folium
import numpy as np
import geopandas as gpd
import folium
import geemap.colormaps as cm
import geemap.foliumap as gmf
import geemap
import bundle
import pickle
import ee
import math
import pandas as pd
from shapely import wkt
from shapely.ops import split
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
# import georasters as gr
from scipy import spatial
import tagee
import predict
import EEE
import osmnx as ox
import json
import utils
import tempfile
import os
import uuid

import warnings
warnings.filterwarnings('ignore')

def uploaded_file_to_gdf(data):
    _, file_extension = os.path.splitext(data.name)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f"{file_id}{file_extension}")

    with open(file_path, "wb") as file:
        file.write(data.getbuffer())

    if file_path.lower().endswith(".kml"):
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    return gdf

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


def explode(indf):
    geometriess = []
    for x in range(len(indf.geoms)):
        geometriess.append(indf[x])
    return geometriess

def TAGEE(coordinates):
    srtm = ee.Image("USGS/SRTMGL1_003")
    worldCover = ee.ImageCollection("ESA/WorldCover/v100").sort('system:time_start', False).limit(1, 'system:time_start', False) \
                .first().clip(ee.Geometry.Polygon(coordinates))

    cityPoly = Polygon(coordinates)
    # cutBlocks =  _quadrat_cut_geometry(cityPoly, 0.098, 15)
    cutBlocks =  _quadrat_cut_geometry(cityPoly, 0.2, 7)
    st.write('4. cutBlocks done')

    geom = ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon(coordinates)))
    gaussianFilter = ee.Kernel.gaussian(radius=3, sigma=2, units='pixels', normalize=True)
    srtmSmooth = srtm.convolve(gaussianFilter).resample("bilinear")
    terrainMetrics = tagee.terrainAnalysis(srtmSmooth, geom.geometry())
    superReducer = ee.Reducer.median().combine(ee.Reducer.minMax(), "", True)
    reduction = terrainMetrics.reduceRegions(geom,superReducer)
    st.write('5. TAGEE_city done')

    cityTagee=[]
    cityTagee = reduction.getInfo()
    cityFeatures = list(cityTagee['features'][0]['properties'].keys())
    cityFeaturesVal = []
    cityFeatures2=[]
    
    for i in range(len(cityFeatures)):
        cityFeaturesVal.append(cityTagee['features'][0]['properties'][cityFeatures[i]])
        cityFeatures2.append(cityFeatures[i]+'_city')
    df = pd.DataFrame(columns = cityFeatures+cityFeatures2)
    
    for x in range(len(cutBlocks.geoms)):
        coords=[]
        for i,j in zip(cutBlocks[x].boundary.xy[0],cutBlocks[x].boundary.xy[1]):
            coords.append([i,j]) 
        geom = ee.FeatureCollection(ee.Feature(ee.Geometry.Polygon(coords)))
        reduction = terrainMetrics.reduceRegions(geom,superReducer)
        blockTagee=[]
        blockTagee = reduction.getInfo()
        df = df.append(blockTagee['features'][0]['properties'], ignore_index=True)

    for x in range(len(cityFeatures2)):
        df[cityFeatures2[x]]=cityTagee['features'][0]['properties'][cityFeatures[x]]
    datalist = explode(cutBlocks)
    cityBlocks = gpd.GeoDataFrame(gpd.GeoSeries(datalist))
    st.write('6. TAGEE_tile done')

    cityBlocks.columns=['geometry']

    df = df.reset_index()
    # df.drop(['index'], axis=1, inplace=True)
    st.dataframe(df)
    # st.write(df.columns)
    file_name = 'location.csv'
    df.to_csv(file_name)

    print('7. data created and saved')
    return df, datalist

# main code
st.title('Urbanization')

m = gmf.Map(basemap='HYBRID', plugin_Draw=True, draw_export=True,
               locate_control=True, plugin_LatLngPopup=True)
m.add_basemap('ROADMAP')

row1col1, row1col2 = st.columns(2)

with row1col1:
    keyword = st.text_input('Search for a location:', '')

with row1col2:
    if keyword:
        locations = gmf.geocode(keyword)
        if locations is not None and len(locations) > 0:
            str_locations = [str(g)[1:-1] for g in locations]
            location = st.selectbox("Select a location:", str_locations)
            loc_index = str_locations.index(location)
            selected_loc = locations[loc_index]
            lat, lng = selected_loc.lat, selected_loc.lng

is_lat_lng = st.checkbox('Use Latitude-Longitude')

if is_lat_lng:
    row2col1, row2col2 = st.columns(2)

    with row2col1:
        lat = st.number_input(label='Enter Latitude')
    with row2col2:
        lng = st.number_input(label='Enter Longitude')

try:
    m.set_center(lng, lat, 12)
    st.session_state["zoom_level"] = 12
except:
    pass

data = st.file_uploader("Upload a GeoJSON file to use as an ROI.",
                        type=["geojson", "kml", "zip"])

if data:
    gdf = uploaded_file_to_gdf(data)
    # st.write(gdf)
    st.write('Area: ', list(utils.get_area_polygon(gdf)))
    st.session_state["roi"] = geemap.geopandas_to_ee(gdf, geodesic=False)
    m.add_gdf(gdf, "ROI")

    coordinates = list(gdf['geometry'].exterior[0].coords)
    # print('coord', coordinates)
    st.write('1. Coordinates Done')

    min_coord, max_coord, diffLat, diffLong = utils.get_bounding_box(coordinates)
    st.write('2. BBox Done')

    # print(min_coord, max_coord)
    temp0, temp1 = [], []
    for coord in coordinates:
        min_max = utils.minmax(coord, min_coord, max_coord, diffLat, diffLong)
        temp0.append(dict({'x':min_max[0],'y':min_max[1]}))
        temp1.append(dict({'x':coord[0],'y':coord[1]}))

    coords_json = {'normalized': temp0, 
                    'original': temp1, 
                    # 'area': get_area_polygon(gdf),
                    'bbox': {'min_x':min_coord[0],
                            'min_y':min_coord[1],
                            'max_x':max_coord[0],
                            'max_y':max_coord[1]
                        },
                    'diff': [diffLat, diffLong]
                    }

    with open("./Generator/src/ts/impl/city_loc.json", "w") as outfile:
        json.dump(coords_json, outfile)
    st.write('3. City polygon json done')
    print(coords_json)
    bbox = [min_coord, [min_coord[0], max_coord[1]], max_coord, [max_coord[0], min_coord[1]]]
    data, multipoly = TAGEE(bbox)

    historic, buildings, forest = EEE.computeEEE(coordinates[:-1], min_coord, max_coord)
    # historic, buildings, forest = [], [], []
    coasts = {'historic': historic, 'buildings': buildings, 'forest': forest}

    with open("./Generator/src/ts/impl/coasts.json", "w") as outfile:
        json.dump(coasts, outfile)
    st.write('10. 3E jsons done')
    print()
    print(coasts)
    data = pd.read_csv('location.csv')
    data.drop(['index'], axis=1, inplace=True)
    st.dataframe(data)
    centroids = predict.gron_image(data, gdf, min_coord, max_coord)

    from PIL import Image
    try:
        image1 = Image.open(r'C:\Users\mahim\Downloads\map (1).png')
        image2 = Image.open(r'C:\Users\mahim\Downloads\map (2).png')
        image3 = Image.open(r'C:\Users\mahim\Downloads\map (3).png')
        image4 = Image.open(r'C:\Users\mahim\Downloads\map (4).png')
        
        st.image(image1)
        st.image(image2)
        st.image(image3)
        st.image(image4)

    except:
        pass

m.to_streamlit(height=600)
