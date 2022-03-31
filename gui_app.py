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

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def uploaded_file_to_gdf(data):
    import tempfile
    import os
    import uuid

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
    return ([minLatVal, minLongVal], [maxLatVal, maxLongVal], diffLat, diffLong)

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

    cityId=[]
    for z in range(len(cityBlocks)):
        cityId.append('location_'+str(z))
    cityBlocks['id'] = cityId
    cityBlocks.columns=['geometry','id']
    cityBlocks.set_index('id', inplace=True)
    cityBlocks[['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] =  None
    cityBlocks['cls'] = 'Nopattern'
    
    # geemap.ee_export_image(worldCover, filename='Cover.tif', scale=90, file_per_band=True)
    # wc = gr.from_file('Cover.Map.tif')
    # try:
    #     worldCover = wc.to_geopandas()
    #     tree = spatial.KDTree(list(zip(worldCover['x'],worldCover['y'])))
    
    #     cover0=[]
    #     for name in cityBlocks.index.to_list():
    #         centroid = cityBlocks.loc[name].geometry.centroid
    #         point = (centroid.y, centroid.x)
    #         t = tree.query([centroid.x,centroid.y])
    #         cover0.append(worldCover.iloc[t[-1]]['value'])
    #     cityBlocks['cover'] = cover0

    # except:
    #     cover0=list(np.zeros(len(cityBlocks)))
    #     cityBlocks['cover'] = cover0
    # print('6. worldcover done')

    df = df.reset_index()
    # df.drop(['index'], axis=1, inplace=True)
    st.dataframe(df)
    st.write(df.columns)
    file_name = 'location.csv'
    df.to_csv(file_name)

    print('7. data created and saved')
    return df, datalist

def minmax(point, min, max):
    return [(point[0] - min[0]) / (max[0] - min[0]), (point[1] - min[1]) / (max[1] - min[1])]

def get_area_polygon(polygon):
    polygon = polygon.to_crs({'init': 'epsg:3857'})
    return (polygon.area) / 10**6

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
    st.write('Area: ', list(get_area_polygon(gdf)))
    st.session_state["roi"] = geemap.geopandas_to_ee(gdf, geodesic=False)
    m.add_gdf(gdf, "ROI")
    coordinates = list(gdf['geometry'].exterior[0].coords)
    # print('coord', coordinates)
    st.write('1. Coordinates Done')

    min_coord, max_coord, diffLat, diffLong = get_bounding_box(coordinates)
    st.write('2. BBox Done')

    # print(min_coord, max_coord)
    temp0, temp1 = [], []
    for coord in coordinates:
        min_max = minmax(coord, min_coord, max_coord)
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

    # data, multipoly = TAGEE(coordinates)

    historic, buildings, forest = EEE.computeEEE(coordinates[:-1], min_coord, max_coord)
    
    coasts = {'historic': historic, 'buildings': buildings, 'forest': forest}

    with open("./Generator/src/ts/impl/coasts.json", "w") as outfile:
        json.dump(coasts, outfile)
    st.write('10. 3E jsons done')
    
    data = pd.read_csv('location.csv')
    data.drop(['index'], axis=1, inplace=True)
    st.dataframe(data)
    centroids = predict.gron_image(data, gdf, min_coord, max_coord)

m.to_streamlit(height=600)
