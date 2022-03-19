# 数据获取

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from streamlit_folium import folium_static
import osmnx as ox
import geopandas as gpd
import pandas as pd
import folium
import pyproj
import shapely


# 定义全局变量（注意：当前路径是 app.py 所在文件夹的路径）
data_input = './data/input/'
data_output = './data/output/'
data_temp = './data/temp/'


def app():
    st.write('# Data Collection')

    if st.button("Run"):

        st.write('## get bbox from points')
        east, south, west, north = get_bbox_from_points(path=os.path.join(data_input, 'bmh_location.csv'), is_bigbox=True)

        st.write('## get data from bbox')
        road_drive, pois, building_footprints = get_data_from_bbox(east=east, south=south, west=west, north=north, is_download=True)
        
        st.write('## visualize with folium')
        m = vis_folium(road_drive, pois, building_footprints)
        folium_static(m)


@st.cache
def get_bbox_from_points(path, is_bigbox=False):
    # 导入停车场坐标
    df = gpd.read_file(path)
    df[['longtitude', 'latitude']] = df[['longtitude', 'latitude']].apply(pd.to_numeric)
    # 创建 gdf 对象
    parking = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longtitude, df.latitude), crs=pyproj.CRS("WGS84"))
    parking.to_file(os.path.join(data_temp, 'parking.json'), driver='GeoJSON', encoding='utf-8')
    # 创建 MultiPoint 对象，以获取研究区域
    box = shapely.geometry.MultiPoint(df[['longtitude', 'latitude']].values).bounds
    east, south, west, north = box
    # 扩大研究区域，默认不扩大
    if is_bigbox:
        w_e = west - east
        n_s = north - south
        east, south, west, north = east-w_e, south-n_s, west+w_e, north+n_s
    return east, south, west, north

@st.cache(suppress_st_warning=True)
def get_data_from_bbox(east, south, west, north, is_download=False):
    # 城市道路
    road_drive = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    st.write('road_drive download!')
    # POI
    pois = ox.geometries_from_bbox(north, south, east, west, tags={"amenity": True})
    pois = pois.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    pois = pois[pois['geometry'].type.isin(['Point'])]
    st.write('pois download!')
    # 建筑物
    building_footprints = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
    building_footprints = building_footprints.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    building_footprints = building_footprints[building_footprints['geometry'].type.isin(['Polygon'])]
    st.write('building footprints download!')
    # 保存数据
    if is_download:
        if os.path.exists(os.path.join(data_temp, 'pois.geojson')):
            st.write('data already exists!')
        else:
            st.write('Saving data to local!')
            ox.save_graph_shapefile(road_drive, filepath=os.path.join(data_temp, "road_drive/"))
            pois.to_file(os.path.join(data_temp, 'pois.geojson'), driver='GeoJSON')
            building_footprints.to_file(os.path.join(data_temp, 'building_footprints.geojson'), driver='GeoJSON')
            st.write('Saved successfully!')
    return road_drive, pois, building_footprints

@st.cache(hash_funcs={folium.folium.Map: id})
def vis_folium(road_drive, pois, building_footprints):
    m = ox.plot_graph_folium(road_drive, popup_attribute="name", weight=1, color="#8b0000", zoom=14, tiles='openstreetmap')
    folium.GeoJson(pois, name='pois', show=False).add_to(m)
    folium.GeoJson(building_footprints, name='building_footprints', show=False).add_to(m)
    folium.GeoJson(os.path.join(data_temp, 'parking.json'), name='parking').add_to(m)
    m.add_child(folium.LayerControl())
    return m