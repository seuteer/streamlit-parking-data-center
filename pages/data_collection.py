# 数据获取
import os
from turtle import width
import streamlit as st
import geopandas as gpd
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def app():
    st.write('# Data Collection')

    # 定义全局变量，表示下载信息的占位符
    global width, height, download_st, row1_col1, row1_col2
    width = 950
    height = 600
    download_st = st.empty()
    row1_col1, row1_col2 = st.columns([2, 1])

    # 从停车场点要素获取研究框
    east, south, west, north = get_bbox_from_points(path=os.path.join(st.session_state.data_input, 'bmh_location.csv'), is_bigbox=True)
    # 根据研究框获取地理数据
    get_data_from_bbox(east=east, south=south, west=west, north=north, is_download=True)
    # 清空下载信息占位符
    download_st.empty()

    # 地图可视化
    with row1_col2:
        backend = st.selectbox(
            "Select a plotting backend", ["folium", "geemap"], index=0
        )
    if backend == "folium":
        vis_folium()
    elif backend == "geemap":
        vis_geemap()


@st.cache
def get_bbox_from_points(path, is_bigbox=False):
    import pyproj
    import shapely
    # 导入停车场坐标
    df = gpd.read_file(path)
    df[['longtitude', 'latitude']] = df[['longtitude', 'latitude']].apply(pd.to_numeric)
    # 创建 parking
    parking = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longtitude, df.latitude), crs=pyproj.CRS("WGS84"))
    # 保存 parking 点要素
    gdf_to_geojson(gdf=parking, path=os.path.join(st.session_state.data_temp, 'parking.geojson'))
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
def get_data_from_bbox(east, south, west, north, is_download=True):
    import osmnx as ox
    # 城市道路 line
    road_drive = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    # 将网络转换为节点和边
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(road_drive)
    gdf_edges = gdf_edges.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    gdf_nodes = gdf_nodes.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    download_st.write('road_drive download!')
    # POI point
    pois = ox.geometries_from_bbox(north, south, east, west, tags={"amenity": True})
    pois = pois.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    pois = pois[pois['geometry'].type.isin(['Point'])]
    download_st.write('pois download!')
    # 建筑足迹 polygon
    building_footprints = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
    building_footprints = building_footprints.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    building_footprints = building_footprints[building_footprints['geometry'].type.isin(['Polygon'])]
    download_st.write('building footprints download!')
    # 保存数据
    if is_download:
        gdf_to_geojson(gdf=gdf_edges, path=os.path.join(st.session_state.data_temp, 'edges.geojson'))
        gdf_to_geojson(gdf=gdf_nodes, path=os.path.join(st.session_state.data_temp, 'nodes.geojson'))
        gdf_to_geojson(gdf=pois, path=os.path.join(st.session_state.data_temp, 'pois.geojson'))
        gdf_to_geojson(gdf=building_footprints, path=os.path.join(st.session_state.data_temp, 'building_footprints.geojson'))


@st.cache(suppress_st_warning=True)
def gdf_to_geojson(gdf, path):
    if os.path.exists(path):
        download_st.write(f"{os.path.basename(path)} already exists!")
    else:
        gdf.to_file(filename=path, driver='GeoJSON', encodeing='utf-8')
        download_st.write(f"{os.path.basename(path)} saved!")

def vis_folium():
    # visualize with folium
    import folium
    from streamlit_folium import folium_static

    m = folium.Map(location=[52.479415075, -1.9001145545], zoom_start=14)
    # 设置 多选框 选择可视化数据
    with row1_col2:
        options = st.multiselect(
            'Choose to visualize geographic data',
            # 筛选 data_temp 中所有的 .geojson 文件，生成列表
            [geoj for geoj in os.listdir(st.session_state.data_temp) if ('.geojson' in geoj)],
            # 默认首选的元素
            ['parking.geojson']
            )
    for geoj in options:
        folium.GeoJson(gpd.read_file(os.path.join(st.session_state.data_temp, geoj)), name=geoj).add_to(m)
    with row1_col1:
        m.add_child(folium.LayerControl())
        folium_static(m, width=width, height=height)

def vis_geemap():
    import geemap
    # 设置环境变量
    os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]

    m = geemap.Map(center=[52.479415075, -1.9001145545], zoom=14)
    # 设置 多选框 选择可视化数据
    with row1_col2:
        options = st.multiselect(
            'Choose to visualize geographic data',
            # 筛选 data_temp 中所有的 .geojson 文件，生成列表
            [geoj for geoj in os.listdir(st.session_state.data_temp) if ('.geojson' in geoj)],
            # 默认首选的元素
            ['parking.geojson']
            )
    for geoj in options:
        m.add_geojson(os.path.join(st.session_state.data_temp, geoj), layer_name=geoj)
    with row1_col1:
        m.to_streamlit(width=width, height=height)