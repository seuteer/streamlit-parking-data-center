# 地理数据获取

import os
import streamlit as st
import geopandas as gpd
import pandas as pd
import leafmap.foliumap as leafmap

# # 设置geemap环境变量
# os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]

def app():
    st.title('Geographic Data')

    # 定义全局变量，表示提示信息的占位符
    global info_st
    info_st = st.empty()
    gpkg_path = os.path.join(st.session_state.data_temp, 'birmingham.gpkg')

    # 获取存储地理数据的字典
    dict_layer_gdf = dowmload_osm_data(gpkg_path=gpkg_path)

    # 保存gdf数据到gpkg图层库
    gdfs_to_gpkg(dict_layer_gdf=dict_layer_gdf, gpkg_path=gpkg_path)

    # 地理数据可视化
    row1_col1, row1_col2 = st.columns([3, 1])
    with row1_col2:
        # 设置 选择框 选择底图
        basemap = st.radio(
            "请选择地图",
            ('OpenStreetMap', 'ROADMAP', 'HYBRID'))
        # 设置 多选框 选择可视化数据
        layer_list = st.multiselect(
            '请选择图层',
            # 列出所有图层
            list(dict_layer_gdf.keys()),
            # 默认首选的元素
            []
        )
        info_st.success(f"您添加了 {layer_list[-1]} 图层" if len(layer_list) else '请选择图层')
    with row1_col1:
        lon, lat = leafmap.gdf_centroid(dict_layer_gdf['parking'])
        m = leafmap.Map(center=[lat, lon], zoom=13)
        m.add_basemap(basemap=basemap)
        for layer in layer_list:
            m.add_gdf(dict_layer_gdf[layer], layer_name=layer)
        m.to_streamlit(width=900, height=500)


@st.cache(suppress_st_warning=True)
def dowmload_osm_data(gpkg_path):
    import osmnx as ox
    import pyproj

    # 定义字典存储地理数据
    dict_layer_gdf = {
        'parking': None,
        'nodes': None,
        'edges': None,
        'pois': None,
        'buildings': None
    }

    # 首先读取数据，若失败则下载数据
    if os.path.exists(gpkg_path):
        info_st.info("正在加载云端数据...")
        for layer in dict_layer_gdf.keys():
            dict_layer_gdf[layer] = gpd.read_file(gpkg_path, layer=layer)
        info_st.success("云端数据加载完毕!")
    else:
        # 停车场点要素
        info_st.info("parking download...")
        df = gpd.read_file(os.path.join(st.session_state.data_input, 'bmh_location.csv'))
        df[['longtitude', 'latitude']] = df[['longtitude', 'latitude']].apply(pd.to_numeric)
        dict_layer_gdf['parking'] = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longtitude, df.latitude), crs=pyproj.CRS("WGS84"))
        # 获取并扩大研究范围
        west, south, east, north = leafmap.gdf_bounds(dict_layer_gdf['parking'])
        dew, dns = (east-west)/2, (north-south)/2
        west, south, east, north = west-dew, south-dns, east+dew, north+dns
        # 道路网节点和线
        info_st.info("roads download...")
        graph = ox.graph_from_bbox(north, south, east, west, network_type="drive", clean_periphery=True)
        dict_layer_gdf['nodes'], dict_layer_gdf['edges'] = ox.graph_to_gdfs(graph)
        # POI
        info_st.info("pois download...")
        pois = ox.geometries_from_bbox(north, south, east, west, tags={"amenity": True})
        dict_layer_gdf['pois'] = pois[pois['geometry'].type.isin(['Point'])]  # 筛选点要素
        # 建筑实体
        info_st.info("buildings download...")
        buildings = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
        dict_layer_gdf['buildings'] = buildings[buildings['geometry'].type.isin(['Polygon'])]  # 保留 Polygon 面要素
    
    return dict_layer_gdf


@st.cache(suppress_st_warning=True)
def gdfs_to_gpkg(dict_layer_gdf, gpkg_path):
    if os.path.exists(gpkg_path):
        info_st.info(gpkg_path + " exists!")
    else:
        # 批量保存 gdf 数据
        for layer, gdf in dict_layer_gdf.items():
            info_st.info(layer + " saving...")
            # 为了成功保存，转换数据类型为字符串
            gdf = gdf.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
            gdf.to_file(
                filename=gpkg_path,
                driver='GPKG', 
                layer=layer
                )