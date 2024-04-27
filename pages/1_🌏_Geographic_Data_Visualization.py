import os
import re
import pyproj
import branca
import folium
import streamlit as st
import geopandas as gpd
import osmnx as ox
import networkx as nx
import pandas as pd
import leafmap.foliumap as leafmap

# 设置geemap环境变量
os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]


@st.cache_data
def get_geometry_data(parking_file, gpkg_file, place=None, tif_file=None):
    """
    根据停车场经纬度坐标点，获取周边道路网、POI、建筑足迹（From OpenStreetMap）
    parking_file: 停车场经纬度csv表（包含longtitude和latitude列）（None表示停车场数据在gpkg中）
    gpkg_file: 存储地理信息的gpkg文件
    tif_file: 存储停车场区域的TIF高程文件（默认None表示没有该文件）
    place: 行政边界要素，输入行政边界名称（默认None表示不获取）
    """

    dict_layer_gdf = {}  # 定义存储地理数据的字典
    if not os.path.exists(gpkg_file):
        # 停车场
        if parking_file:
            df = gpd.read_file(parking_file)
            df[['longtitude', 'latitude']] = df[['longtitude', 'latitude']].apply(pd.to_numeric)
            parking_lot = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longtitude, df.latitude), crs=pyproj.CRS("WGS84"))
            parking_lot.to_file(filename=gpkg_file, driver='GPKG', layer='parking_lot')
        else:
            parking_lot = gpd.read_file(gpkg_file, layer='parking_lot')

        if place:
            boundary = ox.geocode_to_gdf(place)
            boundary.to_file(filename=gpkg_file, driver='GPKG', layer='boundary')
            boundary.geometry.unary_union

        # 研究范围
        west, south, east, north = leafmap.gdf_bounds(parking_lot)
        dew, dns = (east-west)/5, (north-south)/5
        west, south, east, north = west-dew, south-dns, east+dew, north+dns

        # 机动车道路网
        graph = ox.graph_from_bbox(
            north, south, east, west,
            network_type="drive",
            truncate_by_edge=True,
            clean_periphery=True  # clean_periphery=True 缓冲500m以获取更大范围
        )
        # 计算 edge_centrality
        edge_centrality = nx.closeness_centrality(nx.line_graph(graph))
        nx.set_edge_attributes(graph, edge_centrality, "edge_centrality")
        # 计算 speed_kph travel_time
        graph = ox.speed.add_edge_speeds(graph)
        graph = ox.speed.add_edge_travel_times(graph)
        # 计算节点高程 elevation 及道路坡度 grade (需要准备网络区域的DEM高程数据)
        if tif_file:
            ox.elevation.add_node_elevations_raster(G=graph, filepath=tif_file, cpus=1)
            ox.elevation.add_edge_grades(G=graph, add_absolute=True)
        ox.save_graph_geopackage(graph, filepath=gpkg_file)
        # 将图形另存/加载为graphml文件：这是保存网络的最佳方式，为以后的工作做准备
        ox.save_graphml(graph, gpkg_file.replace('gpkg', 'graphml'))

        pois = ox.geometries_from_bbox(north, south, east, west, tags={"amenity": True})
        pois = pois[pois['geometry'].type.isin(['Point'])]  # 保留 Point 要素
        pois = pois.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)  # 防止报错
        missing_values = pois.replace('nan', None).isnull().mean()
        columns_to_drop = missing_values[missing_values > 0.90].index
        pois.drop(columns_to_drop, axis=1, inplace=True)  # 删除缺失值多的列
        pois.to_file(filename=gpkg_file, driver='GPKG', layer='pois')

        buildings = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
        buildings = buildings[buildings['geometry'].type.isin(['Polygon'])]  # 保留 Polygon 面要素
        buildings = buildings.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)  # 防止报错
        missing_values = buildings.replace('nan', None).isnull().mean()
        columns_to_drop = missing_values[missing_values > 0.90].index
        buildings.drop(columns_to_drop, axis=1, inplace=True)  # 删除缺失值多的列
        buildings['area_m2'] = round(buildings.to_crs(3857).area, 2)  # 投影坐标系计算面积
        buildings.to_file(filename=gpkg_file, driver='GPKG', layer='buildings')

    for layer in ['parking_lot', 'pois', 'buildings', 'nodes', 'edges']:
        dict_layer_gdf[layer] = gpd.read_file(gpkg_file, layer=layer)
    return dict_layer_gdf

def plot_leafmap(dict_layer_gdf):
    row1_col1, row1_col2 = st.columns([1, 1])
    with row1_col1:
        # 设置 选择框 选择底图
        basemap = st.radio(
            "Please select a map",
            ('OpenStreetMap', 'ROADMAP', 'HYBRID'),
            index=1)
    with row1_col2:
        # 设置 多选框 选择可视化数据
        layer_list = st.multiselect(
            'Please select layers',
            # 列出所有图层
            list(dict_layer_gdf.keys()),
            # 默认首选的元素
            ['parking_lot', 'nodes', 'edges']
        )
        
    # 制作地图
    m = leafmap.Map()
    m.add_basemap(basemap=basemap)
    m.zoom_to_gdf(dict_layer_gdf['edges'])
    for layer in layer_list:
        parking_lot = dict_layer_gdf['parking_lot']
        if layer == 'parking_lot':
            m.add_gdf(
                parking_lot,
                marker=folium.Marker(icon=folium.Icon(color='green', icon='car', prefix='fa')),
                layer_name='parking_lot',
            )
        elif layer == 'nodes':
            nodes = dict_layer_gdf['nodes']
            colormap_node = branca.colormap.LinearColormap(
                vmin=nodes['street_count'].min(),
                vmax=nodes['street_count'].max(),
                colors=['green', 'red'],
                caption="Street Count")
            m.add_gdf(
                nodes,
                marker=folium.CircleMarker(radius=1),  # Circle, CircleMarker or Marker
                style_function=lambda x: {"color": colormap_node(x["properties"]["street_count"]),},
                hover_style={"color": 'gray'},
                layer_name='nodes',
                info_mode='on_click',
                zoom_to_layer=False,
                show=False
            )
        elif layer == 'pois':
            pois = dict_layer_gdf['pois']
            df_pois = leafmap.gdf_to_df(pois).copy()
            df_pois['longitude'] = pois['geometry'].x
            df_pois['latitude'] = pois['geometry'].y
            df_pois['value'] = 1  # 创建热力图的值字段
            radius = row1_col2.slider('Please select the heatmap range:', 5, 30, 15)
            m.add_heatmap(df_pois, value="value", radius=radius, name='pois', show=False)
        elif layer == 'edges':
            edges = dict_layer_gdf['edges']
            edges['lanes'] = edges['lanes'].apply(lambda s: sum([int(i) for i in re.findall(r'\d', s)]))
            edges.loc[edges['lanes'] == 0, 'lanes'] = 1
            colormap_edge = branca.colormap.LinearColormap(
                vmin=edges['speed_kph'].min(),
                vmax=edges['speed_kph'].max(),
                colors=['red', 'green', 'darkgreen'],
                caption="Average Driving Speed")
            m.add_gdf(
                edges,
                layer_name='edges',
                style_function=lambda x: {"color": colormap_edge(x["properties"]["speed_kph"]), "weight": x["properties"]["lanes"]},
                hover_style={"fillColor": "#ffaf00", "color": "green", "weight": 3},
            )
        elif layer == 'buildings':
            buildings = dict_layer_gdf['buildings']
            m.add_gdf(
                buildings,
                layer_name='buildings',
                info_mode='on_click',
                zoom_to_layer=False,
                show=False
            )
    folium.LayerControl(collapsed=False).add_to(m)
    m.to_streamlit(height=500)


def app():
    st.header('Geographic Data Visualization')
    st.sidebar.success("Geographic data acquisition and visualization 👉")

    st.write("---")
    st.subheader("geographic data visualization")
    temp = st.info("Loading cloud data...")
    dict_layer_gdf = get_geometry_data(
        parking_file='./data/input/'+'birmingham_loc.csv',
        gpkg_file='./data/temp/'+'birmingham.gpkg',
        place='Birmingham, UK',
        tif_file='./data/input/'+'birmingham_dem.tif',
    )
    temp.success("Cloud data loaded!")
    plot_leafmap(dict_layer_gdf)

app()