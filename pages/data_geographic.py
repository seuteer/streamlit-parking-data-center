# 地理数据获取

import os
import streamlit as st
import geopandas as gpd
import osmnx as ox
import pandas as pd
import leafmap.foliumap as leafmap
from PIL import Image  # 加载本地图片

# # 设置geemap环境变量
# os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]

def app():
    st.header('空间特征分析')
    st.session_state.info_st.success("地理数据获取与可视化👉")

    st.write("---")
    st.subheader("地理数据获取")
    gpkg_path = os.path.join(st.session_state.data_temp, 'birmingham.gpkg')  # 定义存储地理数据的路径
    temp = st.info("正在加载云端数据...")
    dict_layer_gdf = dowmload_osm_data(gpkg_path)
    temp.success("云端数据加载完毕!")


    st.write("---")
    st.subheader("交通网络分析")
    network_analysis(dict_layer_gdf, gpkg_path)

    # 保存地理文件
    # gdfs_to_gpkg(dict_layer_gdf, gpkg_path)

    st.write("---")
    st.subheader("地理数据可视化")
    plot_leafmap(dict_layer_gdf)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def dowmload_osm_data(gpkg_path):
    import pyproj
    dict_layer_gdf = {}  # 定义存储地理数据的字典
    if not os.path.exists(gpkg_path):
        # 若更新或不存在地理数据，则下载地理数据
        # 停车场点要素
        temp = st.info("parking download...")
        df = gpd.read_file(os.path.join(st.session_state.data_input, 'bmh_location.csv'))
        df[['longtitude', 'latitude']] = df[['longtitude', 'latitude']].apply(pd.to_numeric)
        dict_layer_gdf['parking'] = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longtitude, df.latitude), crs=pyproj.CRS("WGS84"))
        # 获取并扩大研究范围
        west, south, east, north = leafmap.gdf_bounds(dict_layer_gdf['parking'])
        dew, dns = (east-west)/2, (north-south)/2
        west, south, east, north = west-dew, south-dns, east+dew, north+dns
        # 道路网络
        temp.info("roads download...")
        graph = ox.graph_from_bbox(north, south, east, west, network_type="drive", clean_periphery=True)
        ox.save_graphml(graph, filepath=gpkg_path.replace('.gpkg', '.graphml'))
        # POI
        temp.info("pois download...")
        pois = ox.geometries_from_bbox(north, south, east, west, tags={"amenity": True})
        dict_layer_gdf['pois'] = pois[pois['geometry'].type.isin(['Point'])]  # 筛选点要素
        # 建筑实体
        temp.info("buildings download...")
        buildings = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
        dict_layer_gdf['buildings'] = buildings[buildings['geometry'].type.isin(['Polygon'])]  # 保留 Polygon 面要素
        temp.success("地理数据更新完毕！")
    else:
        for layer in ['parking', 'pois', 'buildings', 'nodes', 'edges']:
            dict_layer_gdf[layer] = gpd.read_file(gpkg_path, layer=layer)
    return dict_layer_gdf

def network_analysis(dict_layer_gdf, gpkg_path):
    fig_len = Image.open('images/fig_len.png')
    fig_cen = Image.open('images/fig_cen.png')
    fig_time = Image.open('images/fig_time.png')
    fig_speed = Image.open('images/fig_speed.png')
    st.success("根据道路长度和中心度渲染网络！")
    col1, col2 = st.columns(2)
    with col1.expander("道路长度", expanded=True):
        st.image(fig_len)
    with col2.expander("道路中心度", expanded=True):
        st.image(fig_cen)
    st.success("根据道路行驶时间和速度渲染网络！")
    col1, col2 = st.columns(2)
    with col1.expander("行驶时间", expanded=True):
        st.image(fig_time)
    with col2.expander("行驶速度", expanded=True):
        st.image(fig_speed)

    # import networkx as nx
    # if not os.path.exists(gpkg_path.replace('.gpkg', '.graphml')):
    #     # 如果不存在道路网络文件，则抛出异常
    #     st.session_state.info_st.error("未发现网络文件，请更新地理数据！")
    # else:
    #     # 如果更新或不存在节点和边的数据，则进行网络分析
    #     graph = ox.load_graphml(filepath=gpkg_path.replace('.gpkg', '.graphml'))
    #     temp = st.info("正在计算网络的基本描述性几何和拓扑度量...")
    #     G_proj = ox.project_graph(graph)  # 网络投影
    #     nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)  # 节点投影
    #     graph_area_m = nodes_proj.unary_union.convex_hull.area
    #     stats_dict = ox.basic_stats(G_proj, area=graph_area_m, clean_int_tol=15)
    #     st.json(stats_dict)
    #     temp.success("网络的基本描述性几何和拓扑度量计算完毕！")

    #     temp = st.info("正在计算道路长度和中心度...")
    #     edge_centrality = nx.closeness_centrality(nx.line_graph(graph))
    #     nx.set_edge_attributes(graph, edge_centrality, "edge_centrality")
    #     ec_len = ox.plot.get_edge_colors_by_attr(graph, "length", cmap="Reds")
    #     fig_len, ax = ox.plot_graph(graph, edge_color=ec_len, edge_linewidth=1, node_size=0, bgcolor='white')
    #     ec_cen = ox.plot.get_edge_colors_by_attr(graph, "edge_centrality", cmap="inferno")
    #     fig_cen, ax = ox.plot_graph(graph, edge_color=ec_cen, edge_linewidth=1, node_size=0, bgcolor='white')
    #     temp.success("根据道路长度和中心度渲染网络！")

    #     temp = st.info("正在计算道路行驶时间和速度...")
    #     graph = ox.speed.add_edge_speeds(graph)
    #     graph = ox.speed.add_edge_travel_times(graph)
    #     ec_speed = ox.plot.get_edge_colors_by_attr(graph, "speed_kph", cmap="cool")
    #     fig_speed, ax = ox.plot_graph(graph, edge_color=ec_speed, edge_linewidth=1, node_size=0, bgcolor='white')
    #     ec_time = ox.plot.get_edge_colors_by_attr(graph, "travel_time", cmap="Blues")
    #     fig_time, ax = ox.plot_graph(graph, edge_color=ec_time, edge_linewidth=1, node_size=0, bgcolor='white')
    #     temp.success("根据道路行驶时间和速度渲染网络！")

    #     col1, col2 = st.columns(2)
    #     with col1.expander("道路长度", expanded=True):
    #         st.pyplot(fig=fig_len)
    #     with col2.expander("道路中心度", expanded=True):
    #         st.pyplot(fig=fig_cen)
    #     col1, col2 = st.columns(2)
    #     with col1.expander("行驶时间", expanded=True):
    #         st.pyplot(fig=fig_time)
    #     with col2.expander("行驶速度", expanded=True):
    #         st.pyplot(fig=fig_speed)

    #     dict_layer_gdf['nodes'], dict_layer_gdf['edges'] = ox.graph_to_gdfs(graph)
    #     return dict_layer_gdf

# def gdfs_to_gpkg(dict_layer_gdf, gpkg_path):
#     if not os.path.exists(gpkg_path):
#         # 批量保存 gdf 数据
#         temp = st.info("正在存储地理数据...")
#         download_prog = st.progress(0); i=0
#         for layer, gdf in dict_layer_gdf.items():
#             temp.info(layer + ' saving...')
#             # 为了成功保存，转换数据类型为字符串
#             gdf = gdf.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
#             gdf.to_file(filename=gpkg_path, driver='GPKG', layer=layer)
#             i+=1/len(dict_layer_gdf); download_prog.progress(i)  # 模型进度为0.0-1.0
#         temp.success("地理数据存储完毕！")

def plot_leafmap(dict_layer_gdf):
    import branca  # 配合 folium 进行配色
    import folium

    row1_col1, row1_col2 = st.columns([3, 1])
    with row1_col2:
        temp = st.info("自定义显示地图")
        # 设置 选择框 选择底图
        basemap = st.radio(
            "请选择地图",
            ('OpenStreetMap', 'ROADMAP', 'HYBRID'),
            index=1)
        # 设置 多选框 选择可视化数据
        layer_list = st.multiselect(
            '请选择图层',
            # 列出所有图层
            list(dict_layer_gdf.keys()),
            # 默认首选的元素
            ['parking', 'pois', 'buildings', 'edges']
        )
        temp.success(f"您添加了 {layer_list[-1]} 图层" if len(layer_list) else '请选择图层')
    with row1_col1:
        m = leafmap.Map()
        m.add_basemap(basemap=basemap)
        m.zoom_to_gdf(dict_layer_gdf['nodes'])
        for layer in layer_list:
            if layer == 'parking':
                m.add_gdf(  # 继承 folium.GeoJSON 类的参数
                    dict_layer_gdf['parking'],
                    tooltip=folium.GeoJsonTooltip(fields=['SystemCodeNumber'], aliases=['停车场编号']),
                    marker=folium.Marker(icon=folium.Icon(color='green', icon='car', prefix='fa')),
                    layer_name='parking',
                    zoom_to_layer=False
                )
            elif layer == 'nodes':
                nodes = dict_layer_gdf['nodes']
                nodes['street_count'] = nodes['street_count'].astype(dtype="int8")  # 此处会改变输出结果的值，我们是允许的
                colormap = branca.colormap.LinearColormap(
                    vmin=nodes['street_count'].min(),  
                    vmax=nodes['street_count'].max(),
                    colors=['green','red'],
                    caption="Street Count")
                m.add_gdf(  # 继承 folium.GeoJSON 类的参数
                    dict_layer_gdf['nodes'],
                    popup=folium.GeoJsonPopup(fields=['street_count'], aliases=['街道连接数']),  # GeoJsonPopup
                    marker=folium.CircleMarker(radius=2),  # Circle, CircleMarker or Marker
                    style_callback=lambda x: {"color": colormap(x["properties"]["street_count"]),},
                    hover_style={"color": 'gray'},
                    layer_name='nodes',
                    zoom_to_layer=False,
                    show=False
                )
            elif layer == 'pois':
                df_pois = leafmap.gdf_to_df(dict_layer_gdf['pois']).copy()
                df_pois['longitude'] = dict_layer_gdf['pois']['geometry'].x
                df_pois['latitude'] = dict_layer_gdf['pois']['geometry'].y
                df_pois['value'] = 1  # 创建热力图的值字段
                radius = row1_col2.slider('请选择热力图范围：', 5, 30, 15)
                m.add_heatmap(df_pois, value="value", radius=radius, name='pois', show=False)
            elif layer == 'edges':
                edges = dict_layer_gdf['edges']
                edges['edge_centrality'] = round(edges['edge_centrality'].astype(dtype="float64"), 5)  # 此处会改变输出结果的值，我们是允许的
                edges['length'] = round(edges['length'].astype(dtype="float64"), 1)
                edges['lanes'] = edges['lanes'].apply(lambda x: x if x!='nan' else 1)
                colormap = branca.colormap.LinearColormap(
                    vmin=edges['edge_centrality'].quantile(0.0),  # 0分位数
                    vmax=edges['edge_centrality'].quantile(1),  # 1分位数
                    colors=['darkgreen', 'green', 'red'],  # 调色板
                    caption="Edge Centrality")  # 标题
                m.add_gdf(
                    dict_layer_gdf['edges'],
                    layer_name='edges',
                    style_callback=lambda x: {"color": colormap(x["properties"]["edge_centrality"]), "weight": x["properties"]["lanes"]},
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name','edge_centrality','length','speed_kph','travel_time'], 
                        aliases=['道路名称','道路中心度','道路长度(m)','行驶速度(km/h)','行驶时间(s)']),
                    hover_style={"fillColor": "#ffaf00", "color": "green", "weight": 3},
                    zoom_to_layer=False,
                    show=False
                )
            elif layer == 'buildings':
                dict_layer_gdf['buildings'] = dict_layer_gdf['buildings'].to_crs(3857)  # 3857是投影坐标系，4326是经纬度坐标
                dict_layer_gdf['buildings']['area'] = round(dict_layer_gdf['buildings'].area, 2)   # 此处会改变输出结果的值，我们是允许的
                m.add_gdf(
                    dict_layer_gdf['buildings'], 
                    layer_name='buildings',
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name','area','amenity'], 
                        aliases=['建筑名称','总建筑面积(平方米)','建筑设施']),
                    zoom_to_layer=False, 
                    show=False
                )
        folium.LayerControl(collapsed=False).add_to(m)
        m.to_streamlit(width=900, height=500)