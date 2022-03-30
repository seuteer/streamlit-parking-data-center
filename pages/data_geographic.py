# 地理数据获取

import os
import streamlit as st
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap


# # 设置geemap环境变量
# os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]

def app():
    st.title('Geographic Data')
    st.session_state.info_st.success("地理数据获取与可视化👉")

    st.write("---")
    st.subheader("地理数据获取")
    update = st.button("更新地理数据", help="这可能需要花费几分钟的时间从 OSM 获取数据（默认直接加载云端缓存数据）")
    gpkg_path = os.path.join(st.session_state.data_temp, 'birmingham.gpkg')  # 定义存储地理数据的路径
    dict_layer_gdf = dowmload_osm_data(gpkg_path, update)

    st.write("---")
    st.subheader("交通网络分析")
    update = st.button("显示网络分析", help="如需显示网络分析请点击此按钮（默认不显示网络分析）")
    dict_layer_gdf = network_analysis(dict_layer_gdf, gpkg_path, update)
    gdfs_to_gpkg(dict_layer_gdf, gpkg_path)

    st.write("---")
    st.subheader("地理数据可视化")
    plot_leafmap(dict_layer_gdf)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def dowmload_osm_data(gpkg_path, update):
    import pyproj
    dict_layer_gdf = {}  # 定义存储地理数据的字典
    if not update and os.path.exists(gpkg_path):
        # 如果不更新数据且存在数据的话
        temp = st.info("正在加载云端数据...")
        for layer in ['parking', 'pois', 'buildings', 'nodes', 'edges']:
            dict_layer_gdf[layer] = gpd.read_file(gpkg_path, layer=layer)
        temp.success("云端数据加载完毕!")
    else:
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
    return dict_layer_gdf

def network_analysis(dict_layer_gdf, gpkg_path, update):
    import networkx as nx
    if not update and len(dict_layer_gdf.keys())==5:
        # 如果不更新且存在节点和边数据的话（默认不显示网络分析）
        pass
    elif not os.path.exists(gpkg_path.replace('.gpkg', '.graphml')):
        # 如果不存在网络文件，则抛出异常
        st.session_state.info_st.error("未发现网络文件，请更新地理数据！")
    else:
        graph = ox.load_graphml(filepath=gpkg_path.replace('.gpkg', '.graphml'))
        temp = st.info("正在计算网络的基本描述性几何和拓扑度量...")
        G_proj = ox.project_graph(graph)  # 网络投影
        nodes_proj = ox.graph_to_gdfs(G_proj, edges=False)  # 节点投影
        graph_area_m = nodes_proj.unary_union.convex_hull.area
        stats_dict = ox.basic_stats(G_proj, area=graph_area_m, clean_int_tol=15)
        st.json(stats_dict)
        temp.success("网络的基本描述性几何和拓扑度量计算完毕！")

        temp = st.info("正在计算道路中心度...")
        edge_centrality = nx.closeness_centrality(nx.line_graph(graph))
        nx.set_edge_attributes(graph, edge_centrality, "edge_centrality")
        ec = ox.plot.get_edge_colors_by_attr(graph, "edge_centrality", cmap="inferno")
        fig, ax = ox.plot_graph(graph, edge_color=ec, edge_linewidth=1, node_size=0, bgcolor='white')
        st.pyplot(fig=fig)
        temp.success("根据道路中心度渲染网络！")

        temp = st.info("正在计算道路速度和行驶时间...")
        graph = ox.speed.add_edge_speeds(graph)
        graph = ox.speed.add_edge_travel_times(graph)
        ec_speed = ox.plot.get_edge_colors_by_attr(graph, "speed_kph", cmap="inferno")
        fig, ax = ox.plot_graph(graph, edge_color=ec_speed, edge_linewidth=1, node_size=0, bgcolor='white')
        st.pyplot(fig=fig)
        ec_time = ox.plot.get_edge_colors_by_attr(graph, "travel_time", cmap="inferno")
        fig, ax = ox.plot_graph(graph, edge_color=ec_time, edge_linewidth=1, node_size=0, bgcolor='white')
        st.pyplot(fig=fig)
        temp.success("根据道路速度和行驶时间渲染网络！")

        temp = st.info("正在计算节点高程及道路坡度...")
        ox.elevation.add_node_elevations_raster(G=graph, filepath=os.path.join(st.session_state.data_input, 'DEM-birmingham.tif'), cpus=1)
        ox.elevation.add_edge_grades(G=graph, add_absolute=True)
        grades = pd.Series([d["grade_abs"] for _, _, d in ox.get_undirected(graph).edges(data=True)])
        grades = grades.replace([np.inf, -np.inf], np.nan).dropna()
        st.write("道路坡度平均值：{:.1f}%".format(np.mean(grades) * 100))
        st.write("道路坡度中位数：{:.1f}%".format(np.median(grades) * 100))
        nc = ox.plot.get_node_colors_by_attr(graph, "elevation", cmap="plasma")
        ec = ox.plot.get_edge_colors_by_attr(graph, "grade_abs", cmap="plasma", num_bins=4, equal_size=True)
        fig, ax = ox.plot_graph(graph, node_color=nc, node_size=20, edge_linewidth=1, edge_color=ec, bgcolor='white')
        st.pyplot(fig=fig)
        temp.success("根据节点高程及道路坡度渲染网络！")

        dict_layer_gdf['nodes'], dict_layer_gdf['edges'] = ox.graph_to_gdfs(graph)
    return dict_layer_gdf

def gdfs_to_gpkg(dict_layer_gdf, gpkg_path):
    if not os.path.exists(gpkg_path):
        # 批量保存 gdf 数据
        temp = st.info("正在存储地理数据...")
        download_prog = st.progress(0); i=0
        for layer, gdf in dict_layer_gdf.items():
            temp.info(layer + ' saving...')
            # 为了成功保存，转换数据类型为字符串
            gdf = gdf.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
            gdf.to_file(filename=gpkg_path, driver='GPKG', layer=layer)
            i+=20; download_prog.progress(i)  # 模型进度为0-100
        temp.success("地理数据存储完毕！")

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
            ['parking', 'pois']
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
                )
            elif layer == 'pois':
                df_pois = leafmap.gdf_to_df(dict_layer_gdf['pois']).copy()
                df_pois['longitude'] = dict_layer_gdf['pois']['geometry'].x
                df_pois['latitude'] = dict_layer_gdf['pois']['geometry'].y
                df_pois['value'] = 1  # 创建热力图的值字段
                radius = row1_col2.slider('请选择热力图范围：', 5, 30, 15)
                m.add_heatmap(df_pois, value="value", radius=radius, name='pois')
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
                        fields=['name','edge_centrality','length','speed_kph','travel_time','grade'], 
                        aliases=['道路名称','道路中心度','道路长度(m)','行驶速度(km/h)','行驶时间(s)','平均坡度(%)']),
                    hover_style={"fillColor": "#ffaf00", "color": "green", "weight": 3},
                    zoom_to_layer=False
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
                    zoom_to_layer=False
                )
        m.to_streamlit(width=900, height=500)