# 数据分析

import os
import streamlit as st
import geemap

# 设置环境变量
os.environ["EARTHENGINE_TOKEN"] = st.secrets["EARTHENGINE_TOKEN"]

# 定义全局变量（注意：当前路径是 app.py 所在文件夹的路径）
data_input = './data/input/'
data_output = './data/output/'
data_temp = './data/temp/'


def app():
    st.write('# Data Analysis')

    st.write('## visualize with geemap')
    m = vis_geemap()
    m.to_streamlit()

@st.cache(allow_output_mutation=True)
def vis_geemap():
    m = geemap.Map(center=[52.479215075, -1.9041145545], zoom=14)
    m.add_geojson(os.path.join(data_temp, 'building_footprints.geojson'), layer_name='building footprints')
    m.add_geojson(os.path.join(data_temp, 'pois.geojson'), layer_name='pois')
    m.add_geojson(os.path.join(data_temp, 'parking.json'), layer_name='parking')
    m.add_shapefile(os.path.join(data_temp, 'road_drive/edges.shp'), layer_name='road drive edges')
    return m