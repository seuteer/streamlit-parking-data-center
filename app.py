import datetime
import streamlit as st
from streamlit.server.server import Server
from multiapp import MultiPage
from pages import (
	data_geographic,
	data_parking,
	home,
	model_building,
)

st.set_page_config(
	page_title='seuteer',  # 站点标题
	page_icon=':traffic_light:',  # logo
	layout='wide',  # 页面布局
	initial_sidebar_state='auto')  # 侧边栏

# 初始化全局配置，方便不同程序文件内进行调用
if 'first_visit' not in st.session_state:
	st.session_state.first_visit=True
	# st.snow()  # 第一次访问时todo
else:
	st.session_state.first_visit=False

# 设置全局变量
st.session_state.data_input = './data/input/'
st.session_state.data_output = './data/output/'
st.session_state.data_temp = './data/temp/'
st.session_state.date_time=datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)  # 北京时间
sessions = Server.get_current()._session_info_by_id
st.session_state.current_persons = len(sessions)  # 在线人数
st.session_state.info_st = st.sidebar.empty()  # 侧边栏的提示信息


# 实例化MultiPage对象
app = MultiPage()

# 添加页面
app.add_page('Home', home.app)
app.add_page("Geographic Data", data_geographic.app)
app.add_page('Parking Data', data_parking.app)
app.add_page('Building Model', model_building.app)

# Run
app.run()