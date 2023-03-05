import datetime
import streamlit as st
from multiapp import MultiPage
from pages import (
	analysis,
	data,
	home,
	model,
)

# 定义全局配置
st.set_page_config(
	page_title='seuteer',  # 站点标题
	page_icon=':parking:',  # logo
	layout='wide',  # 页面布局
	initial_sidebar_state='auto'  # 侧边栏
	)

# 定义全局变量，首次运行初始化，方便不同程序文件内进行调用
st.session_state.data_input = './data/input/'
st.session_state.data_output = './data/output/'
st.session_state.data_temp = './data/temp/'
st.session_state.date_time = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)  # 北京时间
st.session_state.info_st = st.sidebar.empty()  # 侧边栏的提示信息
st.session_state.simplified_mode = False  # 简化模式

# 实例化MultiPage对象
app = MultiPage()

# 分页
app.add_page('Parking Visualization Center', home.app)
app.add_page('Geographic Data Visualization', data.app)
app.add_page('Spatiotemporal Correlation', analysis.app)
app.add_page('Parking Occupancy Prediction', model.app)

# 运行
app.run()