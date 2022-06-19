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

# 全局配置
st.set_page_config(
	page_title='seuteer',  # 站点标题
	page_icon=':traffic_light:',  # logo
	layout='wide',  # 页面布局
	initial_sidebar_state='auto'  # 侧边栏
	)

# 设置全局变量，首次运行初始化，方便不同程序文件内进行调用
st.session_state.data_input = './data/input/'
st.session_state.data_output = './data/output/'
st.session_state.data_temp = './data/temp/'
st.session_state.date_time=datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)  # 北京时间
sessions = Server.get_current()._session_info_by_id
st.session_state.current_persons = len(sessions)  # 在线人数
st.session_state.info_st = st.sidebar.empty()  # 侧边栏的提示信息
st.session_state.simplified_mode = False  # 简化模式

# 实例化MultiPage对象
app = MultiPage()

# 添加多个页面
app.add_page('主页', home.app)
app.add_page('空间特征分析', data_geographic.app)
app.add_page('时间序列分析', data_parking.app)
app.add_page('停车占有率预测', model_building.app)

# 运行app
app.run()