# 页面布局

# 导入第三方库
import os
import datetime
import streamlit as st
from streamlit.server.server import Server
from streamlit.script_run_context import get_script_run_ctx as get_report_ctx
# 从.py文件导入类
from multiapp import MultiPage
# 从文件夹中导入.py文件
from pages import (
	home,
	data_collection,
	data_processing, 
	data_analysis,
	model_building
)

# 设置站点标题和LOGO
st.set_page_config(
	page_title='seuteer',  # 站点标题
	page_icon=':traffic_light:',  # logo
	layout='wide',  # 页面布局
	initial_sidebar_state='auto')  # 侧边栏

# 初始化全局配置，方便不同程序文件内进行调用
if 'first_visit' not in st.session_state:
	st.session_state.first_visit=True
	st.snow()  # 第一次访问放气球
else:
	st.session_state.first_visit=False

# 设置全局变量：显示当前时间
# datetime.timedelta(hours=8) : Streamlit Cloud的时区是UTC，加8小时即北京时间
# .replace(microsecond=0) : 表示去掉毫秒
st.session_state.date_time=datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)

# 设置全局变量：显示当前在线人数
session_id = get_report_ctx().session_id
sessions = Server.get_current()._session_info_by_id
session_ws = sessions[session_id].ws
st.session_state.current_persons = len(sessions)


# 实例化MultiPage对象
app = MultiPage()

# 添加页面
app.add_page('Home', home.app)
app.add_page("Data Collection", data_collection.app)
app.add_page('Data Processing', data_processing.app)
app.add_page('Data Analysis', data_analysis.app)
app.add_page('Model Building', model_building.app)


if __name__ == '__main__':
	app.run()