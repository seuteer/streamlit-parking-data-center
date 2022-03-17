# 页面布局

# 导入第三方库
import streamlit as st
import datetime
# 从.py文件导入类
from multiapp import MultiPage
# 从文件夹中导入.py文件
from pages import (
	home, 
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

# 初始化全局配置，在这里可以定义任意多个全局变量，方便不同程序文件内进行调用
# first_visit记录了是否第一次访问，若不是第一次访问，则为False
if 'first_visit' not in st.session_state:
	st.session_state.first_visit=True
else:
	st.session_state.first_visit=False
if st.session_state.first_visit:
	st.balloons()  # 第一次访问放气球
# datetime.timedelta(hours=8) : Streamlit Cloud的时区是UTC，加8小时即北京时间
# .replace(microsecond=0) : 表示去掉毫秒
st.session_state.date_time=datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)



# 实例化MultiPage对象
app = MultiPage()

# 添加页面
app.add_page('Home', home.app)
app.add_page('Data Processing', data_processing.app)
app.add_page('Data Analysis', data_analysis.app)
app.add_page('Model Building', model_building.app)


if __name__ == '__main__':
	app.run()