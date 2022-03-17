# 页面布局

# 导入第三方库
import streamlit as st
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


# 实例化MultiPage对象
app = MultiPage()

# 添加页面
app.add_page('Home', home.app)
app.add_page('Data Processing', data_processing.app)
app.add_page('Data Analysis', data_analysis.app)
app.add_page('Model Building', model_building.app)


if __name__ == '__main__':
	app.run()