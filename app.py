# 页面布局

# 导入第三方库
import streamlit as st
# 从文件夹中导入.py文件
from pages import home, data_processing, data_analysis, model_building


# 定义管理多页面的类，根据选择显示不同的页面
class MultiPage:
	def __init__(self) -> None:
		self.pages = []  # 定义列表存储页面信息

	# 定义添加页面的函数
	def add_page(self, title, func):
		# 每个页面用字典封装title和function
		self.pages.append(
			{
				'title': title,
				'function': func
			}
		)

	# 定义运行函数，当选择某页面时执行该页面
	def run(self):
		# 使用侧边栏下拉框选择页面
		page = st.sidebar.selectbox(
			'Page Navigation',
			self.pages,
			format_func=lambda page: page['title']
		)
		# 获取到页面后，调用function并执行
		page['function']()


# 设置站点标题和LOGO
st.set_page_config(
	page_title='seuteer',  # 站点标题
	page_icon=':traffic_light:',  # logo
	layout='wide',  # 页面布局
	initial_sidebar_state='auto')  # 侧边栏

# 设置页面标题
st.title('Welcome To Parking Prediction System')


# 实例化MultiPage对象
app = MultiPage()

# 添加页面
app.add_page('Home', home.app)
app.add_page('Data Processing', data_processing.app)
app.add_page('Data Analysis', data_analysis.app)
app.add_page('Model Building', model_building.app)


if __name__ == '__main__':
	app.run()