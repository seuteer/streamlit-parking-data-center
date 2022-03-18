import streamlit as st


# 定义管理多页面的类，根据选择显示不同的页面
class MultiPage:
	def __init__(self):
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

		st.sidebar.write('正在播放 稻香-周杰伦 :musical_note:')
		st.sidebar.audio('./musics/稻香-周杰伦.mp3', format='audio/mp3')

		# 使用侧边栏下拉框选择页面
		page = st.sidebar.selectbox(
			'Navigation',
			self.pages,
			format_func=lambda page: page['title']
		)

		st.sidebar.title("About")
		st.sidebar.info("""
		This web [app](https://share.streamlit.io/seuteer/streamlit_app_seuteer/main/app.py) 
		is maintained by **seuteer**. You can follow me on social media:
		[GitHub](https://github.com/seuteer) | [Blog](https://seuteer.icu/)
		"""
		)

		st.sidebar.info(f"""
		Current time {st.session_state.date_time.date()} / {st.session_state.date_time.time()}    
		Current online {st.session_state.current_persons} persons
		"""
		)

		# 获取到页面后，调用function并执行
		page['function']()