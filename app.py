import datetime
import streamlit as st
from streamlit_lottie import st_lottie  # 加载动态文件
import requests  # 请求网页文件
from PIL import Image  # 加载本地图片
import json  # 加载本地json文件

# 1.定义全局配置
st.set_page_config(
	page_title='seuteer',  # 站点标题
	page_icon=':traffic_light:',  # logo
	layout='wide',  # 页面布局
	initial_sidebar_state='auto'  # 侧边栏
	)
date_time = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)  # 北京时间

st.sidebar.info('Please click navigation bar to select different pages 👆')
st.sidebar.title("About")
st.sidebar.info("""
This web [app](https://parking-visualization-center.streamlit.app/) 
is maintained by **seuteer**. You can follow me on social media:
[GitHub](https://github.com/seuteer) | [CV](https://maifile.cn/est/d2856781071318/pdf)
"""
)
st.sidebar.info(f"Current time {date_time.date()} / {date_time.time()}")


# 2.定义缓存函数
@st.cache_data
def load_lottie(url):
    # 通过url下载lottie中的元素
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def load_css(css_file):
    with open(css_file) as f:
        # unsafe_allow_html=True 表示允许html
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    # lottie_url = 'https://assets5.lottiefiles.com/private_files/lf30_rg5wrsf4.json'
    # lottie_logo = load_lottie(lottie_url)   # 从 url 获取 json 文件，受限于网速
    with open('images/logo.json', 'r') as f:
        lottie_logo = json.load(f)  # 从本地获取，读取速度更快
    img1 = Image.open('images/1.png')
    img2 = Image.open('images/2.png')
    img3 = Image.open('images/3.png')
    img4 = Image.open('images/4.png')
    return lottie_logo, img1, img2, img3, img4


# 3.主页布局
st.header('Parking Occupancy Prediction Visualization Data Center')
lottie_logo, img1, img2, img3, img4 = load_all_data()
load_css('style/style.css')

# Part 1
with st.container():
	st.write('---')
	st.subheader('Overview')

	row1col1, row1col2 = st.columns(2)
	with row1col1:
		st.image(img1)
		st.info("[Geographic Data Visualization](https://parking-visualization-center.streamlit.app/Geographic_Data_Visualization)")
	with row1col2:
		st.image(img2)
		st.info('[Temporal Correlation Analysis](https://parking-visualization-center.streamlit.app/Spatiotemporal_Correlation)')

	row2col1, row2col2 = st.columns(2)
	with row2col1:
		st.image(img3)
		st.info("[Spatial Correlation Analysis](https://parking-visualization-center.streamlit.app/Spatiotemporal_Correlation)")
	with row2col2:
		st.image(img4)
		st.info('[Parking Occupancy Prediction](https://parking-visualization-center.streamlit.app/Parking_Occupancy_Prediction)')

# Part 2
with st.container():
	st.write('---')
	st.header('Contact Me')
	# Documention: https://formsubmit.co/
	contact_form = """
	<form action="https://formsubmit.co/1240124885@qq.com" method="POST">
		<input type="hidden" name="_captcha" value="false">
		<input type="text" name="name" placeholder="Your name" required>
		<input type="email" name="email" placeholder="Your email" required>
		<textarea name="message" placeholder="Your message here" required></textarea>
		<button type="submit">Send</button>
	</form>
	"""
	col_left, col_right = st.columns(2)
	with col_left:
		st.markdown(contact_form, unsafe_allow_html=True)
	with col_right:
		# 每个带有键的小部件都会自动添加到会话状态
		st_lottie(lottie_logo, height=300, key='logo')