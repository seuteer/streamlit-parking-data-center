import streamlit as st
from streamlit_lottie import st_lottie  # 加载动态文件
import requests  # 请求网页文件
from PIL import Image  # 加载本地图片
import json  # 加载本地json文件

def app():
    st.header('Parking Occupancy Prediction Visualization Data Center')

    # 加载缓存资源
    lottie_logo, img1, img2, img3, img4 = load_all_data()
    # 加载非缓存资源
    load_css('style/style.css')
    
    # Part 1
    with st.container():
        st.write('---')
        st.subheader('Overview')

        row1col1, row1col2 = st.columns(2)
        with row1col1:
            st.image(img1)
            st.info("Geographic Data Visualization")
        with row1col2:
            st.image(img2)
            st.info('Temporal Correlation Analysis')

        row2col1, row2col2 = st.columns(2)
        with row2col1:
            st.image(img3)
            st.info("Spatial Correlation Analysis")
        with row2col2:
            st.image(img4)
            st.info('Parking Occupancy Prediction')

    if not st.session_state.simplified_mode:
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

# 通过url下载lottie中的元素
st.cache_data()
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 设置页面显示风格（不能缓存，因为使用了st库：st.markdown）
st.cache_data()
def load_css(css_file):
    with open(css_file) as f:
        # unsafe_allow_html=True 表示允许html
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 利用缓存机制加载资源，最好不要在缓存的函数内使用st库，因为该函数在后续rerun中不会被调用了
# 除非你就是想要仅在第一次调用时实现某些功能，这时你可以把st.写入缓存函数中，并修改缓存以消除警告 @st.cache(suppress_st_warning=True)
@st.cache_data()
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