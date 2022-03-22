# 主页

import streamlit as st
from streamlit_lottie import st_lottie
import requests  # 获取网页文件
from PIL import Image  # 加载本地图片
import json  # 加载本地json文件


def app():
    st.title('Welcome To Parking Prediction System')

    # 加载缓存资源
    lottie_logo,img_bg_gd,img_bg_qyqx,img_bg_welt = load_all_data()
    # 加载非缓存资源
    load_css('style/style.css')
    
    # Part 1
    with st.container():
        st.write("---")
        # 分两列
        col_left, col_right = st.columns(2)
        with col_left:
            st.header("About Me")
            st.info("""
            Hi, I am a student in SEU.  
            [This is my Github](https://github.com/seuteer)  
            [Learn More About Me...](https://seuteer.icu/)  
            """)
        with col_right:
            # 每个带有键的小部件都会自动添加到会话状态
            st_lottie(lottie_logo, height=300, key='logo')

    # Part 2
    with st.container():
        st.write('---')
        # 鬼刀
        st.header('Background')
        col_img, col_text = st.columns((1,2))  # 自定义占比(1,2)
        with col_img:
            st.image(img_bg_gd)
        with col_text:
            st.info('鬼刀')
        # 千与千寻
        col_img, col_text = st.columns((1,2))
        with col_img:
            st.image(img_bg_qyqx)
        with col_text:
            st.info('千与千寻')
        # 薇尔莉特
        col_img, col_text = st.columns((1,2))
        with col_img:
            st.image(img_bg_welt)
        with col_text:
            st.info('薇尔莉特')

    # Part 3
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
            st.info("""
            您可以通过填写表单给我发邮箱信息！  
            期待与您的联系！
            """)


# 通过url下载lottie中的元素
st.cache
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 设置页面显示风格（不能缓存，因为使用了st库：st.markdown）
st.cache
def load_css(css_file):
    with open(css_file) as f:
        # unsafe_allow_html=True 表示允许html
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 利用缓存机制加载资源，最好不要在缓存的函数内使用st库，因为该函数在后续rerun中不会被调用了
# 除非你就是想要仅在第一次调用时实现某些功能，这时你可以把st.写入缓存函数中，并修改缓存以消除警告 @st.cache(suppress_st_warning=True)
@st.cache
def load_all_data():
    # lottie_url = 'https://assets5.lottiefiles.com/private_files/lf30_rg5wrsf4.json'
    # lottie_logo = load_lottie(lottie_url)   # 从 url 获取 json 文件，受限于网速
    with open('images/logo.json', 'r') as f:
        lottie_logo = json.load(f)  # 从本地获取，读取速度更快
    img_bg_gd = Image.open('images/bg_鬼刀.jpg')
    img_bg_qyqx = Image.open('images/bg_千与千寻.jpg')
    img_bg_welt = Image.open('images/bg_薇尔莉特.jpg')
    return lottie_logo,img_bg_gd,img_bg_qyqx,img_bg_welt