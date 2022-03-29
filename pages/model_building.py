# 模型构建

import streamlit as st
from streamlit_option_menu import option_menu

def app():
    st.title('Model Building')
    st.write('---')

    menu = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal")
    menu