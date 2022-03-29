# 模型构建

import streamlit as st
from streamlit_option_menu import option_menu

def app():
    st.title('Model Building')
    st.write('---')

    menu = option_menu(None, ["Preprocessing", "Training", "Evaluation", 'Prediction'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal")

    if menu == 'Preprocessing':
        st.header("Preprocessing")
        st.subheader("Load data.")
        st.subheader("Transform data.")
        st.subheader("Visualize data.")
    elif menu == 'Training':
        st.header("Training")
    elif menu == 'Evaluation':
        st.header("Evaluation")
    elif menu == 'Prediction':
        st.header("Prediction")
        st.subheader("Provide sample.")
        st.subheader("Assess result.")
