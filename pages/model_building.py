# 模型构建

import streamlit as st
from streamlit_option_menu import option_menu

def app():
    st.title('Model Building')
    st.session_state.info_st.success("停车占有率预测模型构建👉")
    menu = option_menu(None, ["数据预处理", "模型训练", "模型评估", '模型预测'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="horizontal")

    if menu == '数据预处理':
        st.subheader("划分特征和标签")
        st.subheader("划分训练集和测试集")
        st.subheader("构造时间序列数据集")
        st.subheader("构建批训练数据集")
    elif menu == '模型训练':
        st.subheader("构建 LSTM 神经网络")
        st.subheader("定义 Tensorboard 日志")
        st.subheader("编译、训练并保存模型")
    elif menu == '模型评估':
        st.subheader("TensorBoard 面板分析")
    elif menu == '模型预测':
        st.subheader("训练集和测试集的预测")