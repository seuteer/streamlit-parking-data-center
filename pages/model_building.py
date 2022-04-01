# 模型构建

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE


def app():
    st.title('Model Building')
    st.session_state.info_st.success("停车占有率预测模型构建👉")

    # 定义全局变量
    data = pd.read_csv(os.path.join(st.session_state.data_output, 'timeSeriesFeatures.csv'), index_col=0)
    locations = pd.read_csv(os.path.join(st.session_state.data_output, 'locations_processed.csv'))
    col = st.selectbox(
        '请选择模型训练的停车场:',
        data.columns
        )

    st.write("---")
    st.subheader("数据预处理")
    train_dataset, train_labels, test_dataset, test_labels, train_batch_dataset, test_batch_dataset = preprocess(data, locations, col)

    st.write("---")
    st.subheader("模型训练")
    training(col, train_dataset, train_batch_dataset, test_batch_dataset, epochs=30)

    st.write("---")
    st.subheader("模型评估")
    evaluate()

    st.write("---")
    st.subheader("模型预测")
    prediction(col, train_dataset, train_labels, test_dataset, test_labels)


def preprocess(data, locations, col):
    train_ratio = 0.8
    SEQ_LEN = 18  # 8:00 - 16:30 的数据长度
    batch_size = 32

    temp = st.info("划分特征和标签")
    X = data
    y = data.loc[:, data.columns == col]
    st.write('特征维度(时间序列长度, 停车场数据): ', X.shape)
    st.write('标签维度(时间序列长度, 1): ', y.shape)
    Spatialweight = locations[col]
    st.write(col, '停车场的空间权重：', pd.DataFrame(Spatialweight).T)
    X = X.mul(list(Spatialweight))
    temp.success("划分特征和标签")

    temp = st.info("划分训练集和测试集")
    st.write('训练集占比: ', train_ratio)
    Xtrain, Xtest, Ytrain, Ytest = split_dataset(X, y, train_ratio=train_ratio)
    altdata = data.reset_index()
    altdata['index'] = altdata.index
    altdata['train_valid'] = ["Train" if x <=len(Ytrain) else "Vaild" for x in altdata.index]
    line = alt.Chart(altdata).mark_line().encode(
        x='index:Q',
        y=f'{col}:Q',
        color=alt.Color('train_valid:N', legend=None),
    ).interactive()
    st.altair_chart(line, use_container_width=True)
    temp.success("划分训练集和测试集")

    temp = st.info("构造时间序列数据集并进行批处理")
    st.write('LSTM 滑动窗口长度: ', SEQ_LEN)
    st.write('批处理的batch_size: ', batch_size)
    train_dataset, train_labels = create_dataset(Xtrain, Ytrain, seq_len=SEQ_LEN)
    test_dataset, test_labels = create_dataset(Xtest, Ytest, seq_len=SEQ_LEN)
    st.write('时间序列特征维度(训练集长度, 滑动窗口长度, 特征维度): ', train_dataset.shape)
    st.write('时间序列标签维度(训练集长度, 标签维度): ', train_labels.shape)
    train_batch_dataset = create_batch_dataset(train_dataset, train_labels, batch_size=batch_size)
    test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False, batch_size=batch_size)
    st.write("构建批数据的目的是加速模型训练。")
    temp.success("构造时间序列数据集并进行批处理")
    return train_dataset, train_labels, test_dataset, test_labels, train_batch_dataset, test_batch_dataset

def training(col, train_dataset, train_batch_dataset, test_batch_dataset, epochs=30):
    if os.path.exists(os.path.join('./data/output/models/', col)):
        temp = st.info('模型正在从云端加载...')
        model = tf.keras.models.load_model(os.path.join('./data/output/models/', col))
        temp.success('模型及权重已成功加载！')
    else:
        temp = st.info('训练 LSTM 神经网络...')
        model = keras.Sequential([
            keras.layers.LSTM(128, input_shape=train_dataset.shape[-2:], return_sequences=True),
            keras.layers.Dropout(0.5),
            keras.layers.LSTM(64),
            keras.layers.Dense(1)  # 全连接层，输出为1
        ])
        log_dir=f"./data/output/logs/fit/{col}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        st.write('模型优化函数: adam')
        st.write('模型损失函数: mse')
        st.write('模型训练轮次: ', epochs)
        model.compile(optimizer='adam', loss="mse")
        history = model.fit(train_batch_dataset,
            epochs=epochs,
            validation_data=test_batch_dataset,
            callbacks=[tensorboard_callback],
            verbose=0)  # 沉默输出
        model.save(os.path.join('./data/output/models/', col), save_format='h5')
        temp.success('模型训练完毕！')


def evaluate():
    import sys
    import ssl
    from pyngrok import ngrok
    import streamlit.components.v1 as components
    ssl._create_default_https_context = ssl._create_unverified_context
    
    if len(ngrok.get_tunnels()) == 0:
        # 若没有网页，则开启端口生成网页
        if sys.platform.startswith('win'):
            os.system('start tensorboard --logdir ./data/output/logs/fit/ --port 6006')  # start 开启新进程
        elif sys.platform.startswith('linux'):
            os.system(f'ngrok authtoken {st.secrets["NGROK_TOKEN"]}')
            os.system('tensorboard --logdir ./data/output/logs/fit/ --port 6006 &')  # & 开启新进程
        http_tunnel = ngrok.connect(addr='6006', proto='http', bind_tls=True)
    if len(ngrok.get_tunnels()) == 1:
        # 若已有网页，则直接获取网页
        st.write('访问网页: ', ngrok.get_tunnels()[0].public_url)
        components.iframe(ngrok.get_tunnels()[0].public_url, height=600, scrolling=True)
    if len(ngrok.get_tunnels()) >= 2:
        # 若网页数量大于2，则清空网页
        for i in ngrok.get_tunnels():
            ngrok.disconnect(i.public_url)

def prediction(col, train_dataset, train_labels, test_dataset, test_labels):
    if not os.path.exists(os.path.join('./data/output/models/', col)):
        st.session_state.info_st.error("未发现模型文件，请重新训练模型！")
    else:
        model = tf.keras.models.load_model(os.path.join('./data/output/models/', col))
        col1, col2 = st.columns(2)
        with col1.expander("训练集预测", expanded=True):
            # 训练集的预测
            train_pred = model.predict(train_dataset)
            plot_predict(train_labels,train_pred)
        with col2.expander("测试集预测", expanded=True):
            # 测试集的预测
            test_pred = model.predict(test_dataset)
            plot_predict(test_labels,test_pred)

# 划分训练集和测试集
def split_dataset(X, y, train_ratio=0.8):
    '''基于X和y, 切分为train和test
    Params:
        X : 特征数据集
        y : 标签数据集
        train_ratio : 训练集占X的比例
    Returns:
        X_train, X_test, y_train, y_test
    '''
    X_len = len(X) # 特征数据集X的样本数量
    train_data_len = int(X_len * train_ratio) # 训练集的样本数量
    X_train = X[:train_data_len] # 训练集
    y_train = y[:train_data_len] # 训练标签集
    X_test = X[train_data_len:] # 测试集
    y_test = y[train_data_len:] # 测试集标签集
    # 返回值
    return X_train, X_test, y_train, y_test

# 构造时间序列数据集
def create_dataset(X, y, seq_len=10):
    features = []
    targets = []
    for i in range(0, len(X) - seq_len, 1):
        data = X.iloc[i:i+seq_len] # 序列数据（长度为seq_len的向量）
        label = y.iloc[i+seq_len] # 标签数据（长度为1）
        # 保存到features和labels
        features.append(data)
        targets.append(label)
    return np.array(features), np.array(targets)

# 构造批训练数据，用于加速训练（注意区别训练集和测试集）
def create_batch_dataset(X, y, train=True, buffer_size=100, batch_size=32):
    batch_data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y))) # 数据封装，tensor类型
    if train: # 训练集
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else: # 测试集
        return batch_data.batch(batch_size)

# 绘制误差曲线
def plot_predict(Ytest, Ypred):
    r2 = r2_score(Ytest, Ypred)
    rmse = np.sqrt(MSE(Ytest,Ypred))
    st.write('R2:', round(r2,3), ' RMSE:', round(rmse,3))
    fig = plt.figure(figsize=(8,4))
    plt.plot(range(len(Ytest)), Ytest, c='r', alpha=0.8, label='True')
    plt.plot(range(len(Ytest)), Ypred, c='b', label='Pred')
    plt.legend(fontsize=10)
    st.pyplot(fig)
    # return r2,rmse