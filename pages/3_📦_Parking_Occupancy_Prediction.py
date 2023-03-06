import os
import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import folium
import altair as alt
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
import streamlit.components.v1 as components


def preprocess(data, locations, col):
    train_ratio = 0.8
    SEQ_LEN = 18  # 8:00 - 16:30 的数据长度
    batch_size = 32

    temp = st.info("Divide Features and Labels")
    X = data
    y = data.loc[:, data.columns == col]
    st.write('Feature dimension (time series length, parking lot data): ', X.shape)
    st.write('label_dimension(timeseries_length, 1): ', y.shape)
    Spatialweight = locations[col]
    st.write(col, 'Space weights for the parking lot:', pd.DataFrame(Spatialweight).T)
    X = X.mul(list(Spatialweight))
    temp.success("Divide Features and Labels")

    temp = st.info("Divide training set and test set")
    st.write('Proportion of training set: ', train_ratio)
    Xtrain, Xtest, Ytrain, Ytest = split_dataset(X, y, train_ratio=train_ratio)
    altdata = data.reset_index()
    altdata['index'] = altdata.index
    altdata['train_valid'] = ["Train" if x <=len(Ytrain) else "Vaild" for x in altdata.index]
    line = alt.Chart(altdata).mark_line().encode(
        x='index:Q',
        y=f'{col}:Q',
        color=alt.Color('train_valid:N', legend=None),
    )
    st.altair_chart(line, use_container_width=True)
    temp.success("Divide training set and test set")

    temp = st.info("Construct a time series dataset and batch process it")
    st.write('LSTM sliding window length: ', SEQ_LEN)
    st.write('batch_size: ', batch_size)
    train_dataset, train_labels = create_dataset(Xtrain, Ytrain, seq_len=SEQ_LEN)
    test_dataset, test_labels = create_dataset(Xtest, Ytest, seq_len=SEQ_LEN)
    st.write('Time series feature dimension (training set length, sliding window length, feature dimension): ', train_dataset.shape)
    st.write('Time series label dimension (training set length, label dimension): ', train_labels.shape)
    train_batch_dataset = create_batch_dataset(train_dataset, train_labels, batch_size=batch_size)
    test_batch_dataset = create_batch_dataset(test_dataset, test_labels, train=False, batch_size=batch_size)
    st.write("The purpose of building batch data is to speed up model training.")
    temp.success("Construct a time series dataset and batch process.")
    return train_dataset, train_labels, test_dataset, test_labels, train_batch_dataset, test_batch_dataset

def training(col, train_dataset, train_batch_dataset, test_batch_dataset, epochs=30):
    if os.path.exists(os.path.join('./data/output/models/', col)):
        temp = st.info('Model is loading from cloud...')
        model = tf.keras.models.load_model(os.path.join('./data/output/models/', col))
        temp.success('Model and weights loaded successfully!')
    else:
        temp = st.info('Train an LSTM neural network...')
        model = keras.Sequential([
            keras.layers.LSTM(128, input_shape=train_dataset.shape[-2:], return_sequences=True),
            keras.layers.Dropout(0.5),
            keras.layers.LSTM(64),
            keras.layers.Dense(1)  # 全连接层，输出为1
        ])
        log_dir=f"./data/output/logs/fit/{col}/" + (datetime.datetime.now()+ datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        st.write('Model optimization function: adam')
        st.write('Model loss function: mse')
        st.write('Model training rounds: ', epochs)
        model.compile(optimizer='adam', loss="mse")
        history = model.fit(train_batch_dataset,
            epochs=epochs,
            validation_data=test_batch_dataset,
            callbacks=[tensorboard_callback],
            verbose=0)  # 沉默输出
        model.save(os.path.join('./data/output/models/', col), save_format='h5')
        temp.success('The model is trained!')

def evaluate():
    import ssl
    from pyngrok import ngrok
    import streamlit.components.v1 as components
    ssl._create_default_https_context = ssl._create_unverified_context
    os.system(f'ngrok authtoken {st.secrets["NGROK_TOKEN"]}')
    
    if len(ngrok.get_tunnels()) == 0:                                            
        # 若没有网页，则开启端口生成网页
        if sys.platform.startswith('win'):
            os.system('start tensorboard --logdir ./data/output/logs/fit/ --port 6006')  # start 开启新进程
        elif sys.platform.startswith('linux'):
            os.system('tensorboard --logdir ./data/output/logs/fit/ --port 6006 &')  # & 开启新进程
        http_tunnel = ngrok.connect(addr='6006', proto='http', bind_tls=True)
    if len(ngrok.get_tunnels()) == 1:
        if st.button('Reload website'):
            for i in ngrok.get_tunnels():
                ngrok.disconnect(i.public_url)
            http_tunnel = ngrok.connect(addr='6006', proto='http', bind_tls=True)
        # 若已有网页，则直接获取网页
        st.write('Visit website: ', ngrok.get_tunnels()[0].public_url)
        components.iframe(ngrok.get_tunnels()[0].public_url, height=600, scrolling=True)
    if len(ngrok.get_tunnels()) >= 2:
        # 若网页数量大于2，则清空网页
        st.sidebar.info('TensorBoard failed to start, please refresh the page and try again!')
        for i in ngrok.get_tunnels():
            ngrok.disconnect(i.public_url)

def prediction(col, train_dataset, train_labels, test_dataset, test_labels):
    if not os.path.exists(os.path.join('./data/output/models/', col)):
        st.sidebar.error("Model file not found, please retrain the model!")
    else:
        model = tf.keras.models.load_model(os.path.join('./data/output/models/', col))
        col1, col2 = st.columns(2)
        with col1.expander("train set prediction", expanded=True):
            # 训练集的预测
            train_pred = model.predict(train_dataset)
            plot_predict(train_labels,train_pred)
        with col2.expander("test set prediction", expanded=True):
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

# 创建动态热力图序列
@st.cache_data
def plot_HeatMapWithTime(locations):
    parking_df = pd.read_pickle('./data/output/birmingham_results.pkl')
    SystemCodeNumber = locations['SystemCodeNumber'].unique()
    labels_list = []
    pred_list = []
    for time_id in range(len(parking_df.loc['test_labels'][0])):
        parking_labels_list = []
        parking_pred_list = []
        for parking_id in SystemCodeNumber:
            parking_labels_list.append([
                locations.loc[locations['SystemCodeNumber']==parking_id, 'latitude'].values[0],
                locations.loc[locations['SystemCodeNumber']==parking_id, 'longtitude'].values[0],
                parking_df.loc['test_labels', parking_id][time_id]
            ])
            parking_pred_list.append([
                locations.loc[locations['SystemCodeNumber']==parking_id, 'latitude'].values[0],
                locations.loc[locations['SystemCodeNumber']==parking_id, 'longtitude'].values[0],
                parking_df.loc['test_pred', parking_id][time_id][0]
            ])
        labels_list.append(parking_labels_list)
        pred_list.append(parking_pred_list)
    # 间隔3个取值，达到间隔3帧播放的效果
    labels_list = labels_list[0:len(labels_list):3]
    pred_list = pred_list[0:len(pred_list):3]
    return labels_list, pred_list


def app():
    st.header('Parking Occupancy Prediction')
    st.sidebar.success("Construction of parking occupancy prediction model 👉")

    # 定义全局变量
    data = pd.read_csv(os.path.join('./data/temp/', 'birmingham_time_series.csv'), index_col=0)
    locations = pd.read_csv(os.path.join('./data/temp/', 'birmingham_loc_pro.csv'))

    col = st.selectbox(
        'Please select the parking lot for model training:',
        data.columns
        )

    st.write("---")
    st.subheader("Data preprocessing")
    train_dataset, train_labels, test_dataset, test_labels, train_batch_dataset, test_batch_dataset = preprocess(data, locations, col)

    st.write("---")
    st.subheader("Model training")
    training(col, train_dataset, train_batch_dataset, test_batch_dataset, epochs=30)

    st.write("---")
    st.subheader("Model evaluation")
    evaluate()

    st.write("---")
    st.subheader("Model prediction")
    prediction(col, train_dataset, train_labels, test_dataset, test_labels)

    st.write('---')
    st.subheader('Parking Occupancy Prediction Dynamic Heat Map')
    labels_list, pred_list = plot_HeatMapWithTime(locations=locations)
    lon, lat = locations['longtitude'].mean(), locations['latitude'].mean()
    m = folium.plugins.DualMap(location=(lat, lon), zoom_start=14)
    folium.plugins.HeatMapWithTime(data=labels_list,auto_play=True, radius=60, display_index=False, name='Raw data').add_to(m.m1)
    folium.plugins.HeatMapWithTime(data=pred_list, auto_play=True, radius=60, display_index=False, name='Predict data').add_to(m.m2)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join('./data/output/', 'birmingham.html'))
    map_html = open(os.path.join('./data/output/', 'birmingham.html'),"r",encoding='utf-8').read()
    components.html(html=map_html, height=500)

app()