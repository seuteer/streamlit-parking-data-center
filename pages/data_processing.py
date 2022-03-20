# 数据处理

import streamlit as st
import pandas as pd

def app():
    st.write('# Data Processing')

    # load data
    parking_data, locations = load_data()
    st.write('parking_data', parking_data)
    st.write('locations', locations)

    # remove parking no space
    parking_data_remove, locations_remove = remove_parking_no_space(parking_data, locations)
    st.write('parking_data_remove', parking_data_remove)
    st.write('locations_remove', locations_remove)

    # create OccupancyRate
    parking_data_create = create_or(parking_data_remove)
    st.write('parking_data_create', parking_data_create)
    st.write("bhm_processed.csv saved!")

    # create CorrelationMatrix
    timeSeriesFeatures, locations_create = create_rs(parking_data_create, locations_remove)
    st.write('locations_create', locations_create)
    st.write("locations_processed.csv saved!")
    st.write('timeSeriesFeatures', timeSeriesFeatures)
    st.write("timeSeriesFeatures.csv saved!")


@st.cache
def load_data():
    '''加载数据'''
    # 导入原始数据（停车占有率表+位置经纬度表）
    parking_data = pd.read_csv(st.session_state.data_input + 'birmingham.csv')
    locations = pd.read_csv(st.session_state.data_input + 'bmh_location.csv')
    return parking_data, locations

@st.cache
def remove_parking_no_space(parking_data, locations):
    '''数据清洗'''
    # 筛选出具有空间坐标的停车场数据
    parking_data = parking_data[parking_data['SystemCodeNumber'].isin(locations['SystemCodeNumber'])]
    SystemCodeNumber = list(set(parking_data['SystemCodeNumber']))
    # 删除时间序列过短的停车场
    [SystemCodeNumber.remove(i) for i in ['BHMBRTARC01', 'NIA North']]
    parking_data_remove = parking_data[parking_data['SystemCodeNumber'].isin(SystemCodeNumber)]
    locations_remove = locations[locations['SystemCodeNumber'].isin(SystemCodeNumber)]
    return parking_data_remove, locations_remove

@st.cache
def create_or(parking_data):
    '''创建占有率指标'''
    data = parking_data.copy()  # 防止改变输入parking_data的值
    # 创建Vacant表示停车位余量
    data['Vacant'] = data['Capacity'] - data['Occupancy']
    # 创建OccupancyRate表示停车占有率
    data['OccupancyRate'] = data['Occupancy'] / data['Capacity']
    # 时空去重（同一停车场同一时间只能有一行数据）
    data.drop_duplicates(subset=['SystemCodeNumber','LastUpdated'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['LastUpdated'] = pd.to_datetime(data['LastUpdated'], format="%Y/%m/%d %H:%M")
    data.sort_values(by=['SystemCodeNumber', 'LastUpdated'], inplace=True)
    data.to_csv(st.session_state.data_output + 'bhm_processed.csv', index=False)
    return data

@st.cache
def create_rs(parking_data, locations):
    '''创建时间序列相关性指标'''
    data = parking_data[['LastUpdated', 'SystemCodeNumber', 'OccupancyRate']]
    # 构建每列表示每个停车场时间序列
    data_space = pd.pivot(data, index='LastUpdated', columns='SystemCodeNumber')
    # 删除特殊时间点（只有几个停车场有数据的时间点为特殊时间点）
    data_space.dropna(thresh=15, inplace=True)
    # 填补缺失值（先按列填充差值bfill，再按列用上一步的值填充ffill）
    data_space = data_space.fillna(axis=0,method='bfill').fillna(axis=0,method='ffill')
    # 多层索引转换为单层索引
    data_space.columns = [x[1] for x in data_space.columns.values]
    # 求解相关性矩阵
    rs = data_space.corr()
    # 提取索引用于后续表的连接
    rs['SystemCodeNumber'] = rs.index
    # 将相关性矩阵连接到静态表，用于构建空间特征权重
    locations_processed = locations.merge(rs, on='SystemCodeNumber')
    # 保存空间表，用于构建多变量特征
    data_space.to_csv(st.session_state.data_output + 'timeSeriesFeatures.csv')
    # 保存静态数据
    locations_processed.to_csv(st.session_state.data_output + 'locations_processed.csv', index=False)
    return data_space, locations_processed