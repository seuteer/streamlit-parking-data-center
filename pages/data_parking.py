# 停车数据处理

import folium
import streamlit as st
from streamlit_folium import folium_static
import pandas as pd
import datetime

def app():
    st.title('Parking Data')

    # 定义全局变量，表示提示信息的占位符
    global info_st
    info_st = st.empty()

    # load data
    parking_data, locations = load_data()
    with st.expander('parking data and locations'):
        st.write('parking data', parking_data)
        st.write('locations', locations)

    # remove parking no space
    parking_data_remove, locations_remove = remove_parking_no_space(parking_data, locations)
    info_st.info('remove parking no space...')

    # create OccupancyRate
    parking_data_create = create_or(parking_data_remove)
    info_st.info('create OccupancyRate...')

    # create CorrelationMatrix
    timeSeriesFeatures, locations_create = create_rs(parking_data_create, locations_remove)
    info_st.info('create CorrelationMatrix...')
    with st.expander('data processing'):
        st.write('parking data with OccupancyRate', parking_data_create)
        st.write('locations with CorrelationMatrix', locations_create)
        st.write('timeSeriesFeatures', timeSeriesFeatures)

    info_st.success("Done!")

    # Time series data visualization
    st.write('---')
    st.altair_chart(
        plot_altair(parking_data_create, locations_create), 
        use_container_width=True
        )

    # Geospatial Visualization
    st.write('---')
    time_list, time_index = plot_folium(locations_create, timeSeriesFeatures)
    lon, lat = locations_create['longtitude'].mean(), locations_create['latitude'].mean()
    m = folium.Map(location=(lat, lon), zoom_start=14)
    folium.plugins.HeatMapWithTime(data=time_list, index=time_index, auto_play=True, radius=50).add_to(m)
    folium_static(m, width=900, height=600)
   

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

@st.cache(allow_output_mutation=True)
def plot_altair(parking_data, locations):
    import altair as alt

    # 转换长数据
    long_data = pd.merge(parking_data, locations, on='SystemCodeNumber')
    long_data['LastUpdated'] = pd.to_datetime(long_data['LastUpdated'], format="%Y/%m/%d %H:%M")
    long_data['weekday'] = long_data['LastUpdated'].dt.weekday
    long_data['hour'] = long_data['LastUpdated'].dt.hour
    long_data['is_weekends'] = (long_data['weekday'] == 5) | (long_data['weekday'] == 6)
    long_data = long_data.groupby(['SystemCodeNumber', 'weekday', 'hour']).mean()
    long_data = long_data.reset_index()
    long_data['datetime'] = pd.to_datetime(
        (long_data['weekday']+1).astype("str") + ' ' + (long_data['hour']).astype("str"),
        format='%d %H'
        ) - datetime.timedelta(hours=8)  # 设置显示北京时间

    # 定义选择器
    selection = alt.selection(fields=['SystemCodeNumber'], type='single', on='mouseover', nearest=True)
    # 定义颜色配置
    color_scale = alt.Scale(domain=[True, False], range=['#F5B041', '#5DADE2'])
    # 定义全局配置
    base = alt.Chart(long_data).properties(
        width=350,
        height=200
    ).add_selection(selection)
    # 位置散点图
    scatter = base.mark_circle().encode(
        x=alt.X(
            'mean(longtitude)',
            scale=alt.Scale(domain=(long_data['longtitude'].min(), long_data['longtitude'].max()))
        ),
        y=alt.Y(
            'mean(latitude)',
            scale=alt.Scale(domain=(long_data['latitude'].min(), long_data['latitude'].max()))
        ),
        color=alt.condition(
            selection,
            alt.value("lightgray"),
            "mean(OccupancyRate):Q",
            legend=None
        ),
        size=alt.Size('mean(OccupancyRate):Q', legend=None),
        tooltip=['SystemCodeNumber', 'mean(OccupancyRate):Q'],
    )
    # 时间序列图
    sequential = base.mark_line().encode(
        x='datetime:T',
        y=alt.Y('mean(OccupancyRate):Q', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('weekday:N', legend=None),
    ).transform_filter(
        selection
    )
    # 置信区间图
    line = base.mark_line().encode(
        x='hours(datetime):O',
        y=alt.Y('mean(OccupancyRate):Q', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('is_weekends', legend=None, scale=color_scale)
    ).transform_filter(
        selection
    )
    band = base.mark_errorband(extent='ci').encode(
        x='hours(datetime):O',
        y=alt.Y('OccupancyRate:Q', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('is_weekends', legend=None, scale=color_scale)
    ).transform_filter(
        selection
    )
    # 气泡表格图
    table = base.mark_circle().encode(
        x='hours(datetime):O',
        y='day(datetime):O',
        size=alt.Size('mean(OccupancyRate):Q', legend=None),
        color=alt.Color('is_weekends', legend=None, scale=color_scale),
        tooltip='mean(OccupancyRate):Q',
    ).transform_filter(
        selection
    )
    fig = alt.vconcat(
        (scatter | sequential),
        ((band + line) | table)
    )
    return fig

@st.cache
def plot_folium(locations, data_space):
    SystemCodeNumber = locations['SystemCodeNumber'].unique()
    time_list = []
    for time_id in range(len(data_space)):
        parking_list = []
        for parking_id in SystemCodeNumber:
            parking_list.append([
                locations.loc[locations['SystemCodeNumber']==parking_id, 'latitude'].values[0],
                locations.loc[locations['SystemCodeNumber']==parking_id, 'longtitude'].values[0],
                data_space[parking_id][time_id]
            ])
        time_list.append(parking_list)
    data_space = data_space.reset_index()
    time_index = list(data_space['LastUpdated'].astype(dtype="str"))
    return time_list, time_index