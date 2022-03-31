# 停车数据处理

import folium
import streamlit as st
import pandas as pd
import datetime
import streamlit.components.v1 as components  # 自定义组件显示 folium,altair 的 html

def app():
    st.title('Parking Data')
    st.session_state.info_st.success("停车场时间序列探索与分析👉")

    st.write("---")
    st.subheader("停车场时间序列处理")
    parking_data, locations = load_data()
    temp = st.info("加载停车场时间序列数据...")
    parking_data_remove, locations_remove = remove_parking_no_space(parking_data, locations)
    temp.info('删除无空间属性停车场...')
    parking_data_create = create_or(parking_data_remove)
    temp.info('创建停车占有率指标...')
    timeSeriesFeatures, locations_create = create_rs(parking_data_create, locations_remove)
    temp.info('创建空间自相关指标...')
    temp.success("数据处理完毕！")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander('原始数据：停车场时间序列+停车场空间坐标👇'):
            st.write('停车场数据', parking_data)
            st.write('空间坐标点', locations)
    with col2:
        with st.expander('处理数据：占有率和空间相关性的停车场数据👇'):
            st.write('创建占有率指标', parking_data_create)
            st.write('创建空间相关性指标', locations_create)
            st.write('所有停车场的时间序列数据', timeSeriesFeatures)

    st.write("---")
    st.subheader("时间序列热力图分析")
    temp = st.info("正在绘制时间序列热力图...")
    col1, col2 = st.columns((3,1))
    time_list, time_index = plot_folium(locations_create, timeSeriesFeatures)
    lon, lat = locations_create['longtitude'].mean(), locations_create['latitude'].mean()
    m = folium.Map(location=(lat, lon), zoom_start=14)
    with col2:
        radius = st.slider('请选择热力图范围：', 30, 100, 60)
    folium.plugins.HeatMapWithTime(data=time_list, index=time_index, auto_play=True, radius=radius).add_to(m)
    fig_folium = folium.Figure().add_child(m)
    with col1:
        components.html(html=fig_folium.render(), height=500)  # 宽度自适应
    temp.success("时间序列热力图绘制完毕！")

    st.write("---")
    st.subheader("时间序列空间自相关分析")
    st.info("您可以将鼠标悬浮到停车场附近，交互式分析停车场占有率时间序列的时空关系")
    fig_altair = plot_altair(parking_data_create, locations_create)
    st.altair_chart(fig_altair, use_container_width=True)  # fig_altair 不属于 altair.vegalite.v2.api.Chart 类型，因此没法自适应宽度


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
        x='hour',
        y=alt.Y('mean(OccupancyRate):Q', scale=alt.Scale(domain=[0,1])),
        color=alt.Color('is_weekends', legend=None, scale=color_scale)
    ).transform_filter(
        selection
    )
    band = base.mark_errorband(extent='ci').encode(
        x='hour',
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
        ((band + line) | table),
    )

    return fig

@st.cache(suppress_st_warning=True)
def plot_folium(locations, data_space):
    SystemCodeNumber = locations['SystemCodeNumber'].unique()
    time_list = []
    download_prog = st.progress(0); i=0
    for time_id in range(len(data_space)):
        parking_list = []
        for parking_id in SystemCodeNumber:
            parking_list.append([
                locations.loc[locations['SystemCodeNumber']==parking_id, 'latitude'].values[0],
                locations.loc[locations['SystemCodeNumber']==parking_id, 'longtitude'].values[0],
                data_space[parking_id][time_id]
            ])
        time_list.append(parking_list)
        i+=1/len(data_space); download_prog.progress(i)
    data_space = data_space.reset_index()
    time_index = list(data_space['LastUpdated'].astype(dtype="str"))
    return time_list, time_index