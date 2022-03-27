# 模型构建

import streamlit as st

def app():
    st.title('Model Building')
    import pandas as pd
    import numpy as np
    import altair as alt

    df = pd.DataFrame(
        np.random.randn(200, 3),
        columns=['a', 'b', 'c'])

    c = alt.Chart(df).mark_circle().encode(
        x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c']).properties(
        width=350,
        height=200
    )

    st.altair_chart(c, use_container_width=True)