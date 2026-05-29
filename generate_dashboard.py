import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 设置日期范围 
end_date = datetime.now().date()
start_date = end_date - timedelta(days=730)

# 2. Open-Meteo API 链接（返回 CSV）
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude=31.2304&longitude=121.4737&"
    f"start_date={start_date}&end_date={end_date}&"
    f"daily=windspeed_10m_max,precipitation_sum,temperature_2m_max,temperature_2m_min,weather_code,relative_humidity_2m_mean&"
    f"timezone=Asia/Shanghai&format=csv"
)

# 3. 关键步骤：添加 User-Agent，提高国内访问稳定性
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# 4. 发送请求
try:
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # 如果状态码不是200，会抛出HTTPError
    # 正常处理响应
    df = pd.read_csv(StringIO(response.text), skiprows=2)
    df.to_csv('shanghai_weather_last30days.csv', index=False)
    print("✅ 数据已保存为 CSV")
except requests.exceptions.RequestException as e:
    print(f"❌ 请求失败: {e}，尝试使用本地历史数据")
    # 备选：读取之前保存的 CSV
    try:
        df = pd.read_csv('shanghai_weather_last30days.csv')
        print("✅ 已加载本地历史数据")
    except FileNotFoundError:
        print("❌ 没有本地历史数据，程序退出")
        raise
df['time'] = pd.to_datetime(df['time']) #改类型
df.set_index('time', inplace=True) 

df_last30 = df.tail(30).copy()  #展示最后30天
df_last30
df_last30['avg_temp'] = (df_last30['temperature_2m_max (°C)'] + df_last30['temperature_2m_min (°C)']) / 2 #添加avg_temp字段
df_last30['difference_temp']=df_last30['temperature_2m_max (°C)']-df_last30['temperature_2m_min (°C)']  # 添加difference_temp字段

def text(x):   #   判断条件
    if x >0:
     return '升温'
    if x <0:
     return '降温'
df_last30['temp_change'] = (df_last30['avg_temp'] - df_last30['avg_temp'].shift(1)).apply(text)   #  添加temp_change字段

def simplify_weather_code(code):
    """
    将 WMO 天气代码（0-99）简化为易读的中文类别
    """
    if code == 0:
        return '晴'
    elif code in [1, 2, 3]:
        return '多云'
    elif code in [45, 48]:
        return '雾'
    elif code in [51, 53, 55]:
        return '毛毛雨'
    elif code in [56, 57]:
        return '冻毛毛雨'
    elif code in [61, 63, 65]:
        return '雨'
    elif code in [66, 67]:
        return '冻雨'
    elif code in [71, 73, 75]:
        return '雪'
    elif code == 77:
        return '雪粒'
    elif code in [80, 81, 82]:
        return '阵雨'
    elif code in [85, 86]:
        return '阵雪'
    elif code == 95:
        return '雷暴'
    elif code in [96, 99]:
        return '雷暴伴冰雹'
    else:
        return '未知'
df_last30['weather_code (wmo code)']=df_last30['weather_code (wmo code)'].apply(simplify_weather_code)
## 30天常见天气出现频次(饼状图)
import plotly.graph_objects as go
import plotly.express as px
weather_counts = df_last30['weather_code (wmo code)'].value_counts()
colors = px.colors.qualitative.Set3
fig = go.Figure()
fig.add_trace(go.Pie(
    labels=weather_counts.index,
    values=weather_counts.values,
    textinfo='label+percent',      # 显示天气名称和百分比
    textposition='auto',           # 自动放置标签
    hoverinfo='label+value+percent', # 悬停显示详细信息
    marker=dict(colors=colors),
    pull=[0.025, 0.025, 0.025, 0.05]
    # 突出第一个扇区（可选）
))
fig.update_layout(
    width=800, height=450,
    title='过去30天常见天气出现频次',
    legend_title_text='天气类型',
    annotations=[dict(text='总天数: 30天', x=1, y=-0.3, showarrow=False)]
)
fig.write_html('weather_trend.html')