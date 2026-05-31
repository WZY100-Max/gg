import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
import time
import folium
import plotly
import os
from plotly.subplots import make_subplots



def generate_dashboard(figs, output='index.html', template_path='template.html'):
    """
    figs: 字典，格式 {'section_id': ('标题', plotly_figure), ...}
    """
    if not os.path.exists(template_path):
        print(f"❌ 模板文件 {template_path} 不存在")
        return
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    sections_html = []
    for section_id, (title, fig) in figs.items():
        # 生成完整的图表 HTML（注意 include_plotlyjs='cdn' 确保只加载一次）
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        section = f'''
<section id="{section_id}">
    <h2>{title}</h2>
    <div class="chart">{fig_html}</div>
</section>
'''
        sections_html.append(section)
    
    # 替换占位符
    final_html = template.replace('{{ sections }}', '\n'.join(sections_html))
    final_html = final_html.replace('{{ timestamp }}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"✅ 仪表盘已生成: {output}")

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

print("饼图数据前5行:", df_last30['temperature_2m_max (°C)'].head())
## 30天常见天气出现频次(饼状图)
colors = px.colors.qualitative.Set3
fig_pie = go.Figure()
weather_counts = df_last30['weather_code (wmo code)'].value_counts()
fig_pie.add_trace(go.Pie(
    labels=weather_counts.index,
    values=weather_counts.values,
    textinfo='label+percent',      # 显示天气名称和百分比
    textposition='auto',           # 自动放置标签
    hoverinfo='label+value+percent', # 悬停显示详细信息
    marker=dict(colors=colors),
    pull=[0.025, 0.025, 0.025, 0.05]
    # 突出第一个扇区（可选）
))
fig_pie.update_layout(
    width=800, height=450,
    title='过去30天常见天气出现频次',
    legend_title_text='天气类型',
    annotations=[dict(text='总天数: 30天', x=1, y=-0.3, showarrow=False)]
)

print("温度趋势数据前5行:", df_last30['temperature_2m_max (°C)'].head())
 #趋势
fig_temp = go.Figure()
# 最低温（先添加，作为填充的基线）
fig_temp.add_trace(go.Scatter(
    x=df_last30.index,
    y=df_last30['temperature_2m_min (°C)'],
    mode='lines+markers',
    name='最低温',
    line=dict(color='blue', width=2),
    marker=dict(size=4)
))

# 最高温（添加并填充到前一条曲线）
fig_temp.add_trace(go.Scatter(
    x=df_last30.index,
    y=df_last30['temperature_2m_max (°C)'],
    mode='lines+markers',
    name='最高温',
    line=dict(color='red', width=2),
    marker=dict(size=4),
    fill='tonexty',                      # 填充到前一条曲线（最低温）
    fillcolor='rgba(128, 128, 128, 0.2)' # 半透明灰色，与原图 alpha=0.2 对应
))

# 布局调整
fig_temp.update_layout(
    title='上海过去30天温度走势',
    xaxis_title='日期',
    yaxis_title='温度 (°C)',
    xaxis_tickangle=45,                  # 日期旋转45度
    legend=dict(orientation='h', yanchor='bottom', y=1.02),  # 图例水平放在顶部
    hovermode='x unified'                # 悬停时统一显示两条线的数值
)

##变化
df_last7=df_last30.tail(7).copy()
print("过去7天数据:", df_last7['temperature_2m_max (°C)'].head())

fig_change=go.Figure()
fig_change.add_trace(go.Scatter(
    x=df_last7.index,
    y=df_last7['temperature_2m_min (°C)'],
    mode='lines+markers',
    name='最低温(℃)',
    line=dict(color='blue',width=2),
    #text=[f"{y}℃" for y in df_last6['temperature_2m_min (°C)']],
    textposition='top center',
    textfont=dict(size=10)
))

fig_change.add_trace(go.Scatter(
    x=df_last7.index,
    y=df_last7['temperature_2m_max (°C)'],
    mode='lines+markers',
    name='最高温(℃)',
    line=dict(color='red',width=2),
   # text=[f"{y}℃" for y in df_last6['temperature_2m_max (°C)']],
    textposition='top center',
    textfont=dict(size=10)
))

weather_annotations = []
for i , (idx , row) in enumerate(df_last7.iterrows()):
    fig_change.add_annotation(
        x=idx,
        y=row['temperature_2m_max (°C)']-2.5,
        text=row['weather_code (wmo code)'],
        showarrow=False,
        font=dict(size=10,color='green'),
        xanchor='center',
        yanchor='top'
    )

fig_change.update_layout(
    title=f"上海过去一周温度变化:最高温{df_last7['temperature_2m_max (°C)'].max()}℃,最低温{df_last7['temperature_2m_min (°C)'].min()}℃",
    xaxis_title='日期',
    yaxis_title='温度(℃)',
    xaxis_tickangle=45,
    legend=dict(orientation='h',yanchor='bottom',y=1.02),
    hovermode='x unified',
    width=1000,
    height=500
)


print("湿度数据:", df_last7['relative_humidity_2m_mean (%)'].head())
# 湿度雨降雨（修正版）
fig_rain = make_subplots(specs=[[{"secondary_y": True}]])
def rain_color(rain_mm):
    if rain_mm == 0:
        return 'lightgray'
    elif rain_mm < 10:
        return 'lightblue'
    elif rain_mm < 25:
        return 'deepskyblue'
    else:
        return 'darkblue'

colors = df_last7['precipitation_sum (mm)'].apply(rain_color).tolist()

# 1. 降水量柱状图（放在次轴，数值通常较小，柱子不会与折线重叠视觉）
fig_rain.add_trace(
    go.Bar(
        x=df_last7.index,
        y=df_last7['precipitation_sum (mm)'],
        name='降水量 (mm)',
        marker=dict(color=colors),
        textposition='outside',
        textfont=dict(size=10),
        opacity=0.8   # 稍微透明，避免完全遮挡
    ),
    secondary_y=True   # 次轴（右侧）
)

# 2. 湿度折线图（放在主轴）
fig_rain.add_trace(
    go.Scatter(
        x=df_last7.index,
        y=df_last7['relative_humidity_2m_mean (%)'],
        mode='lines+markers',
        name='平均湿度 (%)',
        line=dict(color='darkorange', width=2),
        marker=dict(size=6, color='darkorange'),    
        textposition='top center',
        textfont=dict(size=10, color='darkred')
    ),
    secondary_y=False   # 主轴（左侧）
)


# 设置坐标轴标题
fig_rain.update_xaxes(title_text="日期")
fig_rain.update_yaxes(title_text="相对湿度 (%)", secondary_y=False, color='darkred')
fig_rain.update_yaxes(title_text="降水量 (mm)", secondary_y=True, color='steelblue')

# 整体布局
fig_rain.update_layout(
    title='过去7天降水量与相对湿度',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    width=1000,
    height=500
)

figs = {
    'weather_pie': ('🍰 过去30天天气分布', fig_pie),
    'temp_trend': ('🌡️ 过去30天温度趋势', fig_temp),
    'weather_day7':('☀️过去7天天气变化',fig_change),
    'humidity_rain':('🌧️湿度与降雨',fig_rain)
    # 继续添加其他图...
}

generate_dashboard(figs, output='index.html')

