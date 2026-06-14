import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm  # 自动定阶
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import os
from plotly.subplots import make_subplots
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_dashboard(figs, output='index.html', template_path='template.html'):
    """
    figs: 字典，格式 {'section_id': (title, figure), ...}
          或 {'section_id': (title, figure, description), ...}
    """
    if not os.path.exists(template_path):
        print(f"❌ 模板文件 {template_path} 不存在")
        return
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    sections_html = []
    for section_id, item in figs.items():
        # 兼容两种格式
        if len(item) == 2:
            title, fig = item
            description = ""
        else:
            title, fig, description = item
        
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # 如果 description 非空，加上一个带样式的段落
        desc_html = f'<p class="description">{description}</p>' if description else ''
        
        section = f'''
<section id="{section_id}">
    <h2>{title}</h2>
    {desc_html}
    <div class="chart">{fig_html}</div>
</section>
'''
        sections_html.append(section)
    
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

df['weather_simple']=df['weather_code (wmo code)'].apply(simplify_weather_code)
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

weather_counts = df_last30['weather_code (wmo code)'].value_counts()
most_common = weather_counts.idxmax()
most_common_cnt = weather_counts.max()
total_days = len(df_last30)
pie_desc = f"过去30天中，{most_common}天气出现了{most_common_cnt}天，占比{most_common_cnt/total_days:.1%}，是最常见的天气类型。"


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

# 2. 30天温度趋势描述
max_temp_30 = df_last30['temperature_2m_max (°C)'].max()
min_temp_30 = df_last30['temperature_2m_min (°C)'].min()
avg_temp_30 = df_last30['avg_temp'].mean()
temp_trend_desc = f"过去30天最高温{max_temp_30:.1f}°C，最低温{min_temp_30:.1f}°C，平均温度{avg_temp_30:.1f}°C。{'气温整体偏高' if avg_temp_30 > 20 else '气温较凉爽'}。"

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
# 3. 过去7天天气变化描述
max_temp_7 = df_last7['temperature_2m_max (°C)'].max()
min_temp_7 = df_last7['temperature_2m_min (°C)'].min()
delta = df_last7['temperature_2m_max (°C)'].iloc[-1] - df_last7['temperature_2m_max (°C)'].iloc[0]
trend_word = "升温" if delta > 0 else "降温" if delta < 0 else "持平"
week_desc = f"过去一周，最高温达到{max_temp_7:.1f}°C，最低温{min_temp_7:.1f}°C。相比一周前，整体{trend_word}{abs(delta):.1f}°C。"


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

# 4. 湿度与降雨描述
total_rain = df_last7['precipitation_sum (mm)'].sum()
max_rain = df_last7['precipitation_sum (mm)'].max()
rain_desc = f"过去7天总降水量{total_rain:.1f}mm，最大日降水量{max_rain:.1f}mm。{'雨水较多，注意出行' if total_rain > 20 else '降水较少，天气干燥'}。"

# ================== 终极版：最高温预测图（日期准确 + 置信区间可见）==================
# ================== 终极版：最高温预测图（日期准确 + 置信区间可见）==================

import plotly.graph_objects as go
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

forecast_data = df['temperature_2m_max (°C)']

# 1. 确保数据是纯历史（不包含今天及未来）
today = pd.Timestamp.today().normalize()
historical_data = forecast_data[forecast_data.index < today]   # 只取今天之前
if len(historical_data) < 30:
    # 如果不足30天，就取全部
    history = historical_data
else:
    history = historical_data.tail(30)

print(f"历史数据范围: {history.index.min()} 至 {history.index.max()}")
print(f"历史数据最后一天: {history.index[-1]}")

# 2. 自动选择 ARIMA 模型（使用全部历史数据）
auto_model = auto_arima(
    forecast_data,   # 仍然用全量数据训练（不含未来）
    seasonal=True, m=7,
    trace=True, stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    n_jobs=-1,
    information_criterion='aic'
)
print(f"最佳阶数: {auto_model.order}, 季节阶数: {auto_model.seasonal_order}")

# 3. 预测未来7天
forecast_steps = 7
forecast, conf_int = auto_model.predict(n_periods=forecast_steps, return_conf_int=True)

# 4. 生成未来日期（从明天开始）
first_forecast_date = history.index[-1] + pd.Timedelta(days=1)   # 明天
future_dates = pd.date_range(start=first_forecast_date, periods=forecast_steps)
print(f"预测日期范围: {future_dates[0]} 至 {future_dates[-1]}")

# 5. 绘图
fig_forecast = go.Figure()

# 历史温度（蓝色实线）
fig_forecast.add_trace(go.Scatter(
    x=history.index, y=history.values,
    mode='lines+markers', name='历史温度 (实际观测)',
    line=dict(color='red', width=2), marker=dict(size=4)
))

# 预测温度（红色虚线）
fig_forecast.add_trace(go.Scatter(
    x=future_dates, y=forecast,
    mode='lines+markers', name='高温预测',
    line=dict(color='red', width=2, dash='dash'), marker=dict(size=4)
))

# 置信区间：使用不可见边界线 + 填充（保证填充可见）
# 上边界（完全透明，但线宽设为0.1以避免渲染bug）
fig_forecast.add_trace(go.Scatter(
    x=future_dates, y=conf_int[:, 1],
    mode='lines',
    line=dict(color='rgba(0,0,0,0)', width=0.1),   # 极细透明线
    showlegend=False,
    hoverinfo='skip'
))
# 下边界，并填充到上边界
fig_forecast.add_trace(go.Scatter(
    x=future_dates, y=conf_int[:, 0],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(255, 192, 203, 0.5)',
    line=dict(color='rgba(0,0,0,0)', width=0.1),
    name='95% 置信区间',
    hoverinfo='skip'
))


import plotly.graph_objects as go
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

forecast_data1 = df['temperature_2m_min (°C)']

# 1. 确保数据是纯历史（不包含今天及未来）
today1 = pd.Timestamp.today().normalize()
historical_data1 = forecast_data1[forecast_data1.index < today]   # 只取今天之前
if len(historical_data1) < 30:
    # 如果不足30天，就取全部
    history1 = historical_data1
else:
    history1 = historical_data1.tail(30)

print(f"历史数据范围: {history1.index.min()} 至 {history1.index.max()}")
print(f"历史数据最后一天: {history1.index[-1]}")

# 2. 自动选择 ARIMA 模型（使用全部历史数据）
auto_model1 = auto_arima(
    forecast_data1,   # 仍然用全量数据训练（不含未来）
    seasonal=True, m=7,
    trace=True, stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    n_jobs=-1,
    information_criterion='aic'
)
print(f"最佳阶数: {auto_model1.order}, 季节阶数: {auto_model1.seasonal_order}")

# 3. 预测未来7天
forecast_steps = 7
forecast1, conf_int1 = auto_model1.predict(n_periods=forecast_steps, return_conf_int=True)

# 4. 生成未来日期（从明天开始）
first_forecast_date1 = history1.index[-1] + pd.Timedelta(days=1)   # 明天
future_dates1 = pd.date_range(start=first_forecast_date1, periods=forecast_steps)
print(f"预测日期范围: {future_dates1[0]} 至 {future_dates1[-1]}")

# 5. 绘图


# 历史温度（蓝色实线）
fig_forecast.add_trace(go.Scatter(
    x=history1.index, y=history1.values,
    mode='lines+markers', name='历史温度 (实际观测)',
    line=dict(color='blue', width=2), marker=dict(size=4)
))

# 预测温度（红色虚线）
fig_forecast.add_trace(go.Scatter(
    x=future_dates1, y=forecast1,
    mode='lines+markers', name='低温预测',
    line=dict(color='blue', width=2, dash='dash'), marker=dict(size=4)
))

# 置信区间：使用不可见边界线 + 填充（保证填充可见）
# 上边界（完全透明，但线宽设为0.1以避免渲染bug）
fig_forecast.add_trace(go.Scatter(
    x=future_dates1, y=conf_int1[:, 1],
    mode='lines',
    line=dict(color='rgba(0,0,0,0)', width=0.1),   # 极细透明线
    showlegend=False,
    hoverinfo='skip'
))
# 下边界，并填充到上边界
fig_forecast.add_trace(go.Scatter(
    x=future_dates1, y=conf_int1[:, 0],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(0,0,255,0.3)',
    line=dict(color='rgba(0,0,0,0)', width=0.1),
    name='95% 置信区间',
    hoverinfo='skip'
))


# 布局
fig_forecast.update_layout(
    title='最近30天最高气温及未来7天预测',
    xaxis_title='日期', yaxis_title='温度 (°C)',
    hovermode='x unified', width=1000, height=500
)



# 预测描述
if hasattr(forecast, 'iloc'):
    trend = "上升" if forecast.iloc[-1] > forecast.iloc[0] else "下降"
    temp_min = forecast.min()
    temp_max = forecast.max()
else:
    trend = "上升" if forecast[-1] > forecast[0] else "下降"
    temp_min = np.min(forecast)
    temp_max = np.max(forecast)


# 预测描述
if hasattr(forecast, 'iloc'):
    trend1 = "上升" if forecast1.iloc[-1] > forecast1.iloc[0] else "下降"
    temp_min1 = forecast1.min()
    temp_max1 = forecast1.max()
else:
    trend1 = "上升" if forecast[-1] > forecast[0] else "下降"
    temp_min1 = np.min(forecast1)
    temp_max1 = np.max(forecast1)
forecast_desc = f"预计未来7天最高气温在{temp_min:.1f}°C ~ {temp_max:.1f}°C之间，整体呈{trend}趋势;预计未来7天最低气温在{temp_min1:.1f}°C ~ {temp_max1:.1f}°C之间，整体呈{trend1}趋势"


df['weather_晴'] = (df['weather_simple'] == '晴').astype(int)
df['weather_雨'] = (df['weather_simple'] == '雨').astype(int)
df['temp_lag1'] = df['temperature_2m_max (°C)'].shift(1)
final_exog_cols = ['windspeed_10m_max (km/h)', 'precipitation_sum (mm)', 
                   'weather_晴', 'weather_雨', 'temp_lag1']

X = df[final_exog_cols].dropna()
y = df['relative_humidity_2m_mean (%)'].loc[X.index]

# 训练最终模型
final_model = SARIMAX(y, exog=X, order=(2,1,1), seasonal_order=(0,0,0,0))
fitted_final = final_model.fit(disp=False)

def get_future_exog(lat, lon, today_temp):
    """
    lat, lon: 经纬度
    today_temp: 今天的实际最高温（用于预测明天的湿度）
    返回 DataFrame，包含未来7天的外生变量，索引为未来日期。
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"daily=windspeed_10m_max,precipitation_sum,weather_code,temperature_2m_max&"
        f"timezone=Asia/Shanghai"
    )
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=30)
    data = resp.json()
    dates = pd.to_datetime(data['daily']['time'])
    df_fut = pd.DataFrame({
        'windspeed': data['daily']['windspeed_10m_max'],
        'precip': data['daily']['precipitation_sum'],
        'weather_code': data['daily']['weather_code'],
        'temp_max': data['daily']['temperature_2m_max']   # 未来每天的最高温
    }, index=dates)

    # 简化天气代码 -> 生成 weather_晴, weather_雨
    def simpl(code):
        if code == 0: return '晴'
        elif code in [1,2,3]: return '多云'
        elif code in [51,53,55,61,63,65,80,81,82]: return '雨'
        else: return '其他'
    df_fut['weather_simple'] = df_fut['weather_code'].apply(simpl)
    df_fut['weather_晴'] = (df_fut['weather_simple'] == '晴').astype(int)
    df_fut['weather_雨'] = (df_fut['weather_simple'] == '雨').astype(int)

    # 构造 temp_lag1：对于未来第1天，使用 today_temp；对于第2天，使用未来第1天的 temp_max；以此类推
    # 即向后偏移一天
    df_fut['temp_lag1'] = df_fut['temp_max'].shift(1)
    df_fut.iloc[0, df_fut.columns.get_loc('temp_lag1')] = today_temp

    # 选择最终需要的列（顺序必须与训练时一致）
    X_future = df_fut[['windspeed', 'precip', 'weather_晴', 'weather_雨', 'temp_lag1']]
    X_future.columns = final_exog_cols  # 重命名为训练时的列名
    return X_future

today_temp = df['temperature_2m_max (°C)'].iloc[-1]

X_future = get_future_exog(lat=31.2304, lon=121.4737, today_temp=today_temp)
humidity_forecast = fitted_final.forecast(steps=7, exog=X_future)
humidity_conf_int = fitted_final.get_forecast(steps=7, exog=X_future).conf_int()
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

# 绘图
fig_humidity = go.Figure()
history = y.tail(30)
fig_humidity.add_trace(go.Scatter(x=history.index, y=history, mode='lines+markers', name='历史湿度'))
fig_humidity.add_trace(go.Scatter(x=future_dates, y=humidity_forecast, mode='lines+markers', name='预测湿度', line=dict(dash='dash')))
fig_humidity.add_trace(go.Scatter(x=future_dates, y=humidity_conf_int.iloc[:,1], mode='lines', line=dict(width=0), showlegend=False))
fig_humidity.add_trace(go.Scatter(x=future_dates, y=humidity_conf_int.iloc[:,0], mode='lines', name='95%置信区间', fill='tonexty', fillcolor='rgba(255,192,203,0.5)'))
fig_humidity.update_layout(title='未来7天相对湿度预测', xaxis_title='日期', yaxis_title='湿度 (%)', hovermode='x unified')



# 湿度预测描述（包含所有外生变量的影响，不过度依赖p值）
# 外生变量顺序与 final_exog_cols 一致
exog_names = final_exog_cols   # ['windspeed_10m_max (km/h)', 'precipitation_sum (mm)', 'weather_晴', 'weather_雨', 'temp_lag1']
coef = fitted_final.params[exog_names]

# 构建解释文本
effects = []
# 风速（系数为负：风速越大湿度越低）
if coef['windspeed_10m_max (km/h)'] < 0:
    effects.append(f"风速越大，湿度倾向于越低（每增加1km/h约下降{abs(coef['windspeed_10m_max (km/h)']):.2f}%）")
else:
    effects.append(f"风速越大，湿度倾向于越高（每增加1km/h约上升{coef['windspeed_10m_max (km/h)']:.2f}%）")

# 降水量
effects.append(f"降水会使湿度明显上升（每1mm降水约增加{coef['precipitation_sum (mm)']:.2f}%）")

# 晴天
effects.append(f"晴天比非晴天平均低{abs(coef['weather_晴']):.1f}%")

# 雨天
effects.append(f"雨天比非雨天平均高{coef['weather_雨']:.1f}%")

# 昨日最高温
effects.append(f"昨日最高温每升高1°C，今日湿度约上升{coef['temp_lag1']:.2f}%")

exog_desc = "；".join(effects) + "。"

# 预测范围
hum_min = humidity_forecast.min()
hum_max = humidity_forecast.max()
hum_mean = humidity_forecast.mean()

humidity_desc = f"未来7天相对湿度预计在{hum_min:.0f}%~{hum_max:.0f}%之间，平均约{hum_mean:.0f}%。根据历史数据规律，{exog_desc}"



figs = {
    'weather_pie': ('🍰 过去30天天气分布', fig_pie, pie_desc),
    'temp_trend': ('🌡️ 过去30天温度趋势', fig_temp, temp_trend_desc),
    'weather_day7': ('☀️ 过去7天天气变化', fig_change, week_desc),
    'humidity_rain': ('🌧️ 湿度与降雨', fig_rain, rain_desc),
    'weather_forecast': ('🔮 未来天气预测', fig_forecast, forecast_desc),
    'humidity_forecast': ('🌬️ 未来湿度预测', fig_humidity, humidity_desc)
}

generate_dashboard(figs, output='index.html')
