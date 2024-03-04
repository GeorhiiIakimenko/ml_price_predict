import streamlit as st
from datetime import date
import yfinance
from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Price prediction AI")

stocks = ("GOOG", "AAPL", "BTC", "GC=F")
selected_stock = st.selectbox("Select stock", stocks)

n_days = st.slider("Days of prediction:", 1, 14)
period = n_days


@st.cache_data
def load_data(stock):
    data = yfinance.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data


data = load_data(selected_stock)
st.subheader("Stock data")
st.write(data.tail())

# Получение данных о VIX
vix_data = yfinance.download("^VIX", START, TODAY)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
fig1.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
fig1.layout.update(title_text="Time Series data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

# Fit SARIMA model
model = SARIMAX(data["Close"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit()

forecast_dates = pd.date_range(start=TODAY, periods=period)
forecast = result.predict(start=len(data), end=len(data) + period - 1, typ='levels')

st.subheader("Forecast data")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="Forecast"))
fig2.layout.update(title_text="Forecast", xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)


# Визуализация данных о VIX (индекс волатильности)
st.subheader("VIX (Volatility Index) Data")
st.markdown("Этот график показывает изменения в индексе волатильности (VIX), который используется для измерения волатильности на финансовых рынках.")
fig_vix = go.Figure()
fig_vix.add_trace(go.Scatter(x=vix_data.index, y=vix_data["Close"], mode='lines', name='VIX'))
fig_vix.update_layout(
    title="VIX (Volatility Index)",
    xaxis_title="Date",
    yaxis_title="VIX Value",
    template="plotly_white"  # Используем светлую тему для лучшей читаемости
)
st.plotly_chart(fig_vix)

