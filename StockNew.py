#pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from plotly import graph_objs as go

START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price History and Forecast')

selected_stock = st.text_input("Enter Stock Ticker 👇", "Type here...")

if(st.button('Submit')):
	n_years = st.slider('Years of prediction:', 1, 4)
	period = n_years * 365



@st.cache
def load_data(ticker):
	data = yf.download(ticker, START, TODAY)
	data.reset_index(inplace=True)
	return data

data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

