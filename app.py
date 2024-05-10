import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained Keras model for stock price prediction
model = load_model('C:\Projects\MLStock\stcok_prediction.keras')

# Set up the Streamlit app header
st.header('Stock Prediction App')

# Define start and end dates for downloading historical stock data
start = '2003-01-01'
end = '2024-12-30'

# Default stock ticker symbol (can be changed by user input)
stock = 'VALE'

# Prompt user to input a stock ticker symbol using a text input widget
stock = st.text_input('Enter a stock ticker symbol:', stock)

# Download historical stock data using Yahoo Finance API (yfinance)
data = yf.download(stock, start, end)

# Display stock data as a subheader and DataFrame using Streamlit
st.subheader('Stock Data')
st.write(data)

# Split the downloaded data into training (80%) and testing (20%) sets based on closing prices
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Initialize MinMaxScaler for normalizing data to range [0,1]
scaler = MinMaxScaler(feature_range=(0,1))

# Extract the last 100 days of training data for continuity with test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

# Scale the test data using MinMaxScaler fitted on the concatenated data
data_test_scale = scaler.fit_transform(data_test)

# Display stock price vs. 50-day moving average using Matplotlib and Streamlit
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig1)

# Display stock price vs. 50-day and 100-day moving averages using Matplotlib and Streamlit
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Display stock price vs. 100-day and 200-day moving averages using Matplotlib and Streamlit
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Price')
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare test data sequences and targets for prediction using the model
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])  # Sequences of 100 days (features)
    y.append(data_test_scale[i,0])       # Target (next day's closing price)

# Convert sequences and targets to numpy arrays
x, y = np.array(x), np.array(y)

# Make predictions using the pre-trained model
predict = model.predict(x)

# Rescale the predicted and actual prices back to their original scale
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Display original vs. predicted prices using Matplotlib and Streamlit
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)
