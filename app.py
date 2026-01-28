import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from model_utils import PredictionModel # Ensure you have model_utils.py
from langchain_google_genai import ChatGoogleGenerativeAI
import os

st.set_page_config(page_title="AI Stock Analyst", layout="wide")
os.environ["GOOGLE_API_KEY"] = "AIzaSyDn0okka93-MmpehkjsM2mzbZ7R2iF5wr0"

@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load('stock_model.pth', map_location='cpu'))
    model.eval()
    return scaler, model

scaler, model = load_assets()

st.sidebar.header("Task 1: Forecasting")
ticker = st.sidebar.text_input("Enter Ticker", value="AAPL")
days_to_predict = 7

st.title(f"📈 {ticker} Market Intelligence Dashboard")

if st.sidebar.button("Generate Forecast"):
    df = yf.download(ticker, start="2020-01-01")
    
    close_prices = scaler.transform(df[['Close']])
    last_30_days = close_prices[-30:].reshape(1, 30, 1)
    last_30_days_tensor = torch.tensor(last_30_days, dtype=torch.float32)
    
    with torch.no_grad():
        prediction_scaled = model(last_30_days_tensor)
        prediction = scaler.inverse_transform(prediction_scaled.numpy())
    
    st.success(f"Predicted Price for tomorrow: ${prediction[0][0]:.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index[-50:], df['Close'][-50:], label="Actual")
    st.pyplot(fig)

st.divider()
st.subheader("💬 Task 2: AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Why did the stock move today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    response = llm.invoke(f"You are a financial expert. Analyze the ticker {ticker} and answer: {prompt}")
    
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})