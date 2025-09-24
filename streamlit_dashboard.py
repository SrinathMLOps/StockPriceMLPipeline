#!/usr/bin/env python3
# Streamlit Dashboard for Stock Price Prediction

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Stock Price ML Pipeline",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("🚀 Stock Price ML Pipeline Dashboard")
st.markdown("**MLOps Pipeline with MLflow, FastAPI, and Docker**")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Info", "Monitoring"])

if page == "Prediction":
    st.header("📊 Stock Price Prediction")
    
    # Input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ma_3 = st.number_input("3-Period Moving Average", value=150.0, min_value=0.0, max_value=1000.0)
    
    with col2:
        pct_change = st.number_input("Percent Change (1 Day)", value=0.02, min_value=-1.0, max_value=1.0, format="%.4f")
    
    with col3:
        volume = st.number_input("Trading Volume", value=1000000, min_value=0, max_value=10000000)
    
    # Prediction button
    if st.button("🔮 Make Prediction", type="primary"):
        try:
            # Call API
            response = requests.post(
                "http://localhost:8000/predict",
                params={
                    "ma_3": ma_3,
                    "pct_change_1d": pct_change,
                    "volume": volume
                }
            )
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                
                # Display result
                st.success(f"💰 Predicted Stock Price: **${prediction:.2f}**")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Predicted Price ($)"},
                    gauge = {
                        'axis': {'range': [None, max(500, abs(prediction) * 1.2)]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, abs(prediction) * 0.5], 'color': "lightgray"},
                            {'range': [abs(prediction) * 0.5, abs(prediction)], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': abs(prediction) * 0.9
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ API Error: Could not get prediction")
                
        except Exception as e:
            st.error(f"❌ Connection Error: {e}")
    
    # Sample scenarios
    st.subheader("📋 Try Sample Scenarios")
    
    scenarios = {
        "Normal Market": {"ma_3": 150.0, "pct_change": 0.01, "volume": 1000000},
        "Bull Market": {"ma_3": 200.0, "pct_change": 0.05, "volume": 2000000},
        "Bear Market": {"ma_3": 100.0, "pct_change": -0.03, "volume": 500000},
        "High Volatility": {"ma_3": 175.0, "pct_change": 0.08, "volume": 5000000}
    }
    
    cols = st.columns(len(scenarios))
    for i, (name, params) in enumerate(scenarios.items()):
        with cols[i]:
            if st.button(f"📊 {name}", key=f"scenario_{i}"):
                st.rerun()

elif page == "Model Info":
    st.header("🤖 Model Information")
    
    try:
        # Get model info
        response = requests.get("http://localhost:8000/model/info")
        if response.status_code == 200:
            model_info = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Type", model_info["model_type"])
                st.metric("Model Version", model_info["model_version"])
            
            with col2:
                st.write("**Features:**")
                for feature in model_info["features"]:
                    st.write(f"• {feature}")
        
        # API Health
        health_response = requests.get("http://localhost:8000/")
        if health_response.status_code == 200:
            st.success("✅ API is healthy")
        else:
            st.error("❌ API is not responding")
            
    except Exception as e:
        st.error(f"❌ Could not connect to API: {e}")
    
    # MLflow info
    st.subheader("🔬 MLflow Integration")
    st.info("MLflow UI: http://localhost:5000")
    st.write("- Experiment tracking")
    st.write("- Model registry") 
    st.write("- Model versioning")

elif page == "Monitoring":
    st.header("📈 Model Monitoring")
    
    # Generate sample monitoring data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
    
    # Simulate predictions over time
    np.random.seed(42)
    predictions = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'prediction': predictions,
        'confidence': np.random.uniform(0.8, 0.99, len(dates))
    })
    
    # Time series chart
    fig = px.line(df, x='timestamp', y='prediction', title='Predictions Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Prediction", f"${df['prediction'].mean():.2f}")
    
    with col2:
        st.metric("Std Deviation", f"${df['prediction'].std():.2f}")
    
    with col3:
        st.metric("Min Prediction", f"${df['prediction'].min():.2f}")
    
    with col4:
        st.metric("Max Prediction", f"${df['prediction'].max():.2f}")

# Footer
st.markdown("---")
st.markdown("**🚀 MLOps Pipeline:** FastAPI + MLflow + Docker + Streamlit")