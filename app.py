import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION & ASSETS ---
st.set_page_config(page_title="YouTube Monetization Modeler", layout="wide")

@st.cache_resource
def load_assets():
    """Load model and encoders once to save memory."""
    try:
        model = joblib.load('final_model.pkl')
        encoders = joblib.load('label_encoders.pkl')
        return model, encoders
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, encoders = load_assets()

# --- 2. UI HEADER & MODEL PERFORMANCE ---
st.title("🎥 YouTube Monetization Modeler")
st.markdown("Precision Revenue Forecasting | **Social Media Analytics**")

# Technical Metrics Section for Evaluation
st.write("---")
m1, m2, m3 = st.columns(3)
m1.metric("Model Accuracy (R²)", "94.97%")
m2.metric("Mean Absolute Error", "$3.60")
m3.metric("Model Type", "Random Forest")
st.write("---")

# --- 3. SIDEBAR INPUTS ---
with st.sidebar:
    predict_btn = st.button("🚀 Calculate Forecast", use_container_width=True)
    st.divider()
    
    st.header("📥 Input Video Metrics")
    views = st.number_input("Views", min_value=1, value=50000)
    likes = st.number_input("Likes", min_value=0, value=2500)
    comments = st.number_input("Comments", min_value=0, value=150)
    watch_time = st.number_input("Watch Time (min)", min_value=0.0, value=120000.0)
    video_len = st.number_input("Video Length (min)", min_value=0.0, value=12.5)
    subs = st.number_input("Subscribers", min_value=0, value=10000)
    
    st.divider()
    
    # Categorical Dropdowns
    category = st.selectbox("Category", encoders['category'].classes_)
    device = st.selectbox("Device", encoders['device'].classes_)
    country = st.selectbox("Country", encoders['country'].classes_)

# Feature Engineering
eng_rate = (likes + comments) / (views + 1)

# --- 4. PREDICTION LOGIC & DASHBOARD ---
if predict_btn:
    try:
        # Transform categories and force into single integers
        cat_id = int(encoders['category'].transform([category])[0])
        dev_id = int(encoders['device'].transform([device])[0])
        cou_id = int(encoders['country'].transform([country])[0])
        
        # Build features in the EXACT order defined in train_model.py
        # [views, likes, comments, watch_time_minutes, video_length_minutes, subscribers, category, device, country, engagement_rate]
        input_data = np.array([[
            float(views), 
            float(likes), 
            float(comments), 
            float(watch_time), 
            float(video_len), 
            float(subs), 
            cat_id, 
            dev_id, 
            cou_id, 
            float(eng_rate)
        ]])

        # Predict and extract result safely to avoid scalar error
        prediction = model.predict(input_data)
        revenue = max(0.0, float(prediction[0]))

        # --- 5. RESULTS KPI ---
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Predicted Revenue", f"${revenue:,.2f}")
        kpi2.metric("Engagement Rate", f"{eng_rate:.2%}")
        cpm = (revenue / views * 1000) if views > 0 else 0
        kpi3.metric("Estimated CPM", f"${cpm:.2f}")

        st.write("---")
        
        # --- 6. STRATEGIC INSIGHTS ---
        st.header("📊 Strategic Insights")
        t1, t2, t3, t4 = st.tabs(["Strategy", "Forecast", "Support", "Ad ROI"])
        
        with t1:
            st.subheader("💡 Content Strategy Optimization")
            st.info(f"Targeting **{category}** in **{country}**.")
            st.write(f"Estimated revenue shift with +10% views: **${(revenue * 0.1):.2f}**")
            
        with t2:
            st.subheader("📅 Revenue Forecasting")
            freq = st.select_slider("Videos Per Month", options=[1, 4, 8, 15, 30], value=4)
            st.success(f"Projected Monthly Revenue: **${(revenue * freq):,.2f}**")
            
        with t3:
            st.subheader("🛠️ Creator Support Tools")
            if eng_rate < 0.02:
                st.warning("Trend: Low engagement. Consider optimizing your 'Call to Action'.")
            else:
                st.balloons()
                st.success("Trend: High interaction depth detected!")
                
        with t4:
            st.subheader("🎯 Ad Campaign Planning")
            reach = int(1000 / (revenue/views) if revenue > 0.01 else 0)
            st.write(f"Estimated Reach per $1,000 Ad Spend: **{reach:,} views**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
else:
    st.info("👈 Enter metrics in the sidebar and click **Calculate Forecast** to see results.")

# --- 7. DATA ANALYSIS ---
st.write("---")
st.header("📈 Data Trends & Analysis")
col_img1, col_img2 = st.columns(2)
with col_img1:
    try:
        st.image("correlation_heatmap.png", caption="Feature Correlations", use_container_width=True)
    except:
        st.caption("Heatmap image not found.")
with col_img2:
    try:
        st.image("category_insights.png", caption="Revenue by Category", use_container_width=True)
    except:
        st.caption("Category chart not found.")
