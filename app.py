import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SETUP ---
st.title("🎥 YouTube Revenue Predictor")
st.write("Enter your video details to see how much you could earn.")

# --- 2. LOADING (Simple & Direct) ---
# We load these here so the app knows what to do
model = joblib.load('final_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# --- 3. INPUTS (Sidebar) ---
st.sidebar.header("Video Statistics")

# We use simple variables for the inputs
v_views = st.sidebar.number_input("Views", value=50000)
v_likes = st.sidebar.number_input("Likes", value=2500)
v_comments = st.sidebar.number_input("Comments", value=150)
v_watch = st.sidebar.number_input("Watch Time (min)", value=120000.0)
v_length = st.sidebar.number_input("Video Length (min)", value=10.0)
v_subs = st.sidebar.number_input("Subscribers", value=10000)

# Dropdowns
v_cat = st.sidebar.selectbox("Category", encoders['category'].classes_)
v_dev = st.sidebar.selectbox("Device", encoders['device'].classes_)
v_cou = st.sidebar.selectbox("Country", encoders['country'].classes_)

# --- 4. CALCULATION & PREDICTION ---
# This button triggers the update!
if st.sidebar.button("Predict Revenue"):
    
    # Feature Engineering (The Requirement)
    # We add 1 to avoid a "divide by zero" error
    v_eng_rate = (v_likes + v_comments) / (v_views + 1)
    
    # Converting text to numbers (Essential for the model)
    # We add [0] at the end to get the actual number
    cat_num = encoders['category'].transform([v_cat])[0]
    dev_num = encoders['device'].transform([v_dev])[0]
    cou_num = encoders['country'].transform([v_cou])[0]
    
    # Putting everything in a list in the CORRECT order
    # order: views, likes, comments, watch_time, length, subs, cat, dev, cou, eng_rate
    features = [[
        float(v_views), 
        float(v_likes), 
        float(v_comments), 
        float(v_watch), 
        float(v_length), 
        float(v_subs), 
        int(cat_num), 
        int(dev_num), 
        int(cou_num), 
        float(v_eng_rate)
    ]]
    
    # Get prediction
    prediction = model.predict(features)
    final_rev = prediction[0]
    
    # --- 5. DISPLAY RESULTS ---
    st.header(f"💰 Estimated Revenue: ${final_rev:,.2f}")
    
    # Using columns for a cleaner look
    col1, col2 = st.columns(2)
    col1.metric("Engagement Rate", f"{v_eng_rate:.2%}")
    
    # CPM Calculation
    cpm = (final_rev / v_views) * 1000 if v_views > 0 else 0
    col2.metric("Estimated CPM", f"${cpm:.2f}")

    st.write("---")
    st.subheader("Business Advice")
    if v_eng_rate > 0.05:
        st.success("Your engagement is excellent! This content strategy is working.")
    else:
        st.warning("Your engagement is a bit low. Try encouraging more comments.")

else:
    st.info("Adjust the numbers in the sidebar and click the button to see the forecast!")


# --- 6. DATA VISUALIZATIONS (EDA) ---
st.write("---")
st.header("📈 Data Trends & Analysis")

# We create two simple columns for the images
col_left, col_right = st.columns(2)

with col_left:
    st.image("correlation_heatmap.png", caption="Feature Correlations")
    
with col_right:
    st.image("category_insights.png", caption="Revenue by Category")

st.write("Visualizing the strong correlation between Watch Time and Ad Revenue.")
nd.")
