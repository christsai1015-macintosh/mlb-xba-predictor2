import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. 載入模型與相關參數
@st.cache_resource
def load_model():
    # 注意：現在載入的是一個字典
    data = joblib.load('xBA_park_model.joblib')
    return data['model'], data['features'], data['teams']

model, all_features, teams = load_model()

# --- 網頁介面設計 ---
st.title("⚾️ MLB 球場校正版 xBA 預測器")
st.write("這是一個進階模型，考慮了擊球初速、仰角、方向以及球場特定因素。")

with st.sidebar:
    st.header("輸入擊球數據")
    launch_speed = st.slider("擊球初速 (Exit Velocity, mph)", 40.0, 125.0, 95.0)
    launch_angle = st.slider("擊球仰角 (Launch Angle, deg)", -90.0, 90.0, 15.0)
    
    # 新增：擊球方向選擇
    st.info("💡 擊球方向：-45° 為左線邊，0° 為中外野，45° 為右線邊")
    spray_angle = st.slider("擊球方向角 (Spray Angle)", -45.0, 45.0, 0.0)
    
    # 選擇球場
    selected_team = st.selectbox("選擇比賽球場 (主隊)", teams)

# --- 預測邏輯 ---
if st.button("計算預期打擊率 (xBA)"):
    # 1. 準備基礎數據
    input_df = pd.DataFrame({
        'launch_speed': [launch_speed],
        'launch_angle': [launch_angle],
        'spray_angle': [spray_angle]
    })

    # 2. 處理球場 One-Hot Encoding
    # 初始化所有球場欄位為 0
    for feat in all_features:
        if feat.startswith('stadium_'):
            input_df[feat] = 0
    
    # 將選擇的球場設為 1
    target_stadium = f'stadium_{selected_team}'
    if target_stadium in input_df.columns:
        input_df[target_stadium] = 1

    # 3. 確保欄位順序與訓練時完全一致 (XGBoost 非常看重順序)
    input_df = input_df[all_features]

    # 4. 進行預測
    # predict_proba 會回傳 [不活安打機率, 安打機率]
    prob = model.predict_proba(input_df)[0][1]

    # --- 顯示結果 ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("預期打擊率 (xBA)", f"{prob:.3f}")
    with col2:
        # 簡單的分級邏輯
        if prob > 0.7:
            st.success("🔥 這是一支強勁安打！")
        elif prob > 0.4:
            st.warning("⚾️ 有機會形成安打。")
        else:
            st.error("📉 高機率出局。")

    st.write(f"模型分析：在 **{selected_team}** 球場，以這個角度和速度擊球的歷史數據顯示...")