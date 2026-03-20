import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 頁面設定 ---
st.set_page_config(page_title="MLB xBA Predictor", page_icon="⚾️")

# --- 1. 載入模型函式 (加入快取與報錯處理) ---
@st.cache_resource
def load_model_data():
    model_path = 'xBA_park_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"找不到模型檔案: {model_path}。請確認檔案已上傳至 GitHub。")
        st.stop()
    
    try:
        data = joblib.load(model_path)
        # 確保資料結構正確
        return data['model'], data['features'], data['teams']
    except Exception as e:
        st.error(f"讀取模型失敗: {e}")
        st.stop()

# 載入核心組件
model, all_features, teams = load_model_data()

# --- 2. 側邊欄：使用者輸入 ---
st.title("⚾️ MLB 球場校正版 xBA 預測器")
st.markdown("這是一個進階機器學習模型，考慮了**擊球物理量**與**球場幾何因素**。")

st.sidebar.header("📊 輸入擊球參數")

l_speed = st.sidebar.slider("擊球初速 (mph)", 40.0, 125.0, 95.0, help="Exit Velocity")
l_angle = st.sidebar.slider("擊球仰角 (deg)", -90.0, 90.0, 15.0, help="Launch Angle")
s_angle = st.sidebar.slider("擊球方向角 (deg)", -45.0, 45.0, 0.0, 
                             help="-45° 為左線邊, 0° 為中外野, 45° 為右線邊")

selected_team = st.sidebar.selectbox("選擇比賽球場 (主隊)", teams)

# --- 3. 預測邏輯 ---
def get_prediction(speed, angle, spray, team):
    # 建立基礎特徵 DataFrame
    input_data = pd.DataFrame({
        'launch_speed': [speed],
        'launch_angle': [angle],
        'spray_angle': [spray]
    })

    # 動態處理 One-Hot Encoding (球場特徵)
    # 初始化所有訓練時看過的 stadium_ 欄位為 0
    for feat in all_features:
        if feat.startswith('stadium_'):
            input_data[feat] = 0
    
    # 將目前選中的球場欄位設為 1
    target_col = f'stadium_{team}'
    if target_col in all_features:
        input_data[target_col] = 1
    
    # 重要：確保輸入的欄位順序跟訓練時完全一致！
    input_data = input_data[all_features]
    
    # 預測機率 [0:出局, 1:安打]
    prediction_prob = model.predict_proba(input_data)[0][1]
    return prediction_prob

# --- 4. 顯示結果 ---
if st.button("立即計算 xBA"):
    xba = get_prediction(l_speed, l_angle, s_angle, selected_team)
    
    st.divider()
    
    # 使用大數字顯示結果
    st.metric(label="預期打擊率 (xBA)", value=f"{xba:.3f}")
    
    # 視覺化回饋
    if xba > 0.700:
        st.success("🔥 **核彈級擊球！** 這是一支機率極高的長打。")
    elif xba > 0.400:
        st.info("⚾️ **優質擊球。** 在多數球場都能形成安打。")
    elif xba > 0.150:
        st.warning("⚠️ **普通擊球。** 很高機率取決於守備位置或球場大小。")
    else:
        st.error("📉 **軟弱擊球。** 高機率形成出局。")

    # 補充資訊
    st.caption(f"目前計算係根據 **{selected_team}** 球場的歷史數據校正。")

st.markdown("---")
st.caption("Data Source: MLB Statcast | Model: XGBoost Classifier")
