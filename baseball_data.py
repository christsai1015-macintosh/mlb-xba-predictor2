import pandas as pd
import numpy as np
from pybaseball import statcast
from xgboost import XGBClassifier
import joblib
import os

def calculate_spray_angle(df):
    """
    計算擊球方向角 (Spray Angle)
    公式基於 Statcast 座標系：Home Plate 為 (125.42, 198.27)
    """
    df = df.copy()
    df['spray_angle'] = np.arctan((df['hc_x'] - 125.42) / (198.27 - df['hc_y'])) * 180 / np.pi
    return df

def train_advanced_park_model():
    print("正在抓取 2025 全球季數據 (這可能需要幾分鐘，請耐心等待)...")
    raw_data = statcast('2025-04-01', '2025-09-30')

    cols = ['launch_speed', 'launch_angle', 'hc_x', 'hc_y', 'home_team', 'events']
    df = raw_data.dropna(subset=cols).copy()

    df = calculate_spray_angle(df)

    hit_events = ['single', 'double', 'triple', 'home_run']
    df['is_hit'] = df['events'].apply(lambda x: 1 if x in hit_events else 0)

    stadium_dummies = pd.get_dummies(df['home_team'], prefix='stadium')
    df_final = pd.concat([df, stadium_dummies], axis=1)

    stadium_features = stadium_dummies.columns.tolist()
    all_features = ['launch_speed', 'launch_angle', 'spray_angle'] + stadium_features

    X = df_final[all_features]
    y = df_final['is_hit']

    print(f"正在訓練 XGBoost 模型... (樣本數: {len(df)}, 特徵數: {len(all_features)})")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X, y)

    save_data = {
        'model': model,
        'features': all_features,
        'teams': sorted(df['home_team'].unique().tolist())
    }
    
    joblib.dump(save_data, 'xBA_park_model.joblib')
    print(f"✅ 進階模型已儲存！檔案大小約 {os.path.getsize('xBA_park_model.joblib')//1024} KB")

if __name__ == "__main__":
    train_advanced_park_model()
