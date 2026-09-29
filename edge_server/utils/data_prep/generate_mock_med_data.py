import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_med_dataset():
    base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(base_project_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'med_sensor_inference_test.csv')

    np.random.seed(42)
    rows = []
    base_time = datetime.now() - timedelta(minutes=100)

# 1. Normal (60 行)：心率取 168~178，HRV 取 0~40
    for i in range(60):
        t = base_time + timedelta(seconds=i*30)
        hr = np.random.normal(172, 3)
        hrv = np.random.uniform(0.0, 30.0)
        spo2 = np.random.choice([98, 99, 100])
        rows.append([t.strftime('%Y-%m-%d %H:%M:%S'), round(hr, 1), round(hrv, 2), spo2])

    # 2. Fatigue (25 行)：心率取 75~85，HRV 取 0~2
    for i in range(25):
        t = base_time + timedelta(seconds=(60 + i)*30)
        hr = np.random.normal(80, 3)
        hrv = np.random.uniform(0.0, 1.5)
        spo2 = np.random.choice([95, 96, 97])
        rows.append([t.strftime('%Y-%m-%d %H:%M:%S'), round(hr, 1), round(hrv, 2), spo2])

    # 3. Crise Cardiaque (15 行)：心率取 125~140，HRV 设在 150~220
    for i in range(15):
        t = base_time + timedelta(seconds=(85 + i)*30)
        hr = np.random.normal(132, 4)
        hrv = np.random.normal(180, 20)
        spo2 = np.random.choice([88, 90, 91, 92])
        rows.append([t.strftime('%Y-%m-%d %H:%M:%S'), round(hr, 1), round(abs(hrv), 2), spo2])
    df = pd.DataFrame(rows, columns=['Timestamp', 'HeartRate', 'HRV', 'SpO2'])
    df.to_csv(output_path, index=False)
    print(f"Jeu de données d'inférence médicale généré : {output_path} ({len(df)} lignes)")

if __name__ == "__main__":
    generate_mock_med_dataset()