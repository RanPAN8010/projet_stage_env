import pandas as pd
import numpy as np
import os
# dataset2 https://huggingface.co/datasets/kopetri/AutoTherm
# dataset1 https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset

#pour calculer l'index de chaleur
def calculate_heat_index(T_celsius, RH):
    T_f = T_celsius * 1.8 + 32
    HI_f = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))
    mask = T_f >= 80
    if mask.any():
        T = T_f; R = RH
        HI_full = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 
                   6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R + 
                   8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2)
        inner_value = (17 - np.abs(T - 91.)) / 14
        adj_dry = ((13 - R) / 4) * np.sqrt(np.clip(inner_value, 0, None))
        adj_humid = ((R - 85) / 10) * ((87 - T) / 5)
        HI_full = np.where((R < 13) & (T >= 80) & (T <= 112), HI_full - adj_dry, HI_full)
        HI_full = np.where((R > 85) & (T >= 80) & (T <= 87), HI_full + adj_humid, HI_full)
        HI_f = np.where(mask, HI_full, HI_f)
    return (HI_f - 32) / 1.8

#pour générer le dataset à parti de dataset2
def process_car_data(file_path):
    df = pd.read_parquet(file_path).copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by=['file_name', 'Timestamp']).reset_index(drop=True)
    
    # Caractéristiques de variation brute pas à pas
    df['Temp_Rate'] = df['Ambient_Temperature'].diff()
    df['Humidity_Rate'] = df['Ambient_Humidity'].diff()
    
    is_new_vehicle_boundary = df['file_name'] != df['file_name'].shift(1)
    df['Temp_Rate'] = np.where(is_new_vehicle_boundary, 0.0, df['Temp_Rate'])
    df['Humidity_Rate'] = np.where(is_new_vehicle_boundary, 0.0, df['Humidity_Rate'])
    
    df['Temp_Rate'] = df['Temp_Rate'].fillna(0.0)
    df['Humidity_Rate'] = df['Humidity_Rate'].fillna(0.0)
    df['Heat_Index'] = calculate_heat_index(df['Ambient_Temperature'], df['Ambient_Humidity'])
    
    df['Label'] = np.where(df['Ambient_Temperature'] >= 30, 1, 0)
    df = df.rename(columns={'Ambient_Temperature': 'Temperature', 'Ambient_Humidity': 'Humidity'})
    
    cols = ['Timestamp', 'Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index', 'Label']
    return df[cols]

# pour générer le dataset à partir de dataset1
def process_smoke_data(file_path):
    df = pd.read_csv(file_path).copy()
    df = df[df['Fire Alarm'] == 1].copy()
    if df.empty:
        return pd.DataFrame()
        
    df['Timestamp'] = pd.to_datetime(df['UTC'], unit='s')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    df['Temp_Rate'] = df['Temperature[C]'].diff()
    df['Humidity_Rate'] = df['Humidity[%]'].diff()
    df['Temp_Rate'] = df['Temp_Rate'].fillna(0.0)
    df['Humidity_Rate'] = df['Humidity_Rate'].fillna(0.0)
    
    df['Heat_Index'] = calculate_heat_index(df['Temperature[C]'], df['Humidity[%]'])
    df['Label'] = 2
    df = df.rename(columns={'Temperature[C]': 'Temperature', 'Humidity[%]': 'Humidity'})
    
    real_fire_mask = (df['Temp_Rate'] > 0.0)
    df = df[real_fire_mask].copy()
    
    cols = ['Timestamp', 'Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index', 'Label']
    return df[cols]

def main():
    current_path = os.path.abspath(__file__)
    base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    data_dir = os.path.join(base_project_dir, 'data')
    
    smoke_path = os.path.join(data_dir, 'smoke_detection_iot.csv')
    car_train_path = os.path.join(data_dir, 'train-00000-of-00001.parquet')
    car_test_path = os.path.join(data_dir, 'test-00000-of-00001.parquet')
    
    print("Purge des zones mortes de l'incendie (Élimination du chevauchement)...")
    processed_smoke = process_smoke_data(smoke_path)
    processed_car_train = process_car_data(car_train_path)
    processed_car_test = process_car_data(car_test_path)
    
    smoke_train = processed_smoke.sample(frac=0.8, random_state=42)
    smoke_test = processed_smoke.drop(smoke_train.index)
    
    final_train = pd.concat([smoke_train, processed_car_train], ignore_index=True).sort_values('Timestamp')
    final_test = pd.concat([smoke_test, processed_car_test], ignore_index=True).sort_values('Timestamp')
    
    train_output_path = os.path.join(data_dir, 'final_train_data_5_features.csv')
    test_output_path = os.path.join(data_dir, 'final_test_data_5_features.csv')
    
    final_train.to_csv(train_output_path, index=False)
    final_test.to_csv(test_output_path, index=False)
    
    print(f"Pipeline terminé ! Train: {final_train.shape[0]} lignes")

if __name__ == "__main__":
    main()