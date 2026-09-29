import os
import pandas as pd
import numpy as np
import joblib

def run_med_inference():
    # 动态定位项目根目录 edge_server
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return

    data_path = os.path.join(base_project_dir, 'données', 'med_sensor_inference_test.csv')
    scaler_path = os.path.join(base_project_dir, 'ai_engine', 'med', 'data_scaler_xgboost.joblib')
    model_path = os.path.join(base_project_dir, 'ai_engine', 'med', 'xgboost_body_model.joblib')
    output_path = os.path.join(base_project_dir, 'data', 'med_inference_output.csv')

    if not os.path.exists(data_path):
        print(f"Erreur : Fichier de données introuvable -> {data_path}")
        return
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Erreur : Modèle médical ou Scaler manquant dans 'ai_engine/med/'.")
        return

    print("Chargement du jeu de données physiologiques...")
    df = pd.read_csv(data_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    print("Chargement du Scaler et du Modèle XGBoost Médical...")
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # 提取模型所需的 2 个输入生理特征
    feature_cols = ['HeartRate', 'HRV']
    X_scaled = scaler.transform(df[feature_cols])

    print("Exécution des prédictions physiologiques...")
    df['Predicted_Label'] = model.predict(X_scaled)
    prob_matrix = model.predict_proba(X_scaled)

    # 记录三类概率分布
    df['Prob_Normal(0)'] = prob_matrix[:, 0]
    df['Prob_Fatigue(1)'] = prob_matrix[:, 1]
    df['Prob_Crise(2)'] = prob_matrix[:, 2]

    # 映射标签状态
    status_map = {0: 'Normal', 1: 'Fatigue', 2: 'Crise Cardiaque'}
    df['Status'] = df['Predicted_Label'].map(status_map)

    # 规则层辅助兜底：若血氧严重过低直接标记为潜在缺氧危险
    df['Hypoxie_Warning'] = np.where(df['SpO2'] < 92, True, False)

    df.to_csv(output_path, index=False)
    print(f"\nÉvaluation médicale terminée ! Résultats exportés sous : {output_path}")

    print("\n" + "=" * 45)
    print("STATISTIQUES DE DIAGNOSTIC MÉDICAL")
    print("=" * 45)
    counts = df['Status'].value_counts()
    for status_type in ['Normal', 'Fatigue', 'Crise Cardiaque']:
        print(f"  -> {status_type} : {counts.get(status_type, 0)} détections")
    
    hypoxie_cases = df['Hypoxie_Warning'].sum()
    print(f"  -> Alertes Hypoxie (SpO2 < 92%) : {hypoxie_cases} cas")
    print("=" * 45)

    print("\nExtrait des prédictions médicales :")
    print(df[['Timestamp', 'HeartRate', 'HRV', 'SpO2', 'Status', 'Prob_Crise(2)']].head(10))

if __name__ == "__main__":
    run_med_inference()