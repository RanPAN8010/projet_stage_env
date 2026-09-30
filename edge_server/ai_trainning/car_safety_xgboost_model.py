import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

def train_final_model():
    current_path = os.path.abspath(__file__)
    base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    data_dir = os.path.join(base_project_dir, 'data')
    
    train_path = os.path.join(data_dir, 'final_train_data_5_features.csv')
    test_path = os.path.join(data_dir, 'final_test_data_5_features.csv')
    model_output_path = os.path.join(base_project_dir, 'ai_engine', 'car_safety_xgboost_model.json')
    
    print("Chargement des données de train et test...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    feature_cols = ['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index']
    
    X_train = df_train[feature_cols]
    y_train = df_train['Label']
    X_test = df_test[feature_cols]
    y_test = df_test['Label']
    
    print(f"Échantillons d'entraînement : {X_train.shape[0]} lignes")
    print(f"Échantillons de test : {X_test.shape[0]} lignes")
    

    # 放弃 sample_weight 自动平衡
    # 移除过度加权，防止常温背景行被错误放大
    print("Entraînement du modèle XGBoost à 3 classes (Sécurité, Canicule, Feu)...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n=== Rapport d'évaluation du modèle ===")
    print(f"Précision globale (Accuracy) : {accuracy_score(y_test, y_pred):.4f}")
    print("\nRapport détaillé par classe :")
    status_names = ['Sécurité (0)', 'Canicule (1)', 'Feu/Fumée (2)']
    print(classification_report(y_test, y_pred, target_names=status_names))
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save_model(model_output_path)
    print(f"\nModèle exporté avec succès vers : {model_output_path}")

if __name__ == "__main__":
    train_final_model()