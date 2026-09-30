import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
import xgboost as xgb

def tune_hyperparameters():
    # 动态路径定位
    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return
        
    data_dir = os.path.join(base_project_dir, 'data')
    train_path = os.path.join(data_dir, 'final_train_data_5_features.csv')
    
    print("Chargement des données d'entraînement pour l'optimisation...")
    df_train = pd.read_csv(train_path)
    
    feature_cols = ['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index']
    X_train = df_train[feature_cols]
    y_train = df_train['Label']
    
    # 计算样本权重（专门对付 Canicule 样本极少的问题）
    print("Calcul des poids des classes pour équilibrer le jeu de données...")
    # 'balanced' 会自动给稀少样本赋予更高的权重权重
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 定义超参数搜索网格
    # 限制了搜索空间的范围（因为100万行数据太大，范围太广会搜得很慢）
    param_grid = {
        'max_depth': [5, 7],               # 树的深度（5或7）
        'learning_rate': [0.05, 0.1],      # 学习率
        'min_child_weight': [1, 3],        # 决定叶子节点合并的最小权重（越大越防少数类过拟合）
        'n_estimators': [100]              # 保持基木树量为 100
    }
    
    # 初始化基础三分类模型
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # 3折交叉验证 (cv=3)，聚焦提升不均衡样本的 f1_macro 指标
    print("Lancement de la recherche par grille (GridSearchCV) avec validation croisée...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro', # 使用 f1_macro 会强迫模型必须把高温天气（少数类）也预测对
        n_jobs=-1,          # 开启所有 CPU 核心多线程并行加速
        verbose=2
    )
    
    # 执行搜索，同时传入类别平衡权重
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 输出最佳参数结果
    print("\n==================================================")
    print("=== Optimisation terminée avec succès ! ===")
    print("==================================================")
    print(f"Meilleur score F1-Macro obtenu : {grid_search.best_score_:.4f}")
    print("\nVoici les meilleurs paramètres à reporter dans votre script d'entraînement :")
    print("--------------------------------------------------")
    for param, value in grid_search.best_params_.items():
        print(f"  -> {param} : {value}")
    print("--------------------------------------------------")
    print("Veuillez ajouter ces paramètres et la gestion des poids (sample_weight) dans 'car_safety_xgboost_model.py'.")

if __name__ == "__main__":
    tune_hyperparameters()