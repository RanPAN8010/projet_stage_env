import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
import xgboost as xgb

def tune_hyperparameters():
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
    # Calcul des poids des échantillons (spécifiquement pour traiter la rareté des données de Canicule)
    print("Calcul des poids des classes pour équilibrer le jeu de données...")
    # 'balanced' 会自动给稀少样本赋予更高的权重权重
    # 'balanced' attribue automatiquement un poids plus élevé aux classes minoritaires
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # 定义超参数搜索网格
    # Définition de la grille de recherche des hyperparamètres
    param_grid = {
        'max_depth': [5, 7],               # 树的深度（5或7）Profondeur maximale des arbres (5 ou 7)
        'learning_rate': [0.05, 0.1],      # 学习率 Taux d'apprentissage
        # Poids minimal requis pour créer une feuille 
        # (une valeur plus élevée limite le surapprentissage sur la classe minoritaire)
        'min_child_weight': [1, 3],        # 决定叶子节点合并的最小权重（越大越防少数类过拟合）
        'n_estimators': [100]              # 保持基木树量为 100 Maintenir le nombre d'arbres de base à 100
    }
    
    # 初始化基础三分类模型
    # Initialisation du modèle de classification à 3 classes de base
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # 3折交叉验证 (cv=3)
    # Validation croisée à 3 blocs (cv=3),
    print("Lancement de la recherche par grille (GridSearchCV) avec validation croisée...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro', # L'usage de f1_macro contraint le modèle à classifier correctement la classe minoritaire (Canicule)
        n_jobs=-1,          # Activer l'accélération parallèle en utilisant tous les cœurs du processeur
        verbose=2
    )
    
    # 执行搜索，同时传入类别平衡权重
    # Lancer la recherche en appliquant les poids d'équilibrage des classes
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 输出最佳参数结果
    # Afficher les paramètres optimaux obtenus
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