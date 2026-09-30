Système de Surveillance Environnementale Embarqué (IoT & IA)
Présentation du projet

Ce projet est un système IoT de surveillance environnementale pour habitacle de véhicule. Il utilise une architecture de collecte redondante composée de deux nœuds (ESP32-S3 et Pycom FiPy 1.0) transmettant les mesures physiques à un serveur edge (Raspberry Pi). Un modèle XGBoost embarqué analyse les flux temporels (température, humidité, variations dynamiques et indice de chaleur) afin de classifier l'état en trois niveaux : Sécurité (0), Canicule (1) et Incendie/Fumée (2).

Architecture globale

[ Nœud ESP32-S3 ] (MicroPython) ------+
                                        |--> [ Raspberry Pi ] --> [ Inférence XGBoost ] --> Alertes
[ Nœud Pycom FiPy 1.0 ] (MicroPython) +

Structure du projet

.gitignore
README.md
edge_server/
ai_engine/          : Modèle entraîné (car_safety_xgboost_model.json)
ai_trainning/       : Scripts d'entraînement et d'optimisation
data/               : Jeux de données bruts et préparés
network/            : Scripts de réception réseau
utils/
data_prep/      : Nettoyage, fusion et visualisation des données
hardware/       : Pilotes matériels et tests des capteurs
requirements.txt    : Dépendances Python
firmware/
esp32_s3/           : Programme MicroPython ESP32-S3
fipy_10/            : Programme MicroPython FiPy 1.0

Guide d'exécution

Étape 1 : Préparation des données
Télécharger dans edge_server/data/ :

[AutoTherm (Hugging Face) :](https://huggingface.co/datasets/kopetri/AutoTherm) train-00000-of-00001.parquet, test-00000-of-00001.parquet


[Smoke Detection Dataset (Kaggle) :](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset) smoke_detection_iot.csv

Lancer la fusion des caractéristiques :
python edge_server/utils/data_prep/clean_env_data.py
-> Fichiers générés : final_train_data_5_features.csv, final_test_data_5_features.csv

Étape 2 : Entraînement du modèle XGBoost
Lancer l'entraînement :
python edge_server/ai_trainning/car_safety_xgboost_model.py
-> Modèle exporté : edge_server/ai_engine/car_safety_xgboost_model.json

(Optionnel) Optimisation des hyperparamètres et AutoML :
python edge_server/ai_trainning/tune_hyperparameters.py
python edge_server/ai_trainning/car_safety_automl_selection.py

Étape 3 : Inférence et Évaluation
Tester les prédictions sur les données capteurs :
python edge_server/ai_trainning/predict_service.py
-> Résultats exportés : edge_server/data/sensor_inference_output.csv