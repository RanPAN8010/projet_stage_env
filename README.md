# Système de surveillance d'urgence basé sur une détection bi-puce et l'IA embarquée

## Présentation du projet

Ce projet est un système IoT complet développé entièrement en Python. Il utilise une architecture de détection à deux nœuds, composée d'un **ESP32-S3** et d'un **Pycom FiPy 1.0**, permettant de collecter des données environnementales sur plusieurs canaux de manière redondante. Les données physiques mesurées sont transmises en temps réel via le réseau local à un serveur de calcul en périphérie (**Raspberry Pi**). Ce dernier exécute des **algorithmes d'IA** pour détecter et analyser les situations d'urgence en quelques secondes.

---

## Technologies et environnement de développement

### 1. Couche de détection matérielle (Dual-Node Sensing)

* **Nœud de détection A :** Carte de développement ESP32-S3 (sous MicroPython).
* **Nœud de détection B :** Module Pycom FiPy 1.0 branché sur une carte d'extension Pytrack (sous MicroPython).
* **Environnement de développement (IDE) :** Thonny.

### 2. Couche Serveur et IA embarquée (Raspberry Pi Server)

* **Matériel central :** Raspberry Pi.
* **Système d'exploitation :** Raspberry Pi OS (Debian).
* **Langage de programmation :** Python 3.
* **Bibliothèques et protocoles :** TensorFlow Lite / OpenCV (pour l'inférence de l'IA), Sockets Python / MQTT.

---

## 📐 Architecture du système

```text
 [ Nœud ESP32-S3 ] (MicroPython) ──────┐
                                       ├── (Wi-Fi / Réseau Local) ──▶ [ Serveur Raspberry Pi (Python 3) ] ──▶ [ Inférence IA ] ──▶ Alerte d'urgence
 [ Nœud Pycom FiPy 1.0 ] (MicroPython) ┘

```

## 📂 Structure du projet

```text
├── .gitignore               # Configuration des exclusions Git
├── README.md                # Fichier principal de description du projet
├── edge_server/             # Code source du serveur de calcul en périphérie (Python 3)
│   ├── ai_engine/           # Dossier de stockage des modèles d'IA entraînés (ignoré sur Git)
│   │   ├── env/             # Structure pour les modèles liés aux données environnementales (.gitkeep)
│   │   └── med/             # Structure pour les modèles liés aux données médicales (.gitkeep)
│   ├── ai_trainning/        # Scripts d'entraînement des modèles d'IA
│   │   ├── env/             # Entraînement des modèles environnementaux (ex: XGBoost)
│   │   └── med/             # Entraînement des modèles médicaux (ex: Random Forest, KNN)
│   ├── network/             # Scripts de communication réseau (Serveur de réception)
│   ├── utils/               # Outils de traitement de données et pilotes matériels
│   │   ├── data_prep/       # Nettoyage, corrélation et fusion des données (HRV, Fatigueset, etc.)
│   │   └── hardware/        # Pilotes et tests des capteurs (SpO2, ECG, Tension artérielle)
│   ├── data/                # Dossier de stockage des données brutes (.gitkeep)
│   └── requirements.txt     # Liste des dépendances Python pour le serveur
└── firmware/                # Code source des nœuds de détection embarqués (MicroPython)
    ├── esp32_s3/            # Programme et configuration pour le nœud ESP32-S3
    └── fipy_10/             # Programme, configuration et bibliothèques capteurs (L76GNSS, LIS2HH12, etc.) pour le nœud FiPy 1.0
```

---

## Préparation des jeux de données pour l'entraînement

Ce projet repose sur l'entraînement de deux modèles d'IA distincts :
1. **Modèle Environnemental** : Surveillance thermique de l'habitacle et détection précoce d'incendie.
2. **Modèle Médical / Physiologique** : Suivi des constantes vitales du conducteur (détection de fatigue et alertes de crises cardiaques).

Suivez les instructions ci-dessous pour télécharger les données brutes nécessaires et exécuter les scripts de préparation correspondants.

---

### 1. Modèle Environnemental (Sécurité Cabine)

Ce modèle classifie la situation thermique et de sécurité du véhicule en 3 états : Sécurité (0), Canicule (1) et Incendie/Fumée (2).

#### Données sources à télécharger :
* **AutoTherm (Données thermiques habitacle)** :
  * Lien : [Hugging Face - AutoTherm](https://huggingface.co/datasets/kopetri/AutoTherm)
  * Fichiers requis : `train-00000-of-00001.parquet` et `test-00000-of-00001.parquet`
  * Emplacement cible : `edge_server/data/`
* **Smoke Detection Dataset (Données de détection d'incendie)** :
  * Lien : [Kaggle - Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset)
  * Fichier requis : `smoke_detection_iot.csv`
  * Emplacement cible : `edge_server/data/`

#### Script de nettoyage et de fusion :
Exécutez le script suivant pour calculer les caractéristiques dynamiques (taux de variation horaire et indice de chaleur) et fusionner les données :

```bash
python edge_server/utils/data_prep/clean_env_data.py

* **Fichiers générés dans edge_server/data/** :

  * final_train_data_5_features.csv (Jeu d'entraînement)

  * final_test_data_5_features.csv (Jeu de test)

### 2. Modèle Médical / Physiologique (État du Conducteur)

Ce modèle évalue l'état de santé et de vigilance du conducteur en 3 catégories : Normal (0), Fatigue (1) et Crise cardiaque (2).

#### Données sources à télécharger :
* **Fatigueset (Détection du niveau de fatigue mentale) :
  * Lien : Kaggle - Mental Fatigue Level Detection (https://www.kaggle.com/datasets/tanjemahamed/mental-fatigue-level-detection-fatigueset-data)
  * Format : Archive contenant les dossiers des participants et des sessions (fichiers wrist_hr.csv, wrist_skin_temperature.csv, chest_rr_interval.csv et exp_fatigue.csv).
  * Emplacement cible : Décompresser l'archive complète sous le dossier 'edge_server/data/'

* **Heart Disease Dataset (Maladies cardiaques) :
 * Lien : Kaggle - Heart Disease Dataset (https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)
 * Fichier requis : heart_statlog_cleveland_hungary_final.csv
 * Emplacement cible : edge_server/data/

#### Scripts de préparation (à exécuter dans cet ordre) :
Nettoyage, filtrage et rééchantillonnage temporel des signaux de fatigue (fenêtre de 1 seconde) :

```Bash
python edge_server/utils/data_prep/clean_fatigueset.py
Fichier généré : edge_server/data/fatigueset_cleaned.csv

Extraction des indicateurs cardiaques (HeartRate, HRV) et fusion avec le jeu de fatigue :

```Bash
python edge_server/utils/data_prep/merge_heart_and_fatigue.py
Fichier généré : edge_server/data/driver_body_status_train.csv (Jeu de données final prêt pour l'entraînement du modèle physiologique)

### 3. Trainning
```Bash
 * python edge_server\ai_trainning\med\train_xgboost.py
 * python edge_server\ai_trainning\env\car_safety_xgboost_model.py

### 4. Inférence
#### changer d'abord edge_server\ai_trainning\env\predict_service.py: de

    current_path = os.path.abspath(__file__)
    if "edge_server" in current_path:
        base_project_dir = current_path.split("edge_server")[0] + "edge_server"
    else:
        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")
        return

    data_path = os.path.join(base_project_dir, 'données', 'med_sensor_inference_test.csv')
    scaler_path = os.path.join(base_project_dir, 'ai_engine', 'med', 'data_scaler_xgboost.joblib')
    model_path = os.path.join(base_project_dir, 'ai_engine', 'med', 'xgboost_body_model.joblib')
    output_path = os.path.join(base_project_dir, 'données', 'med_inference_output.csv')
#### à
    current_path = os.path.abspath(__file__)

    if "edge_server" in current_path:

        base_project_dir = current_path.split("edge_server")[0] + "edge_server"

    else:

        print("Erreur : Le script n'est pas placé dans le dossier 'edge_server' !")

        return
    data_path = os.path.join(base_project_dir, 'data', 'sensor_data_for_ai 1.csv')
    output_path = os.path.join(base_project_dir, 'data', 'sensor_inference_output.csv')
 
```Bash
 * python edge_server/ai_trainning/env/predict_service.py

## 👤 Organisation et répartition des tâches

* **Pan RAN** : Développement complet et autonome du projet de bout en bout. Réalisation des scripts MicroPython sur PyCharm pour la détection bi-puce (ESP32-S3 et FiPy 1.0), mise en place du protocole de communication sur le réseau local, configuration du serveur Raspberry Pi et déploiement du modèle d'IA pour la détection des urgences.

---

## 📝 Licence

Ce projet est sous licence [MIT]. Développé uniquement dans un cadre académique et expérimental.</Nom_du_Projet>
