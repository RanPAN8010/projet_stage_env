Système de Surveillance Environnementale Embarqué (IoT & IA)

Description du projet
Ce projet surveille l'environnement dans une voiture. Deux cartes électroniques (ESP32-S3 et Pycom FiPy) mesurent les données et les envoient à un micro-ordinateur Raspberry Pi. Un modèle d'intelligence artificielle (XGBoost) analyse ces données (température, humidité, chaleur) pour détecter trois états :

0 : Normal (sécurité)

1 : Forte chaleur (canicule)

2 : Danger (fumée ou début d'incendie)

Fonctionnement du réseau
L'ESP32 et le FiPy mesurent la température et l'humidité.
Ils envoient les données par Wi-Fi (protocole MQTT) au Raspberry Pi.
Le Raspberry Pi enregistre les données et applique le modèle d'IA pour déclencher des alertes si nécessaire.

Organisation des dossiers

firmware/ : contient le code MicroPython pour les cartes ESP32 et FiPy.

edge_server/data/ : stocke les jeux de données bruts, les données préparées et les mesures des capteurs.

edge_server/utils/ : contient les scripts pour nettoyer les données et tester le matériel.

edge_server/ai_trainning/ : contient les scripts pour entraîner et tester le modèle d'IA.

edge_server/ai_engine/ : contient le modèle final entraîné au format JSON.

## Câblage Matériel

Ce projet utilise une carte ESP32-S3 connectée à un module capteur de température et d'humidité DHT11.

> **Remarque** : L'ordre des broches du module DHT11, de gauche à droite, est S (Data), VCC (Alimentation) et GND (Masse).

### Correspondance des broches

| Broche du module DHT11 | Broche ESP32-S3 | Description |
| :--- | :--- | :--- |
| **Broche 1 (S / Data)** | **GPIO X** | Transmission des données |
| **Broche 2 (VCC)** | **3V3** | Alimentation 3.3V |
| **Broche 3 (GND)** | **GND** | Masse | 

Collecte des données réelles
Étape 1 : Connexion au Raspberry Pi
Branchez le câble réseau et l'alimentation sur le Raspberry Pi.
Connectez-vous en SSH avec PuTTY à l'adresse 10.3.183.6 (identifiant : pi, mot de passe : raspberry).
Vérifiez l'adresse Wi-Fi du Raspberry Pi avec la commande ip a show wlan0.

Étape 2 : Réception des données
Sur le Raspberry Pi, entrez dans le dossier ~/IoT puis lancez la commande python data_logger.py. Le programme attend les messages de l'ESP32.

Étape 3 : Envoi des données depuis l'ESP32
Ouvrez votre logiciel de programmation (Thonny).
Dans le fichier config.py, renseignez le nom de votre réseau Wi-Fi, son mot de passe et l'adresse IP Wi-Fi du Raspberry Pi.
Lancez le fichier boot.py puis main.py. L'ESP32 envoie alors une mesure toutes les cinq secondes.

Étape 4 : Récupération du fichier
Sur le Raspberry Pi, les mesures sont écrites dans sensor_data_for_ai.csv.
Utilisez le logiciel WinSCP pour copier ce fichier sur votre ordinateur dans le dossier edge_server/data/.

Entraînement et test de l'IA
Étape 1 : Préparation
Téléchargez les jeux de données AutoTherm et Smoke Detection dans le dossier edge_server/data/.
Lancez la commande : python edge_server/utils/data_prep/clean_env_data.py. Cela génère les fichiers finaux pour l'entraînement.

Étape 2 : Optimisation (optionnel)
Pour trouver les meilleurs réglages, lancez : python edge_server/ai_trainning/tune_car_safety_model.py.
Les paramètres retenus sont : learning_rate = 0.05, max_depth = 5, min_child_weight = 1, n_estimators = 100.

Étape 3 : Entraînement
Lancez : python edge_server/ai_trainning/car_safety_xgboost_model.py. Le modèle fini s'enregistre dans edge_server/ai_engine/car_safety_xgboost_model.json.

Étape 4 : Prédiction sur vos capteurs
Vérifiez que votre fichier de mesures réelles est bien nommé sensor_data_for_ai.csv dans le dossier data.
Lancez la commande : python edge_server/ai_trainning/predict_service.py. Les prédictions finales s'enregistrent dans sensor_inference_output.csv.