import os
import csv
import json
from datetime import datetime
import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost" 
MQTT_TOPIC = "esp32/data"
CSV_FILE_PATH = "sensor_data_for_ai.csv"

if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "temperature", "humidity"])
    print(f"Nouveau fichier créé : {CSV_FILE_PATH}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connextion réussie au Broker MQTT local")
        client.subscribe(MQTT_TOPIC)
        print(f"Abonnement réussi; Ecoute du sujet : {MQTT_TOPIC} ...")
    else:
        print(f"Echec de Connexion; Code d'erreur: {rc}")

def on_message(client, userdata, msg):
    try:
	#解码 Décodage
        payload_str = msg.payload.decode('utf-8')
        print(f"Données brute reçues: {payload_str}")
        
        #解析 JSON 数据
        # Analyse des données JSON
        data = json.loads(payload_str)
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        
        #  数据合规性检查防止传感器异常产生的空数据污染训练集
        # Vérification de la conformité des données 
        # pour éviter que des valeurs nulles dues à des anomalies de capteur ne corrompent le dataset
        if temperature is None or humidity is None:
            print("Données incomplètes reçues.")
            return
            
        #获取当前时间戳
        # Récupération de l'horodatage actuel
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        #写入 CSV 文件
        # Écriture dans le fichier CSV
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, temperature, humidity])
            
        print(f"Succès: {current_time} -> Temp: {temperature}C, Hum: {humidity}%")
        
    except json.JSONDecodeError:
        print(f"Impossible d'analyser le JSON: {msg.payload}")
    except Exception as e:
        print(f"Exception d'un erreur est survenue lors du traitement: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Démarrage du programme de collecte de données...")
client.connect(MQTT_BROKER, 1883, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n Le programme de collecte a été arrêté manuellement.")
