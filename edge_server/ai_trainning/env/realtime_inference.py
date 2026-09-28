import os
import json
import subprocess
from datetime import datetime
import numpy as np
import xgboost as xgb
import paho.mqtt.client as mqtt

# 基础路径配置（动态定位到项目根目录）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'ai_engine', 'env', 'car_safety_xgboost_model.json')
BUZZER_SCRIPT = os.path.join(BASE_DIR, 'script', 'buzzer.sh')

# MQTT 参数
MQTT_BROKER = "localhost"
MQTT_TOPIC = "esp32/data"

# 1. 启动时加载 XGBoost 模型至内存
print(f"Chargement du modèle depuis : {MODEL_PATH}")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("Modèle chargé avec succès.")

# 状态记录（用于计算差分特征）
last_temp = None
last_humidity = None
status_map = {0: 'Sécurité', 1: 'Canicule', 2: 'Feu/Fumée'}

def calculate_heat_index(T_celsius, RH):
    """根据摄氏度和相对湿度计算热指数 (Heat Index)"""
    T_f = T_celsius * 1.8 + 32
    HI_f = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))
    if T_f >= 80:
        T, R = T_f, RH
        HI_full = (-42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - 
                   6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R + 
                   8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2)
        if R < 13 and 80 <= T <= 112:
            HI_full -= ((13 - R) / 4) * np.sqrt((17 - np.abs(T - 91.)) / 14)
        elif R > 85 and 80 <= T <= 87:
            HI_full += ((R - 85) / 10) * ((87 - T) / 5)
        HI_f = HI_full
    return (HI_f - 32) / 1.8

def trigger_alert():
    """触发本地蜂鸣器告警"""
    if os.path.exists(BUZZER_SCRIPT):
        # 异步启动脚本，避免阻塞 MQTT 回调
        subprocess.Popen(["bash", BUZZER_SCRIPT])
    else:
        print(f"[ALERTE] Fichier de script introuvable : {BUZZER_SCRIPT}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connecté au Broker MQTT. Abonnement au topic : {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Échec de connexion, code : {rc}")

def on_message(client, userdata, msg):
    global last_temp, last_humidity
    try:
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        
        temp = data.get("temperature")
        hum = data.get("humidity")
        
        if temp is None or hum is None:
            return

        temp = float(temp)
        hum = float(hum)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 动态特征提取 (Rates & Heat Index)
        temp_rate = 0.0 if last_temp is None else (temp - last_temp)
        hum_rate = 0.0 if last_humidity is None else (hum - last_humidity)
        heat_index = calculate_heat_index(temp, hum)

        # 更新上一帧
        last_temp = temp
        last_humidity = hum

        # 构造特征向量: ['Temperature', 'Humidity', 'Temp_Rate', 'Humidity_Rate', 'Heat_Index']
        features = np.array([[temp, hum, temp_rate, hum_rate, heat_index]])

        # 模型单步推理
        pred_label = int(model.predict(features)[0])
        prob_matrix = model.predict_proba(features)[0]
        status = status_map.get(pred_label, 'Inconnu')
        prob_fire = prob_matrix[2]

        print(f"[{timestamp}] T: {temp:.1f}°C, H: {hum:.1f}% | Rate: ΔT={temp_rate:+.1f}, ΔH={hum_rate:+.1f} | État: {status} (P_Feu: {prob_fire:.4f})")

        # 危险判定与硬件联动
        if pred_label == 2:  # Feu/Fumée
            print(f"⚠️ [DANGER] Incendie/Fumée détecté à {timestamp} ! Déclenchement de l'alerte...")
            trigger_alert()
            
    except json.JSONDecodeError:
        print(f"Erreur décodage JSON : {msg.payload}")
    except Exception as e:
        print(f"Erreur d'inférence en direct : {e}")

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("Démarrage du service d'inférence en temps réel...")
    client.connect(MQTT_BROKER, 1883, 60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nArrêt du service d'inférence.")