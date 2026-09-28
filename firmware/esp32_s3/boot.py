from network import WLAN
import time
import machine
import config
from simple import MQTTClient
import gc # 导入垃圾回收器

mqtt_client = None

def connect_wifi_and_mqtt():
    global mqtt_client
    
    # 在最开始强制回收一次内存，确保 Wi-Fi 享有最高优先级内存
    gc.collect()
    
    wlan = WLAN(mode=WLAN.STA)
    time.sleep_ms(500)
    
    if not wlan.isconnected():
        print("FiPy tente de se connecter au Wi-Fi : {} ...".format(config.WIFI_SSID))
        wlan.connect(config.WIFI_SSID, auth=(None, config.WIFI_PASSWORD))
        
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        print()

    if wlan.isconnected():
        print("\n====== [BOOT] Wi-Fi Connecté avec succès ======")
        print("Free Memory before MQTT: {} bytes".format(gc.mem_free()))
        
        print("Connexion au Broker MQTT : {}...".format(config.RASPBERRY_PI_IP))
        try:
            mqtt_client = MQTTClient("FiPy_Client", config.RASPBERRY_PI_IP)
            mqtt_client.connect()
            print("[BOOT Succès] Connecté au serveur MQTT !\n")
        except Exception as e:
            print("[BOOT Erreur] Échec MQTT : {}".format(e))
            mqtt_client = None
            
        # 连接成功后，再次强制回收网络栈产生的临时垃圾
        gc.collect()
    else:
        print("\n[Erreur] Wi-Fi Timeout.")

connect_wifi_and_mqtt()  