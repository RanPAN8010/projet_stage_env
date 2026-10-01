import time
import gc

try:
    import boot
    client = boot.mqtt_client
except Exception as e:
    client = None

if client is None:
    print("[Attention] Pas de client MQTT. Mode autonome.")
else:
    print("[Info] Client MQTT initialisé avec succès.")

print("Libération de la mémoire avant chargement des pilotes...")
gc.collect()
time.sleep(1)
print("Chargement des pilotes Pytrack et GPS...")
from pycoproc_1 import Pycoproc
from L76GNSS import L76GNSS

py = Pycoproc(Pycoproc.PYTRACK)
gnss = L76GNSS(py, timeout=30)
print("Matériel prêt. Démarrage de la collecte GPS...")

while True:
    coord = gnss.coordinates()
    
    if coord[0] is not None and coord[1] is not None:
        payload = "{{\"latitude\": {}, \"longitude\": {}}}".format(coord[0], coord[1])
    else:
        payload = "{\"latitude\": null, \"longitude\": null}"
        
    print("Envoi: {}".format(payload))
    
    if client is not None and getattr(client, 'sock', None) is not None:
        try:
            client.publish("fipy/gps", payload.encode('utf-8'))
        except Exception as e:
            print("[Erreur] Perte de connexion : {}".format(e))
            client.sock = None

    gc.collect()
    time.sleep(5)