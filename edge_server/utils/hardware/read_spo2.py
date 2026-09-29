import asyncio
import time
import csv
import os
import numpy as np
from datetime import datetime
from bleak import BleakClient

DEVICE_ADDRESS = "00:A0:50:17:2F:14"
NOTIFY_CHARACTERISTIC_UUID = "49535343-1e4d-4bd9-ba61-23c647249616"

start_time = time.time()
last_print_time = 0

collected_spo2 = []
collected_pulse = []
MAX_SAMPLES = 15

window_pulse = []
window_spo2 = []
WINDOW_SIZE = 8

def notification_handler(sender, data):
    global last_print_time
    data_list = list(data)
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if elapsed_time < 8.0:
        return

    if len(collected_spo2) >= MAX_SAMPLES:
        return

    for i in range(0, len(data_list) - 4, 5):
        frame = data_list[i:i+5]
        if len(frame) == 5:
            pulse_val = frame[3]
            spo2_val = frame[4]
            
            if 80 <= spo2_val <= 100 and 40 <= pulse_val <= 200:
                if pulse_val == 127 and elapsed_time < 15.0:
                    continue
                    
                window_pulse.append(pulse_val)
                window_spo2.append(spo2_val)
                
                if len(window_pulse) > WINDOW_SIZE:
                    window_pulse.pop(0)
                    window_spo2.pop(0)
                
                if len(window_pulse) == WINDOW_SIZE:
                    pulse_stable = (max(window_pulse) - min(window_pulse)) <= 3
                    spo2_stable = (max(window_spo2) - min(window_spo2)) <= 2
                    
                    if pulse_stable and spo2_stable:
                        if current_time - last_print_time >= 1.0:
                            collected_spo2.append(spo2_val)
                            collected_pulse.append(pulse_val)
                            
                            progress = len(collected_spo2)
                            print(f"[Collecte {progress}/{MAX_SAMPLES}] SpO2: {spo2_val}% | Pouls: {pulse_val} bpm")
                            last_print_time = current_time
                return

def save_to_inference_csv(avg_spo2, avg_pulse, hrv_est):
    base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(base_project_dir, 'données')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, 'med_realtime_inference_input.csv')
    file_exists = os.path.isfile(file_path)
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "HeartRate", "HRV", "SpO2"])
        writer.writerow([current_timestamp, avg_pulse, hrv_est, avg_spo2])
    print(f"Données enregistrées dans {file_path}")

async def main():
    print(f"Connexion à l'oxymètre [{DEVICE_ADDRESS}]...")
    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            if client.is_connected:
                print("Connexion réussie ! Filtre activé. Veuillez ne pas bouger...")
                await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)
                
                while len(collected_spo2) < MAX_SAMPLES:
                    await asyncio.sleep(0.5)
                
                print("\nCollecte terminée. Déconnexion...")
                try:
                    await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)
                except:
                    pass
                    
                avg_spo2 = round(float(np.mean(collected_spo2)), 1)
                avg_pulse = round(float(np.mean(collected_pulse)), 1)
                # 使用采集期间脉搏序列的标准差模拟 HRV，若方差过小则给底噪补偿（正常静息约在 20-50 之间）
                hrv_estimate = round(float(np.std(collected_pulse) * 12.0 + 35.0), 2)
                
                print("-" * 50)
                print(f"[Rapport Médical Adapté pour IA]")
                print(f"- HeartRate : {avg_pulse} bpm")
                print(f"- HRV estimé : {hrv_estimate}")
                print(f"- SpO2 : {avg_spo2} %")
                print("-" * 50)
                
                save_to_inference_csv(avg_spo2, avg_pulse, hrv_estimate)
                    
    except Exception as e:
        print(f"\nErreur de communication : {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgramme interrompu.")