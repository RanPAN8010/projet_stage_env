import asyncio
import time
import csv
import os
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
WINDOW_SIZE = 8  # 扩大窗口尺寸以增强滤波效果

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
                
                # 过滤硬件刚启动或换手指时的瞬态固定干扰值 127
                # 如果运行时间还很短（处于测量初期）且数值死板地卡在 127，直接丢弃
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

def save_to_csv(spo2, pulse):
    file_name = "spo2.csv"
    file_exists = os.path.isfile(file_name)
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "SpO2", "Pulse"])
        writer.writerow([current_timestamp, spo2, pulse])
    print(f"Donnees enregistrees avec succes dans {file_name}")

async def main():
    print(f"Connexion a l'oxymetre [{DEVICE_ADDRESS}]...")
    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            if client.is_connected:
                print("Connexion reussie ! Filtre de stabilisation active. Veuillez ne pas bouger...")
                await client.start_notify(NOTIFY_CHARACTERISTIC_UUID, notification_handler)
                
                while len(collected_spo2) < MAX_SAMPLES:
                    await asyncio.sleep(0.5)
                
                print("\nCollecte terminee. Deconnexion de l'appareil...")
                try:
                    await client.stop_notify(NOTIFY_CHARACTERISTIC_UUID)
                except:
                    pass
                    
                avg_spo2 = round(sum(collected_spo2) / len(collected_spo2), 1)
                avg_pulse = round(sum(collected_pulse) / len(collected_pulse), 1)
                
                print("-" * 50)
                print(f"[Rapport Final de Mesure]")
                print(f"- Saturation moyenne en oxygene (SpO2): {avg_spo2} %")
                print(f"- Pouls moyen (Frequence cardiaque): {avg_pulse} bpm")
                print("-" * 50)
                
                save_to_csv(avg_spo2, avg_pulse)
                    
    except Exception as e:
        print(f"\nErreur de communication: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur.")