import asyncio
import struct
from scan import scan_ble_packets

async def process_fipy_gps_data():
    print("Starting GPS data parsing business logic...")
    print("=" * 60)
    
    async for address, name, rssi, raw_data in scan_ble_packets(rssi_threshold=-75):
        if name == "FiPy_GPS":
            print(f"[Target Detected] MAC: {address} | RSSI: {rssi} dBm")
            
            if not raw_data:
                print(" └── [Warning] No data payload found in advertisement packet.")
                print("-" * 60)
                continue
                
            # 遍历提取任意可能出现的 Key 结构
            # Parcourir et extraire toute structure de clé potentiellement présente
            for company_id, raw_bytes in raw_data.items():
                # 字节长度如果是 8 字节（2个长整数），直接强制解包
                # Si la taille est de 8 octets (2 entiers longs), forcer directement le dépaquetage
                if len(raw_bytes) == 8:
                    try:
                        lat_raw, lon_raw = struct.unpack("<ii", raw_bytes)
                        latitude = lat_raw / 100000.0 if lat_raw != 0 else None
                        longitude = lon_raw / 100000.0 if lon_raw != 0 else None
                        print(f" └── GPS -> Latitude: {latitude}, Longitude: {longitude}")
                    except Exception as e:
                        print(f" └── Structure unpacking failed: {e}")
            print("-" * 60)

if __name__ == "__main__":
    asyncio.run(process_fipy_gps_data())
