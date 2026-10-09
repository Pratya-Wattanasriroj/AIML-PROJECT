import os, shutil

raw_dir = 'dataset' 
final_dir = 'dataset_custom'

if not os.path.exists(raw_dir):
    print("หาโฟลเดอร์ต้นทางไม่เจอ!")
    exit()

if os.path.exists(final_dir): 
    shutil.rmtree(final_dir)
os.makedirs(final_dir)


target_mapping = {
    'biome_1': 'plains',
    'biome_4': 'forest',
    'biome_3': 'mountains',
    'biome_5': 'taiga',
    'biome_2': 'desert',
    'biome_12': 'snowy_tundra',
    'biome_30': 'snowy_tundra',      # ❄️ ยุบรวมเข้า Snowy Tundra
    'biome_13': 'snowy_tundra',      # ❄️ ยุบรวมเข้า Snowy Tundra
    'biome_35': 'savanna',
    'biome_27': 'birch_forest',
    'biome_29': 'dark_forest',
    'biome_6': 'swamp',
    'biome_21': 'jungle',
    'biome_37': 'badlands'          
}

print("กำลังจัดเตรียมข้อมูล 16 ไบโอม (เน้นๆ เฉพาะตัวท็อป)...")

for folder_name in os.listdir(raw_dir):
    src_folder = os.path.join(raw_dir, folder_name)
    if not os.path.isdir(src_folder): continue
        
    # ถ้าชื่อโฟลเดอร์ตรงกับในโพย ถึงจะก๊อปปี้ไปใช้งาน
    if folder_name in target_mapping:
        biome_name = target_mapping[folder_name]
        shutil.copytree(src_folder, os.path.join(final_dir, biome_name), dirs_exist_ok=True)

print(f"เตรียม Dataset เสร็จแล้ว! คัดมาเฉพาะ {len(target_mapping)} ไบโอม พร้อมเทรนทั้ง 2 โมเดล")