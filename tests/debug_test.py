import pandas as pd
import numpy as np

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]

df_su = pd.read_csv(BASE_DIR / 'data/malatya_gunluk_su_tuketimi_20yil.csv')
df_yagis = pd.read_csv(BASE_DIR / 'data/malatya_gunluk_yagis_20yil.csv')

print("===== HEDEF SÜTUN TESPITI =====")
target_col_list = [c for c in df_su.columns if 'su' in c.lower() and 'tuketim' in c.lower()]
if not target_col_list:
    print("HATA: Hedef sütun bulunamadı!")
else:
    target_col = target_col_list[0]
    print(f"Hedef sütun: {target_col}")
    print(f"Hedef sütun türü: {df_su[target_col].dtype}")
    print(f"Hedef sütundaki NaN: {df_su[target_col].isna().sum()}")

print("\n===== TARİH İŞLEMİ =====")
try:
    df_su['date'] = pd.to_datetime(df_su['Tarih'], errors='coerce')
    df_yagis['date'] = pd.to_datetime(df_yagis['Tarih'], errors='coerce')
    print("Tarih sütunları başarıyla oluşturuldu")
except Exception as e:
    print(f"Tarih hatası: {e}")

print("\n===== BİRLEŞTİRME İŞLEMİ =====")
try:
    common_cols = [c for c in df_yagis.columns if c != 'date' and c != 'Tarih' and c not in df_su.columns]
    print(f"Ortak sütunlar: {common_cols}")
    
    df_merged = df_su.merge(df_yagis[['date'] + common_cols], on='date', how='inner', suffixes=('_su', '_yagis'))
    print(f"Birleştirilmiş veri: {df_merged.shape}")
    
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    print(f"Sıralı veri: {df_merged.shape}")
    
    # Hedef sütun
    print(f"\nHedef sütun: {target_col}")
    print(f"Birleştirilmiş veri sütunları: {df_merged.columns.tolist()}")
    
except Exception as e:
    print(f"Birleştirme hatası: {e}")
    import traceback
    traceback.print_exc()
