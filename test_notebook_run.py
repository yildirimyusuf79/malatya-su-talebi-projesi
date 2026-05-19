import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

print("=" * 60)
print("NOTEBOOK TEST KURULUMU")
print("=" * 60)

# Veri yükleme
df_su = pd.read_csv('malatya_gunluk_su_tuketimi_20yil.csv')
df_yagis = pd.read_csv('malatya_gunluk_yagis_20yil.csv')

print(f"✓ Veri yükleme tamamlandı")
print(f"  Su: {df_su.shape}")
print(f"  Yağış: {df_yagis.shape}")

# Tarih işlemi
df_su['date'] = pd.to_datetime(df_su['Tarih'], errors='coerce')
df_yagis['date'] = pd.to_datetime(df_yagis['Tarih'], errors='coerce')

# Birleştirme
common_cols = [c for c in df_yagis.columns if c != 'date' and c not in df_su.columns]
df_merged = df_su.merge(df_yagis[['date'] + common_cols], on='date', how='inner')
df_merged = df_merged.sort_values('date').reset_index(drop=True)

print(f"✓ Birleştirme tamamlandı: {df_merged.shape}")

# Hedef sütun
target_col = [c for c in df_merged.columns if 'su' in c.lower() and 'tuketim' in c.lower()][0]

# Veri hazırlığı
data = df_merged[[target_col]].dropna().values
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print(f"✓ Veri hazırlığı done: {data_scaled.shape}")

# Lag özellikleri
lookback = 30
X, y = [], []
for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i-lookback:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)

print(f"✓ Lag özellikleri: X={X.shape}, y={y.shape}")

# Model eğitimi
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = XGBRegressor(n_estimators=50, max_depth=5, random_state=42, verbosity=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrikler
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))

mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
r2 = r2_score(y_test_orig, y_pred_orig)
mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)

print(f"\n✓ Model Sonuçları (XGBoost):")
print(f"  MAE: {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²: {r2:.4f}")
print(f"  MAPE: {mape:.2%}")

print("\n✓ Notebook çalışmaya hazır!")
print("=" * 60)
