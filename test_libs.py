print("===== Kütüphane Yükleme Testi =====")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow yüklendi: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow hatası: {e}")
    print("  TensorFlow yükleniyor...")
    import subprocess
    subprocess.run(['pip', 'install', 'tensorflow'], check=True)

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    print("✓ Keras bileşenleri yüklendi")
except ImportError as e:
    print(f"❌ Keras hatası: {e}")

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
    print("✓ Diğer kütüphaneler yüklendi")
except ImportError as e:
    print(f"❌ Kütüphane hatası: {e}")

print("\n✓ Tüm kütüphaneler başarıyla yüklendi!")
