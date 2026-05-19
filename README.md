# 🌊 Malatya Su Talebi Tahmin Projesi

Malatya ili için su talebini **veri analizi**, **makine öğrenmesi**, **zaman serisi modelleme** ve **derin öğrenme** yaklaşımlarıyla incelemek için hazırlanan kapsamlı bir tahmin sistemi.

---

## 📋 Proje Kapsamı

Bu proje aşağıdaki analizleri içermektedir:

- ✅ Su tüketimi, yağış, nüfus, sosyoekonomik ve tarımsal veri setlerinin analizi
- ✅ Makine öğrenmesi modelleri (Random Forest, XGBoost, LightGBM) ile tahmin
- ✅ SARIMAX ile mevsimsel zaman serisi modelleme
- ✅ GRU ve LSTM tabanlı derin öğrenme modelleri
- ✅ Model sonuçlarını karşılaştıran interaktif Streamlit dashboard

---

## 🏗️ Proje Mimarisi

### Veri Akışı

**1. Veri Kaynakları** 📥
   - Su tüketimi (20 yıl günlük veriler)
   - Yağış, sıcaklık ve iklim verileri
   - Tarımsal araziler ve sulama bilgileri
   - Sosyoekonomik göstergeler

**2. Veri İşleme** 🔧
   - Eksik veri temizliği ve imputasyon
   - Outlier tespiti ve düzeltme
   - Özellik mühendisliği ve transformasyon
   - Ölçeklendirme ve normalizasyon

**3. Model Eğitimi** 🤖
   ```
   ┌─────────────────────────────────────────┐
   │  Makine Öğrenmesi                       │
   │  • XGBoost • LightGBM • Random Forest   │
   ├─────────────────────────────────────────┤
   │  Zaman Serisi (SARIMAX)                 │
   │  • Mevsimsel desenleri yakalar          │
   │  • Otoregresif modelleme                │
   ├─────────────────────────────────────────┤
   │  Derin Öğrenme                          │
   │  • GRU / LSTM                           │
   │  • Sekans modellemesi                   │
   └─────────────────────────────────────────┘
   ```

**4. Tahmin & Değerlendirme** 📈
   - 30 günlük ileri tahmin
   - Model performans metrikleri (MAE, RMSE, R²)
   - Çapraz doğrulama ve validasyon
   - Senaryo analizi

**5. Görselleştirme** 📊
   - Streamlit Dashboard
   - İnteraktif grafikler ve karşılaştırmalar
   - Senaryo analizleri
   - Su kaybı ve talep analizleri

## 📊 Model Performans Karşılaştırması

| Model | MAE | RMSE | R² | Açıklama |
|:---:|:---:|:---:|:---:|:---|
| **Makine Öğrenmesi** | 🟢 Düşük | 🟢 Düşük | 🟢 Yüksek | Random Forest, XGBoost, LightGBM |
| **SARIMAX** | 🟡 Orta | 🟡 Orta | 🟡 Orta | Zaman serisi, mevsimsel desenler |
| **GRU** | 🟡 Orta-Yüksek | 🟡 Orta-Yüksek | 🟡 Orta | RNN tabanlı model |
| **LSTM** | 🟡 Orta-Yüksek | 🟡 Orta-Yüksek | 🟡 Orta | Gradient vanishing çözen RNN |

> 💡 **Not:** Detaylı metrikler dashboard ve notebook'larda görüntülenebilir.

---

## 📁 Proje Yapısı

```
.
├── 📂 data/                          # Veri setleri ve tahmin CSV çıktıları
│   ├── malatya_gunluk_su_tuketimi_20yil.csv
│   ├── malatya_gunluk_yagis_20yil.csv
│   ├── malatya_bolgesel_tarim_arazileri_2004_2023.csv
│   ├── malatya_iklim_senaryosu_2024_2053.csv
│   └── ... (diğer veri dosyaları)
│
├── 📂 notebooks/                     # Eğitim ve analiz notebook'ları
│   ├── su_talebi_tahmini.ipynb       # Makine öğrenmesi modeli
│   ├── sarimax_su_tahminleri.ipynb   # Zaman serisi modeli
│   ├── gru_su_tahminleri.ipynb       # GRU derin öğrenme
│   ├── lstm_su_tahminleri.ipynb      # LSTM derin öğrenme
│   ├── eda_veri_setleri.ipynb        # Keşifsel veri analizi
│   └── adim_adim_veri_analizi.ipynb  # Adım adım analiz
│
├── 📂 scripts/                       # Yardımcı scriptler
│   └── create_adim_adim_notebook.py
│
├── 📂 tests/                         # Test dosyaları
│   ├── test_libs.py                  # Kütüphane kontrolü
│   ├── test_notebook_run.py          # Notebook çalışma testi
│   └── debug_test.py                 # Debug testleri
│
├── 📄 su_talebi_dashboard.py         # Ana Streamlit uygulaması
├── 📄 requirements.txt               # Proje bağımlılıkları
├── 📄 README.md                      # Bu dosya
├── 📄 .gitignore                     # Git ignore kuralları
└── 📄 .gitattributes                 # Git LFS ayarları
```

---

## 🛠️ Teknolojiler

| Kategori | Araçlar |
|:---|:---|
| **Dil** | Python 3.8+ |
| **Veri İşleme** | Pandas, NumPy |
| **Makine Öğrenmesi** | scikit-learn, XGBoost, LightGBM |
| **Derin Öğrenme** | TensorFlow, Keras |
| **Görselleştirme** | Matplotlib, Seaborn, Plotly |
| **Web Uygulaması** | Streamlit |
| **Notebook** | Jupyter |

---

## 🚀 Kurulum

### Adım 1: Proje Klasörüne Gidin
```bash
cd "c:\Users\CASPER\OneDrive\Masaüstü\Proje Pazarı"
```

### Adım 2: Sanal Ortam Oluşturun
```powershell
python -m venv .venv
```

### Adım 3: Sanal Ortamı Etkinleştirin
```powershell
.\.venv\Scripts\Activate.ps1
```

### Adım 4: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

---

## 📊 Dashboard'ı Çalıştırma

```bash
streamlit run su_talebi_dashboard.py
```

Tarayıcınız otomatik olarak `http://localhost:8501` adresinde açılacak ve aşağıdaki özellikleri kullanabileceksiniz:

- 🔀 Model karşılaştırması
- 📈 Tahmin grafikleri
- 🎯 Senaryo analizi
- 💧 Su kaybı analizleri

---

## 📚 Hızlı Başlangıç Örnekleri

### 1️⃣ Dashboard'ı Çalıştırmak
```powershell
streamlit run su_talebi_dashboard.py
```

### 2️⃣ Notebook'ları Açmak

**Makine Öğrenmesi Modeli:**
```bash
jupyter notebook notebooks/su_talebi_tahmini.ipynb
```

**SARIMAX Modeli:**
```bash
jupyter notebook notebooks/sarimax_su_tahminleri.ipynb
```

**Derin Öğrenme Modelleri:**
```bash
jupyter notebook notebooks/gru_su_tahminleri.ipynb
jupyter notebook notebooks/lstm_su_tahminleri.ipynb
```

### 3️⃣ Test Dosyalarını Çalıştırmak

Kütüphaneleri kontrol edin:
```bash
python tests/test_libs.py
```

Veri ve model pipeline testi:
```bash
python tests/test_notebook_run.py
```

Debug testi:
```bash
python tests/debug_test.py
```

---

## 🎓 Temel Kavramlar

### 📊 Veri Seti
- **Zaman Aralığı:** 20 yıl günlük veriler (~7.300 gözlem)
- **Hedef Değişken:** Günlük su tüketimi (m³)
- **Dış Değişkenler:**
  - Yağış miktarı
  - Tarımsal alan
  - Nüfus
  - Ekonomik göstergeler

### 🤖 Model Stratejileri

| Model | Yaklaşım | Güçlü Yönü |
|:---|:---|:---|
| **Makine Öğrenmesi** | Özelliklere dayalı tahmin | Hızlı, doğru sonuçlar |
| **SARIMAX** | Zaman serisi | Mevsimsel desenleri yakalar |
| **GRU** | RNN tabanlı | Uzun bağımlılıkları işler |
| **LSTM** | RNN tabanlı | Gradient vanishing sorununu çözer |

### 🔮 Senaryo Analizi
Dashboard'da şu analizler yapılabilir:
- İklim senaryolarının etkisi
- Su kaybı oranı değişiklikleri
- Talep artış/azalış senaryoları

---

## ⚙️ Proje Detayları

### Veri İşleme
- Eksik veriler imputasyon yöntemiyle doldurulmuş
- Zaman serisi özellikleri mühendislik yöntemiyle oluşturulmuş
- Veriler standardizasyon ile ölçeklendirilmiş

### Model Eğitimi
- Train/Test oranı: 80/20
- Cross-validation kullanılmış
- Hiperparametre optimizasyonu yapılmış

### Tahmin Sonuçları
- 30 günlük ileri tahmin yapılıyor
- Senaryo bazlı tahminler mümkün
- Güven aralıkları hesaplanıyor

---

## 📝 Önemli Notlar

⚠️ **Model Dosyaları:**
- Büyük model dosyaları (`*.joblib`, `*.pkl`, `*.h5`, `*.keras`) `.gitignore` ile dışarıda tutulmuştur
- Git LFS yapılandırması `.gitattributes`'de hazırlanmıştır
- Gelecekte model dosyalarını repositorye eklemek isterseniz LFS kullanılmalıdır

📁 **Proje Yapısı:**
- Proje notebook tabanlı analiz ve model denemeleri ile ilerlemektedir
- Tüm kod, notebook ve veri dosyaları GitHub'da tutulmuştur
- Üretilen ağır model artefaktları ayrı tutulmuştur

🔒 **Git LFS Durumu:**
- `.gitattributes` dosyasında model dosya türleri LFS'e yönlendirilecek şekilde konfigüre edilmiştir
- Şu an LFS zorunlu değildir ancak ihtiyaç duyulduğunda hızlıca aktif edilebilir

---

## 📞 İletişim & Bağlantılar

- **Repository:** [github.com/yildirimyusuf79/malatya-su-talebi-projesi](https://github.com/yildirimyusuf79/malatya-su-talebi-projesi)
- **LinkedIn:** [linkedin.com/in/yusuf-yıldırım](https://www.linkedin.com/in/yusuf-yıldırım-190445295)
- **Geliştirici:** Yusuf Yıldırım

---

## 📄 Lisans

Bu proje eğitim ve araştırma amaçlı olup açık kaynak kullanıcılarına sunulmuştur.

---

**✨ Son Güncellenme:** 19 Mayıs 2026