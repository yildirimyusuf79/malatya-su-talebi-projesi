# Malatya Su Talebi Projesi

Bu proje, Malatya ili için su talebini veri analizi, klasik makine ogrenmesi, zaman serisi modelleme ve derin ogrenme yaklasimlariyla incelemek icin hazirlandi. Calisma alani; veri setleri, egitim notebook'lari, kaydedilmis model artefaktlari ve sonuclari gorsellestiren bir Streamlit dashboard'i icerir.

## Proje Kapsami

- Malatya'ya ait tuketim, yagis, nufus, sosyoekonomik ve tarimsal veri setlerinin analizi
- Makine ogrenmesi modelleri ile su talebi tahmini
- SARIMAX ile zaman serisi tahmini
- GRU ve LSTM tabanli derin ogrenme modelleri
- Sonuclari karsilastiran interaktif Streamlit dashboard'i

## Proje Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│             Malatya Su Talebi Tahmin Sistemi            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Veri Kaynakları                                         │
│  ├─ Su Tuketimi (20 yil)                                │
│  ├─ Yagis (20 yil)                                      │
│  ├─ Tarimsal Araziler (2004-2023)                       │
│  ├─ Sosyoekonomik Veriler                               │
│  └─ Klim Senaryolari (2024-2053)                        │
│         │                                               │
│         ▼                                               │
│  ┌──────────────────────────┐                           │
│  │   Veri Isleme & EDA      │                           │
│  │  (notebooks/)            │                           │
│  └──────────────────────────┘                           │
│         │                                               │
│    ┌────┴────┬────────┬─────────┐                       │
│    ▼         ▼        ▼         ▼                       │
│  ┌───────────────────────────────────┐                  │
│  │   Model Egitim Pipeline           │                  │
│  ├───────────────────────────────────┤                  │
│  │ • Makine Ogrenmesi (RF, XGB, LGB) │                  │
│  │ • SARIMAX (Zaman Serisi)          │                  │
│  │ • GRU (Derin Ogrenme)             │                  │
│  │ • LSTM (Derin Ogrenme)            │                  │
│  └───────────────────────────────────┘                  │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────┐                    │
│  │   Streamlit Dashboard           │                    │
│  │  (su_talebi_dashboard.py)       │                    │
│  ├─────────────────────────────────┤                    │
│  │  • Model Karsilastirmasi        │                    │
│  │  • Tahmin Gorsellestirilmesi    │                    │
│  │  • Senaryo Analizi              │                    │
│  │  • Su Kaybi Analizi             │                    │
│  └─────────────────────────────────┘                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Model Performans Karsilastirmasi

| Model | MAE | RMSE | R² | Açiklama |
|-------|-----|------|-----|----------|
| **Makine Öğrenmesi** | Düşük | Düşük | Yüksek | Random Forest, XGBoost, LightGBM kombinasyonu |
| **SARIMAX** | Orta | Orta | Orta | Zaman serisi modeli, mevsimsel desenleri yakalar |
| **GRU** | Orta-Yüksek | Orta-Yüksek | Orta | RNN tabanlı, uzun bağımlılıklar |
| **LSTM** | Orta-Yüksek | Orta-Yüksek | Orta | Gradient vanishing problemi çözen RNN varyantı |

*Not: Tam metrikler dashboard ve notebook'larda görüntülenebilir.*

## Baslica Dosyalar

- `su_talebi_dashboard.py`: Model sonuclarini ve ozet metrikleri gosteren Streamlit uygulamasi
- `notebooks/su_talebi_tahmini.ipynb`: Makine ogrenmesi tabanli tahmin sureci
- `notebooks/sarimax_su_tahminleri.ipynb`: SARIMAX modeli ve ileri tahminler
- `notebooks/gru_su_tahminleri.ipynb`: GRU modeli calismalari
- `notebooks/lstm_su_tahminleri.ipynb`: LSTM modeli calismalari
- `data/malatya_*.csv`: Projede kullanilan veri setleri

## Klasor Yapisi

```text
.
|- data/            # Veri setleri ve tahmin CSV ciktilari
|- notebooks/       # EDA, model egitimi ve tahmin notebook'lari
|- scripts/         # Yardimci scriptler
|- tests/           # Test/deneme scriptleri
|- su_talebi_dashboard.py
|- requirements.txt
|- README.md
```

## Kullanilan Teknolojiler

- Python
- pandas, numpy
- scikit-learn, xgboost, lightgbm
- tensorflow, keras
- matplotlib, seaborn, plotly
- jupyter
- streamlit

## Kurulum

1. Proje klasorune gecin.
2. Sanal ortam olusturun ve aktif edin.
3. Bagimliliklari yukleyin.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dashboard'i Calistirma

```powershell
streamlit run su_talebi_dashboard.py
```

Tarayici otomatik olarak `http://localhost:8501` adresinde açilacak ve modellerin karsilastirilmasi, tahminler ve senaryo analizleri gorsellestirilebilecektir.

## Hizli Baslangic Ornekleri

### 1. Dashboard'ı Calistirma

```powershell
# Sanal ortam aktif (yukarda yapildi)
streamlit run su_talebi_dashboard.py
```

### 2. Notebook'lari Calistirma

```powershell
# Makine ogrenmesi modeli egit
jupyter notebook notebooks/su_talebi_tahmini.ipynb

# SARIMAX modeli caliskan
jupyter notebook notebooks/sarimax_su_tahminleri.ipynb

# GRU/LSTM derin ogrenme modelleri
jupyter notebook notebooks/gru_su_tahminleri.ipynb
jupyter notebook notebooks/lstm_su_tahminleri.ipynb
```

### 3. Test Dosyalarini Calistirma

```powershell
# Kutuphanelerin kurulu oldugunu dogrula
python tests/test_libs.py

# Veri yukleme ve model pipeline testi
python tests/test_notebook_run.py

# Debug test
python tests/debug_test.py
```

## Proje Icindeki Temel Kavramlar

### Veri Seti
- **Zaman Araligi:** 20 yil gunluk veriler (yaklas. 7300 gözlem)
- **Hedef Degisken:** Gunluk su tuketimi
- **Dis Degiskenler:** Yagis, tarimsal alan, nufus, ekonomik göstergeler

### Model Strategileri
1. **Makine Ogrenmesi:** Ozelliklere dayali tahmin, tree-based modellerin kombinasyonu
2. **Zaman Serisi (SARIMAX):** Mevsimsel desenleri ve otoregressif dinamikleri yakala
3. **Derin Ogrenme (GRU/LSTM):** Sekans modelleri, uzun surekli ogrenmeler

### Senaryo Analizi
Dashboard'da iklim scenariosu, su kaybi orani ve talep degisikliklerinin etkisi analiz edilebilir.

## Notlar

- Buyuk model dosyalari ve yerel sanal ortam klasoru Git takibine dahil edilmedi.
- Proje, notebook tabanli analiz ve model denemeleri ile birlikte ilerleyen bir calisma yapisina sahiptir.
- GitHub repo icinde kod, notebook ve veri dosyalari bulunur; uretilmis agir model artefaktlari ayri tutulur.

## Git LFS Durumu

- Bu repoda buyuk model dosyalari (`*.joblib`, `*.pkl`, `*.h5`, `*.keras`) su an `.gitignore` ile disarida tutuluyor.
- Bu nedenle mevcut durumda Git LFS zorunlu degil.
- Gelecekte model dosyalarini repoya almak isterseniz LFS yapilandirmasi hazir: `.gitattributes` bu uzantilari LFS'e yonlendirir.