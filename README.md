# Malatya Su Talebi Projesi

Bu proje, Malatya ili için su talebini veri analizi, klasik makine ogrenmesi, zaman serisi modelleme ve derin ogrenme yaklasimlariyla incelemek icin hazirlandi. Calisma alani; veri setleri, egitim notebook'lari, kaydedilmis model artefaktlari ve sonuclari gorsellestiren bir Streamlit dashboard'i icerir.

## Proje Kapsami

- Malatya'ya ait tuketim, yagis, nufus, sosyoekonomik ve tarimsal veri setlerinin analizi
- Makine ogrenmesi modelleri ile su talebi tahmini
- SARIMAX ile zaman serisi tahmini
- GRU ve LSTM tabanli derin ogrenme modelleri
- Sonuclari karsilastiran interaktif Streamlit dashboard'i

## Baslica Dosyalar

- `su_talebi_dashboard.py`: Model sonuclarini ve ozet metrikleri gosteren Streamlit uygulamasi
- `su_talebi_tahmini.ipynb`: Makine ogrenmesi tabanli tahmin sureci
- `sarimax_su_tahminleri.ipynb`: SARIMAX modeli ve ileri tahminler
- `gru_su_tahminleri.ipynb`: GRU modeli calismalari
- `lstm_su_tahminleri.ipynb`: LSTM modeli calismalari
- `malatya_*.csv`: Projede kullanilan veri setleri

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

## Notlar

- Buyuk model dosyalari ve yerel sanal ortam klasoru Git takibine dahil edilmedi.
- Proje, notebook tabanli analiz ve model denemeleri ile birlikte ilerleyen bir calisma yapisina sahiptir.
- GitHub repo icinde kod, notebook ve veri dosyalari bulunur; uretilmis agir model artefaktlari ayri tutulur.