import json
from pathlib import Path

path = Path(r'c:\Users\CASPER\OneDrive\Masaüstü\Proje Pazarı\adim_adim_veri_analizi.ipynb')

nb = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Adım Adım Veri Analizi\n',
                'Bu not defteri, yüklenen Malatya su, yağış, iklim ve sosyoekonomik veri setleri için adım adım veri analizi işlemlerini açık ve karışıklığa yer vermeden gösterir.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import pandas as pd\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'from pathlib import Path\n',
                'from io import StringIO\n',
                '\n',
                "sns.set(style='whitegrid', palette='muted', font_scale=1.05)\n",
                '\n',
                "csv_files = sorted(Path('.').glob('*.csv'))\n",
                "print('CSV dosyaları:')\n",
                "for f in csv_files:\n",
                "    print('-', f.name)\n"
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 1. Veri Yükleme\n',
                'Her bir CSV dosyasını pandas ile okuyacağız. Bazı dosyalarda tüm satırları tırnak içinde gösteren format var; bu yüzden bu tür durumlar için özel bir yükleme fonksiyonu kullanıyoruz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'def read_csv_robust(path):\n',
                '    text = Path(path).read_text(encoding=\'utf-8\').strip()\n',
                '    if text.startswith(\'\"\') and \'\\n\"\' in text:\n',
                '        text = \'\\n\'.join(line.strip(\'\"\') for line in text.splitlines())\n',
                '    return pd.read_csv(StringIO(text), sep=\',\', decimal=\'.\', encoding=\'utf-8\')\n',
                '\n',
                'raw_data = {}\n',
                'for csv_path in csv_files:\n',
                '    df = read_csv_robust(csv_path)\n',
                '    raw_data[csv_path.name] = df\n',
                '\n',
                'raw_data.keys()\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 2. Veri Keşfi\n',
                'Her bir veri setinin boyutunu, sütunlarını, veri tiplerini ve ilk satırlarını inceleyeceğiz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'for name, df in raw_data.items():\n',
                "    print('\\n---', name, '---')\n",
                "    print('Boyut:', df.shape)\n",
                "    print('\\nSütunlar:')\n",
                '    print(df.columns.tolist())\n',
                "    print('\\nİlk 3 satır:')\n",
                '    display(df.head(3))\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'for name, df in raw_data.items():\n',
                "    print('\\n---', name, '---')\n",
                '    print(df.dtypes)\n',
                "    print('Eksik değer sayısı:')\n",
                '    print(df.isnull().sum())\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 3. Eksik Değerlerin Yönetimi\n',
                'Eksik değer varsa belirleyeceğiz ve eğer makul ise nasıl doldurulabileceğini göstereceğiz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'missing_summary = pd.DataFrame(\n',
                '    [(name, df.isnull().sum().sum(), df.isnull().sum().to_dict()) for name, df in raw_data.items()],\n',
                "    columns=['Veri Seti', 'Toplam Eksik', 'Sütun Bazlı Eksik']\n",
                ')\n',
                'missing_summary\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                'Veri setlerinizde şu anda ekran çıktısına göre doğrudan eksik kayıt görünmüyor. Eğer varsa, aşağıdaki yöntemlerle işleyebiliriz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'cleaned_data = {}\n',
                'for name, df in raw_data.items():\n',
                '    df_clean = df.copy()\n',
                '    df_clean = df_clean.dropna()  # eksik satırları atar\n',
                '    cleaned_data[name] = df_clean\n',
                "    print(name, '->', df.shape, 'boyutundan', df_clean.shape, 'boyuta düştü')\n"
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 4. Yinelenen Kayıtların Kontrolü ve Temizlenmesi\n',
                'Tekrarlanan satır varsa bunları sileriz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'for name, df in cleaned_data.items():\n',
                '    duplicate_count = df.duplicated().sum()\n',
                "    print(name, 'tekrarlanan satır sayısı:', duplicate_count)\n",
                '    if duplicate_count > 0:\n',
                '        cleaned_data[name] = df.drop_duplicates()\n',
                "        print('  Düzeltilmiş boyut:', cleaned_data[name].shape)\n"
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 5. Veri Tipi Dönüşümü\n',
                'Tarih sütunlarını gerçek `datetime` tipine çeviriyoruz ve sayısal olmayan sütunları sayısal tipe dönüştürüyoruz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'def convert_types(df):\n',
                '    out = df.copy()\n',
                "    if 'Tarih' in out.columns:\n",
                "        out['Tarih'] = pd.to_datetime(out['Tarih'], errors='coerce')\n",
                "    if 'Yil' in out.columns and out['Yil'].dtype == object:\n",
                "        out['Yil'] = pd.to_numeric(out['Yil'], errors='coerce').astype('Int64')\n",
                '    for col in out.columns:\n',
                "        if out[col].dtype == object and col not in ['Tarih', 'Bolge_Tipi']:\n",
                "            out[col] = pd.to_numeric(out[col].str.replace(',', '.'), errors='ignore')\n",
                '    return out\n',
                '\n',
                'typed_data = {name: convert_types(df) for name, df in cleaned_data.items()}\n',
                'for name, df in typed_data.items():\n',
                "    print('\\n---', name, '---')\n",
                '    print(df.dtypes)\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 6. Tanımlayıcı İstatistikler\n',
                'Sayısal sütunlar için özet istatistikleri hesaplayarak verinin dağılımını ve merkezi eğilim değerlerini görürüz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'for name, df in typed_data.items():\n',
                "    print('\\n===', name, '===')\n",
                '    display(df.describe(include=\'all\'))\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 7. Veri Görselleştirme\n',
                'Su tüketimi ve yağış verilerini çizerek zaman içindeki değişimi görselleştireceğiz.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plt.figure(figsize=(14, 5))\n',
                "if 'malatya_gunluk_su_tuketimi_20yil.csv' in typed_data:\n",
                "    df = typed_data['malatya_gunluk_su_tuketimi_20yil.csv']\n",
                "    df = df.dropna(subset=['Tarih', 'Su_Tuketimi_m3'])\n",
                "    df.plot(x='Tarih', y='Su_Tuketimi_m3', title='Günlük Su Tüketimi (Malatya)', legend=False)\n",
                "    plt.xlabel('Tarih')\n",
                "    plt.ylabel('Su Tüketimi (m3)')\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "if 'malatya_gunluk_yagis_20yil.csv' in typed_data:\n",
                "    df = typed_data['malatya_gunluk_yagis_20yil.csv']\n",
                "    df = df.dropna(subset=['Tarih', 'Genel_Ortalama_mm'])\n",
                "    df.plot(x='Tarih', y='Genel_Ortalama_mm', title='Günlük Yağış Ortalaması (Malatya)', legend=False)\n",
                "    plt.xlabel('Tarih')\n",
                "    plt.ylabel('Yağış (mm)')\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n"
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 8. Korelasyon Analizi\n',
                'Sayısal sütunlar arasındaki ilişkileri inceleyerek hangi değişkenlerin birlikte hareket ettiğine bakalım.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                "sample_name = 'malatya_gunluk_su_tuketimi_20yil.csv'\n",
                'if sample_name in typed_data:\n',
                '    df = typed_data[sample_name].select_dtypes(include=[\'number\']).copy()\n',
                '    corr = df.corr()\n',
                '    plt.figure(figsize=(8, 6))\n',
                "    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)\n",
                "    plt.title(sample_name + ' - Sayısal Değişken Korelasyonları')\n",
                '    plt.tight_layout()\n',
                '    plt.show()\n',
                'else:\n',
                "    print('Korelasyon için örnek veri seti yüklenemedi.')\n"
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.13.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
