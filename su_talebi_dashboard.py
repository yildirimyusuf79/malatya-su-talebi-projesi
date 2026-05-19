import time
import importlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = Warning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BASE_DIR = Path(__file__).resolve().parent

MODEL_SPECS = {
    "Makine Öğrenmesi Modelleri": {
        "kind": "feature_bundle",
        "artifact": "su_talebi_modeli.joblib",
        "notebook": "notebooks/su_talebi_tahmini.ipynb",
        "description": "Linear Regression, Random Forest, XGBoost ve LightGBM karşılaştırması içinden seçilen en iyi model.",
    },
    "SARIMAX": {
        "kind": "sarimax",
        "artifact": "sarimax_su_tahmin_modeli.pkl",
        "info_file": "sarimax_model_info.joblib",
        "forecast_file": "data/sarimax_gelecek_30gun_tahmin.csv",
        "notebook": "notebooks/sarimax_su_tahminleri.ipynb",
        "description": "Dışsal değişkenli mevsimsel ARIMA modeli. Notebook'taki metrikler ve ileri tahmin artefaktları gösterilir.",
    },
    "GRU": {
        "kind": "sequence",
        "artifact": "gru_su_tahmin_modeli.keras",
        "scaler_file": "gru_scaler.joblib",
        "info_file": "gru_model_info.joblib",
        "notebook": "notebooks/gru_su_tahminleri.ipynb",
        "description": "60 günlük pencere kullanan GRU tabanlı zaman serisi modeli.",
    },
    "LSTM": {
        "kind": "sequence",
        "artifact": "lstm_su_tahmin_modeli.h5",
        "scaler_file": "lstm_scaler.joblib",
        "info_file": "lstm_model_info.joblib",
        "notebook": "notebooks/lstm_su_tahminleri.ipynb",
        "description": "60 günlük pencere kullanan LSTM tabanlı zaman serisi modeli.",
    },
}

MODEL_THEMES = {
    "Makine Öğrenmesi Modelleri": {
        "accent": "#4CC9F0",
        "secondary": "#90E0EF",
        "sidebar": "linear-gradient(180deg, #021526 0%, #03346E 100%)",
        "background": "linear-gradient(180deg, #021526 0%, #03346E 45%, #6EACDA 100%)",
    },
    "SARIMAX": {
        "accent": "#F4A261",
        "secondary": "#E9C46A",
        "sidebar": "linear-gradient(180deg, #2B2D42 0%, #6D597A 100%)",
        "background": "linear-gradient(180deg, #1F1C2C 0%, #3B2F63 45%, #C9ADA7 100%)",
    },
    "GRU": {
        "accent": "#06D6A0",
        "secondary": "#7AE582",
        "sidebar": "linear-gradient(180deg, #081C15 0%, #1B4332 100%)",
        "background": "linear-gradient(180deg, #081C15 0%, #1B4332 45%, #52B788 100%)",
    },
    "LSTM": {
        "accent": "#FF6B6B",
        "secondary": "#FFD166",
        "sidebar": "linear-gradient(180deg, #2B0A3D 0%, #6A0572 100%)",
        "background": "linear-gradient(180deg, #1B1028 0%, #4A266A 45%, #D17B88 100%)",
    },
}

st.set_page_config(page_title="Su Talebi Dashboard", layout="wide")

theme = MODEL_THEMES["Makine Öğrenmesi Modelleri"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Rajdhani', sans-serif;
    }}

    .stApp {{
        background: {theme['background']};
    }}

    [data-testid="stHeader"] {{
        background: rgba(2, 21, 38, 0.35);
    }}

    [data-testid="stSidebar"] {{
        background: {theme['sidebar']};
    }}

    h1 {{
        background: -webkit-linear-gradient(45deg, #f39c12, #d35400);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
    }}

    [data-testid="stMetric"] {{
        background: rgba(20, 25, 35, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 15px 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-5px);
        border-color: rgba(211, 84, 0, 0.5);
    }}

    [data-testid="stMetricValue"] {{
        color: #ff6b6b !important;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.4);
        font-weight: 700;
    }}

    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    .main-title {{
        text-align: center;
        color: #E2F1FF;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        text-align: center;
        color: #C9E7FF;
        font-size: 1.02rem;
        margin-bottom: 0.25rem;
    }}
    .meta {{
        text-align: center;
        color: #D8EEFF;
        font-size: 0.98rem;
    }}
    .model-chip {{
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);
        color: #F8FBFF;
        margin-right: 0.45rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }}
    .best-model-card {{
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.14);
        color: #F4FAFF;
        margin-bottom: 1rem;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.14);
    }}
    .best-model-title {{
        color: {theme['secondary']};
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}
    .best-model-name {{
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }}
    .info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.9rem;
        margin-top: 0.65rem;
        margin-bottom: 1rem;
    }}
    .info-card {{
        border-radius: 16px;
        padding: 1rem;
        background: linear-gradient(145deg, rgba(8, 14, 28, 0.82), rgba(20, 29, 45, 0.62));
        border: 1px solid rgba(144, 192, 255, 0.22);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
    }}
    .info-card-title {{
        color: #D6EBFF;
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 0.35rem;
    }}
    .info-card-text {{
        color: #BFDFFF;
        font-size: 0.95rem;
        line-height: 1.45;
        margin-bottom: 0;
    }}
    .model-pill {{
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        color: #F4FAFF;
        border: 1px solid rgba(255,255,255,0.22);
        background: rgba(255,255,255,0.08);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


def style_time_series_figure(fig, y_axis_title: str):
    fig.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(showspikes=True, spikecolor="#8EC9FF"),
        yaxis_title=y_axis_title,
    )
    fig.update_traces(
        hovertemplate="<b>Tarih:</b> %{x|%d.%m.%Y}<br><b>%{fullData.name}:</b> %{y:,.2f}<extra></extra>"
    )
    return fig


@st.cache_data
def load_data() -> pd.DataFrame:
    df_su = pd.read_csv(BASE_DIR / "data/malatya_gunluk_su_tuketimi_20yil.csv")
    df_yagis = pd.read_csv(BASE_DIR / "data/malatya_gunluk_yagis_20yil.csv")

    def with_date_col(df: pd.DataFrame, label: str) -> pd.DataFrame:
        date_cols = [c for c in df.columns if "tarih" in c.lower() or "date" in c.lower()]
        if not date_cols:
            raise ValueError(f"{label} veri setinde tarih sütunu bulunamadı")
        out = df.copy()
        out["date"] = pd.to_datetime(out[date_cols[0]], errors="coerce")
        return out

    sudf = with_date_col(df_su, "Su")
    yagisdf = with_date_col(df_yagis, "Yağış")

    rain_cols_to_add = [c for c in yagisdf.columns if c != "date" and c not in sudf.columns]
    merged = sudf.merge(yagisdf[["date"] + rain_cols_to_add], on="date", how="inner")

    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["day"] = merged["date"].dt.day
    merged["weekday"] = merged["date"].dt.weekday

    # Tarımsal arazi ve sulu tarım verilerini yıllık dışsal değişken olarak entegre et.
    try:
        tarim = pd.read_csv(BASE_DIR / "data/malatya_bolgesel_tarim_arazileri_2004_2023.csv")
        yil_col = next((c for c in tarim.columns if c.lower() in {"yil", "year"}), None)
        toplam_col = next((c for c in tarim.columns if "Toplam_Tarim_Alani" in c), None)
        sulu_col = next((c for c in tarim.columns if "Sulu_Tarim_Alani" in c), None)
        damla_col = next((c for c in tarim.columns if "Damla_Sulama_Orani" in c), None)

        if yil_col and toplam_col and sulu_col:
            tarim_yillik = (
                tarim.groupby(yil_col, as_index=False)
                .agg(
                    Toplam_Tarim_Alani_Dekar=(toplam_col, "sum"),
                    Sulu_Tarim_Alani_Dekar=(sulu_col, "sum"),
                )
                .rename(columns={yil_col: "year"})
            )

            if damla_col:
                damla = tarim.groupby(yil_col, as_index=False)[damla_col].mean().rename(
                    columns={yil_col: "year", damla_col: "Damla_Sulama_Orani"}
                )
                tarim_yillik = tarim_yillik.merge(damla, on="year", how="left")
            else:
                tarim_yillik["Damla_Sulama_Orani"] = np.nan

            merged = merged.merge(tarim_yillik, on="year", how="left")
            for c in ["Toplam_Tarim_Alani_Dekar", "Sulu_Tarim_Alani_Dekar", "Damla_Sulama_Orani"]:
                merged[c] = merged[c].ffill().bfill()
    except FileNotFoundError:
        pass

    return merged.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_model_bundle(model_file: str) -> dict:
    return joblib.load(BASE_DIR / model_file)


@st.cache_resource
def load_sequence_artifacts(model_name: str):
    spec = MODEL_SPECS[model_name]
    tf = importlib.import_module("tensorflow")
    model = tf.keras.models.load_model(BASE_DIR / spec["artifact"], compile=False)
    scaler = joblib.load(BASE_DIR / spec["scaler_file"])
    return model, scaler


@st.cache_data
def load_model_info(model_name: str) -> dict:
    spec = MODEL_SPECS[model_name]

    if spec["kind"] == "feature_bundle":
        bundle = load_model_bundle(spec["artifact"])
        return {
            "feature_count": len(bundle["features"]),
            "artifacts": [spec["artifact"]],
            "metrics": {},
        }

    info = joblib.load(BASE_DIR / spec["info_file"])
    metrics = {}

    if spec["kind"] == "sarimax":
        raw_metrics = info.get("metrics", {})
        metrics = {
            "MAE": float(raw_metrics.get("MAE", np.nan)),
            "RMSE": float(raw_metrics.get("RMSE", np.nan)),
            "R2": float(raw_metrics.get("R2", np.nan)),
            "MAPE": float(raw_metrics.get("MAPE", np.nan)),
        }
    elif model_name == "GRU":
        raw_metrics = info.get("metrics", {}).get("Test", {})
        metrics = {
            "MAE": float(raw_metrics.get("MAE", np.nan)),
            "RMSE": float(raw_metrics.get("RMSE", np.nan)),
            "R2": float(raw_metrics.get("R2", np.nan)),
            "MAPE": float(raw_metrics.get("MAPE", np.nan)),
        }
    elif model_name == "LSTM":
        info.setdefault("look_back", 60)
        metrics = {
            "MAE": float(info.get("test_mae", np.nan)),
            "RMSE": float(info.get("test_rmse", np.nan)),
            "R2": float(info.get("test_r2", np.nan)),
            "MAPE": np.nan,
        }

    artifacts = [spec["artifact"], spec["info_file"]]
    if "scaler_file" in spec:
        artifacts.append(spec["scaler_file"])
    if "forecast_file" in spec:
        artifacts.append(spec["forecast_file"])

    info["metrics"] = metrics
    info["artifacts"] = artifacts
    return info


@st.cache_data
def load_future_forecast_csv(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / file_name)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def load_leakage_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "data/malatya_kayip_kacak_20yil.csv")
    date_col = next((c for c in df.columns if c.lower() in {"tarih", "date"}), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    return df


@st.cache_data
def load_climate_scenario_data() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "data/malatya_iklim_senaryosu_2024_2053.csv")
    date_col = next((c for c in df.columns if c.lower() in {"tarih", "date"}), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if "Yil" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"Yil": "year"})
    if "Ay" in df.columns and "month" not in df.columns:
        df = df.rename(columns={"Ay": "month"})
    return df


def build_reservoir_optimization_frame(
    climate_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    leakage_reduction_pp: float,
    demand_saving_pct: float,
    demand_pressure_pct: float,
) -> pd.DataFrame:
    cdf = climate_df[(climate_df["year"] >= start_year) & (climate_df["year"] <= end_year)].copy()
    if cdf.empty:
        return pd.DataFrame()

    monthly_leak = leakage_df.groupby("month", as_index=False)["Kayip_Kacak_Orani_%"].mean()
    cdf = cdf.merge(monthly_leak, on="month", how="left")
    cdf["Kayip_Kacak_Orani_%"] = cdf["Kayip_Kacak_Orani_%"].fillna(float(leakage_df["Kayip_Kacak_Orani_%"].mean()))

    base_fill = cdf["Genel_Ortalama"].astype(float)
    rain_mean = float(cdf["Toplam_Yagis"].mean())
    rain_std = float(cdf["Toplam_Yagis"].std()) if float(cdf["Toplam_Yagis"].std()) > 1e-6 else 1.0
    rain_score = ((cdf["Toplam_Yagis"] - rain_mean) / rain_std).clip(-2, 2)

    leakage_gain = 0.40 * leakage_reduction_pp
    net_demand_saving = max(0.0, demand_saving_pct - max(0.0, demand_pressure_pct) * 0.25)
    demand_gain = 0.25 * net_demand_saving
    climate_gain = 2.2 * rain_score

    cdf["Baz_Doluluk_%"] = base_fill.clip(0, 100)
    cdf["Iyilestirilmis_Doluluk_%"] = (base_fill + leakage_gain + demand_gain + climate_gain).clip(0, 100)
    cdf["Doluluk_Kazanimi_%"] = cdf["Iyilestirilmis_Doluluk_%"] - cdf["Baz_Doluluk_%"]

    return cdf


@st.cache_data
def build_model_comparison_table() -> pd.DataFrame:
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        info = load_model_info(model_name)
        metrics = info.get("metrics", {})
        rows.append(
            {
                "Model": model_name,
                "Tür": spec["kind"],
                "MAE": metrics.get("MAE", np.nan),
                "RMSE": metrics.get("RMSE", np.nan),
                "R2": metrics.get("R2", np.nan),
                "MAPE": metrics.get("MAPE", np.nan),
            }
        )
    return pd.DataFrame(rows)


def build_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = df.copy()
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names].copy()


def build_sequence_prediction_frame(
    source_df: pd.DataFrame,
    sequence_model,
    scaler,
    look_back: int,
) -> pd.DataFrame:
    history = source_df[["date", "Su_Tuketimi_m3"]].dropna().sort_values("date").reset_index(drop=True)
    history["prediction"] = np.nan

    if len(history) <= look_back:
        return history

    values = history["Su_Tuketimi_m3"].astype(float).to_numpy().reshape(-1, 1)
    scaled = scaler.transform(values).reshape(-1)
    sequences = np.array([scaled[i - look_back : i] for i in range(look_back, len(scaled))], dtype=np.float32)
    sequences = sequences.reshape(-1, look_back, 1)

    pred_scaled = sequence_model.predict(sequences, verbose=0).reshape(-1, 1)
    pred_values = scaler.inverse_transform(pred_scaled).reshape(-1)
    history.loc[look_back:, "prediction"] = pred_values

    return history


def generate_sequence_forecast(
    source_df: pd.DataFrame,
    sequence_model,
    scaler,
    look_back: int,
    days: int,
) -> pd.DataFrame:
    history = source_df[["date", "Su_Tuketimi_m3"]].dropna().sort_values("date").reset_index(drop=True)
    if len(history) < look_back:
        return pd.DataFrame(columns=["date", "prediction"])

    scaled_history = scaler.transform(history[["Su_Tuketimi_m3"]].astype(float)).reshape(-1).tolist()
    future_scaled = []

    for _ in range(days):
        window = np.array(scaled_history[-look_back:], dtype=np.float32).reshape(1, look_back, 1)
        next_scaled = float(sequence_model.predict(window, verbose=0).reshape(-1)[0])
        scaled_history.append(next_scaled)
        future_scaled.append(next_scaled)

    future_values = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).reshape(-1)
    future_dates = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=days, freq="D")
    return pd.DataFrame({"date": future_dates, "prediction": future_values})


def build_risk_fields(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    rain_cols = [
        c
        for c in [
            "Karakaya_Yagis_mm",
            "Surgu_Yagis_mm",
            "Sultansuyu_Yagis_mm",
            "Genel_Ortalama_mm",
        ]
        if c in df.columns
    ]
    rain_mean = df[rain_cols].mean(axis=1) if rain_cols else pd.Series(0, index=df.index)
    temp = df["Ort_Sicaklik"] if "Ort_Sicaklik" in df.columns else pd.Series(0, index=df.index)

    out = pd.DataFrame(index=df.index)
    out["rain_mean"] = rain_mean
    out["temp"] = temp
    out["low_rain"] = rain_mean < 5
    out["high_temp"] = temp > 28
    out["high_demand"] = df["prediction"] >= threshold

    out["pest_risk"] = np.where(out["low_rain"] & out["high_temp"], "Yüksek", "Normal")
    out["disease_risk"] = np.where(out["low_rain"] & (temp > 30), "Yüksek", "Düşük")

    return out


def summarize_risk(risk_df: pd.DataFrame) -> str:
    ratio = risk_df["high_demand"].mean()
    if ratio > 0.25 or (risk_df["pest_risk"] == "Yüksek").any() or (risk_df["disease_risk"] == "Yüksek").any():
        return "Yüksek"
    if ratio > 0.10:
        return "Orta"
    return "Düşük"


def build_irrigation_plan_frame(
    filtered_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    if not scenario_df.empty and "prediction" in scenario_df.columns:
        plan = scenario_df[["date", "prediction"]].copy().sort_values("date")
    else:
        fallback = filtered_df[["date", "prediction", "Su_Tuketimi_m3"]].copy().sort_values("date")
        fallback["prediction"] = np.where(
            fallback["prediction"].notna(), fallback["prediction"], fallback["Su_Tuketimi_m3"]
        )
        plan = fallback[["date", "prediction"]].tail(30).copy()

    if "rain_mean" in scenario_df.columns:
        rain_source = scenario_df[["date", "rain_mean"]].copy()
        plan = plan.merge(rain_source, on="date", how="left")
    else:
        rain_avg = float(filtered_df["rain_mean"].tail(14).mean()) if "rain_mean" in filtered_df.columns else 0.0
        plan["rain_mean"] = rain_avg

    total_area = float(filtered_df["Toplam_Tarim_Alani_Dekar"].mean()) if "Toplam_Tarim_Alani_Dekar" in filtered_df.columns else np.nan
    irrigated_area = float(filtered_df["Sulu_Tarim_Alani_Dekar"].mean()) if "Sulu_Tarim_Alani_Dekar" in filtered_df.columns else np.nan
    if not np.isfinite(irrigated_area) or irrigated_area <= 0:
        irrigated_area = np.nan

    plan["Sulama_Ihtiyaci_m3_dekar"] = plan["prediction"] / irrigated_area if np.isfinite(irrigated_area) else np.nan
    plan["Oncelik"] = np.where(plan["prediction"] >= threshold, "Yüksek", "Orta")
    plan["Yagis_Etkisi"] = np.select(
        [plan["rain_mean"] >= 8, plan["rain_mean"] <= 3],
        ["Azalt", "Artır"],
        default="Dengele",
    )

    if np.isfinite(total_area) and total_area > 0 and np.isfinite(irrigated_area):
        plan["Sulu_Alan_Orani_%"] = 100 * irrigated_area / total_area
    else:
        plan["Sulu_Alan_Orani_%"] = np.nan

    return plan


def generate_scenario_forecast(
    source_df: pd.DataFrame,
    model,
    scaler,
    feature_names: list[str],
    days: int,
    temp_shift: float,
    rain_scale: float,
) -> pd.DataFrame:
    history = source_df.sort_values("date").copy()
    if history.empty:
        return pd.DataFrame(columns=["date", "prediction", "scenario_temp", "rain_mean"])

    base_rows = history.tail(min(60, len(history))).copy()
    repeat_count = int(np.ceil(days / max(len(base_rows), 1)))
    scenario = pd.concat([base_rows] * repeat_count, ignore_index=True).head(days)

    future_dates = pd.date_range(history["date"].max() + pd.Timedelta(days=1), periods=days, freq="D")
    scenario["date"] = future_dates
    scenario["year"] = scenario["date"].dt.year
    scenario["month"] = scenario["date"].dt.month
    scenario["day"] = scenario["date"].dt.day
    scenario["weekday"] = scenario["date"].dt.weekday

    if "Ort_Sicaklik" in scenario.columns:
        scenario["Ort_Sicaklik"] = scenario["Ort_Sicaklik"] + temp_shift
    if "temp" in scenario.columns:
        scenario["temp"] = scenario["temp"] + temp_shift

    rain_cols = [
        c
        for c in ["Karakaya_Yagis_mm", "Surgu_Yagis_mm", "Sultansuyu_Yagis_mm", "Genel_Ortalama_mm", "rain_mean"]
        if c in scenario.columns
    ]
    for col in rain_cols:
        scenario[col] = np.clip(scenario[col] * rain_scale, 0, None)

    X_future = build_feature_frame(scenario, feature_names)
    future_scaled = scaler.transform(X_future)
    future_scaled_df = pd.DataFrame(future_scaled, columns=feature_names, index=X_future.index)
    scenario_pred = model.predict(future_scaled_df)

    if "Ort_Sicaklik" in scenario.columns:
        scenario_temp = scenario["Ort_Sicaklik"]
    elif "temp" in scenario.columns:
        scenario_temp = scenario["temp"]
    else:
        scenario_temp = pd.Series(0.0, index=scenario.index)

    if rain_cols:
        scenario_rain_mean = scenario[rain_cols].mean(axis=1)
    else:
        scenario_rain_mean = pd.Series(0.0, index=scenario.index)

    return pd.DataFrame(
        {
            "date": future_dates,
            "prediction": scenario_pred,
            "scenario_temp": scenario_temp.values,
            "rain_mean": scenario_rain_mean.values,
        }
    )


try:
    data = load_data()
except FileNotFoundError as exc:
    st.error(f"Gerekli dosya bulunamadı: {exc}")
    st.stop()

st.sidebar.header("Model ve Senaryo")
HOME_OPTION = "Model seçiniz..."
selected_model_name = st.sidebar.selectbox(
    "Model Seçimi",
    [HOME_OPTION] + list(MODEL_SPECS.keys()),
    index=0,
    key="model_selection",
)

if selected_model_name == HOME_OPTION:
    comparison_df = build_model_comparison_table()
    if comparison_df["RMSE"].notna().any():
        best_model_row = comparison_df.sort_values("RMSE", ascending=True).iloc[0]
    else:
        best_model_row = pd.Series({"Model": "-", "RMSE": np.nan, "R2": np.nan, "MAE": np.nan})

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 14% 10%, rgba(39, 101, 191, 0.26), transparent 36%),
                radial-gradient(circle at 86% 18%, rgba(197, 107, 35, 0.18), transparent 33%),
                linear-gradient(165deg, #05080F 0%, #0A1325 52%, #0D1C33 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #04070F 0%, #0A1628 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 class="main-title">İklim Değişikliklerinin Kuraklık ve Yağış<br>Rejimleri Üzerindeki Etkilerinin Zaman Serisi<br>ile İncelenmesi</h1>
        <p class="meta"><strong>Danışman :</strong> Doç. Dr. Canan Batur Şahin - Mühendislik ve Doğa Bilimleri Fakültesi</p>
        <p class="meta"><strong>Yusuf Yıldırım</strong> - Yazılım Mühendisliği</p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        f"""
        <div class="best-model-card">
            <div class="best-model-title">En İyi Model Özeti</div>
            <div class="best-model-name">{best_model_row['Model']}</div>
            <div>RMSE: {'-' if pd.isna(best_model_row['RMSE']) else f"{best_model_row['RMSE']:,.0f}"} | R2: {'-' if pd.isna(best_model_row['R2']) else f"{best_model_row['R2']:.3f}"} | MAE: {'-' if pd.isna(best_model_row['MAE']) else f"{best_model_row['MAE']:,.0f}"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Doğruluk Metrikleri Ne Anlatır?")
    st.markdown(
        """
        <div class="info-grid">
            <div class="info-card">
                <div class="info-card-title">MAE (Mean Absolute Error)</div>
                <p class="info-card-text"><strong>Tanım:</strong> Tahmin hatalarının mutlak değer ortalamasıdır.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Yorum:</strong> Değer küçüldükçe model daha iyi tahmin yapar. Birimi doğrudan m³ olduğu için yorumlaması kolaydır.</p>
            </div>
            <div class="info-card">
                <div class="info-card-title">RMSE (Root Mean Squared Error)</div>
                <p class="info-card-text"><strong>Tanım:</strong> Hataların karesinin ortalamasının kareköküdür.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Yorum:</strong> Büyük hataları daha fazla cezalandırır. Ani sapmaları yakalamada MAE'ye göre daha hassastır.</p>
            </div>
            <div class="info-card">
                <div class="info-card-title">R2 (Belirleme Katsayısı)</div>
                <p class="info-card-text"><strong>Tanım:</strong> Modelin verideki değişkenliği ne oranda açıkladığını gösterir.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Yorum:</strong> 1'e yaklaştıkça daha iyi; 0 civarı zayıf, negatif değerler ise modelin ortalamadan bile kötü olduğunu gösterebilir.</p>
            </div>
            <div class="info-card">
                <div class="info-card-title">MAPE (Mean Absolute Percentage Error)</div>
                <p class="info-card-text"><strong>Tanım:</strong> Hatanın yüzde cinsinden ortalamasıdır.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Yorum:</strong> Modelleri farklı ölçeklerde karşılaştırmayı kolaylaştırır. Küçük değer daha iyi performans anlamına gelir.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-grid">
            <div class="info-card">
                <div class="info-card-title">Makine Öğrenmesi Modelleri Nedir?</div>
                <p class="info-card-text"><strong>Nasıl çalışır:</strong> Regresyon tabanlı yaklaşımda birden fazla algoritma (Linear Regression, Random Forest, XGBoost, LightGBM) eğitilir ve en iyi performans veren model seçilir.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>İçerdiği veriler:</strong> Sıcaklık, yağış, takvim değişkenleri (ay, gün, hafta içi/sonu) ve su tüketimi ilişkilerini birlikte kullanır.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Kullanım amacı:</strong> Açıklanabilir ve hızlı günlük talep tahmini üretmek.</p>
            </div>
            <div class="info-card">
                <div class="info-card-title">SARIMAX Nedir?</div>
                <p class="info-card-text"><strong>Nasıl çalışır:</strong> Mevsimsellik ve trendi zaman serisi denklemi ile öğrenir; dışsal değişkenleri (X) aynı anda modele dahil eder.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>İçerdiği veriler:</strong> Geçmiş su tüketimi + yağış/sıcaklık gibi dışsal iklim değişkenleri.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Kullanım amacı:</strong> Mevsimsel örüntüleri güçlü şekilde yakalayarak kısa-orta vadeli tahmin üretmek.</p>
            </div>
            <div class="info-card">
                <div class="info-card-title">GRU Nedir?</div>
                <p class="info-card-text"><strong>Nasıl çalışır:</strong> Gated Recurrent Unit yapısı, geçmiş adımlardan gelen bilgiyi kapı mekanizmalarıyla tutar veya unutur; bu sayede zaman bağımlılığını öğrenir.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>İçerdiği veriler:</strong> Ölçeklenmiş tüketim serisinden oluşturulan 60 günlük pencere dizileri.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Kullanım amacı:</strong> Doğrusal olmayan dalgalanmaları ve ardışık örüntüleri yakalamak.</p>
            </div>
            <div class="info-card">
                <div class="info-card-title">LSTM Nedir?</div>
                <p class="info-card-text"><strong>Nasıl çalışır:</strong> Uzun-kısa dönemli bellek hücreleri ile geçmişteki önemli bilgiyi daha uzun süre korur; özellikle uzun bağımlılıklarda etkilidir.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>İçerdiği veriler:</strong> GRU gibi 60 günlük zaman pencereleri ve MinMax ölçeklenmiş tüketim serisi.</p>
                <p class="info-card-text" style="margin-top:0.45rem;"><strong>Kullanım amacı:</strong> Uzun dönemli desenlerin etkili olduğu tüketim değişimlerini modellemek.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### İlişkisel Veri Analizi")

    home_df = data.copy()
    rain_cols = [
        c
        for c in ["Karakaya_Yagis_mm", "Surgu_Yagis_mm", "Sultansuyu_Yagis_mm", "Genel_Ortalama_mm"]
        if c in home_df.columns
    ]
    if rain_cols:
        home_df["rain_mean"] = home_df[rain_cols].mean(axis=1)
    else:
        home_df["rain_mean"] = 0.0

    if "Ort_Sicaklik" in home_df.columns:
        home_df["temp"] = home_df["Ort_Sicaklik"]
    else:
        home_df["temp"] = np.nan

    home_df["week_part"] = np.where(home_df["weekday"] >= 5, "Hafta Sonu", "Hafta İçi")

    rel_left, rel_right = st.columns(2)

    with rel_left:
        scatter_df = home_df.dropna(subset=["temp", "Su_Tuketimi_m3", "rain_mean"]).copy()
        fig_temp_demand = px.scatter(
            scatter_df,
            x="temp",
            y="Su_Tuketimi_m3",
            color="rain_mean",
            title="Sıcaklık - Su Talebi İlişkisi (renk: yağış)",
            labels={
                "temp": "Ortalama Sıcaklık (°C)",
                "Su_Tuketimi_m3": "Su Tüketimi (m³)",
                "rain_mean": "Ortalama Yağış (mm)",
            },
            template=PLOTLY_TEMPLATE,
            opacity=0.6,
            color_continuous_scale="Turbo",
        )
        st.plotly_chart(fig_temp_demand, width='stretch', config=PLOTLY_CONFIG)

    with rel_right:
        fig_week = px.box(
            home_df,
            x="week_part",
            y="Su_Tuketimi_m3",
            color="week_part",
            title="Hafta İçi / Hafta Sonu Tüketim Dağılımı",
            labels={"week_part": "Dönem", "Su_Tuketimi_m3": "Su Tüketimi (m³)"},
            template=PLOTLY_TEMPLATE,
            points="outliers",
        )
        st.plotly_chart(fig_week, width='stretch', config=PLOTLY_CONFIG)

    corr_cols = [c for c in ["Su_Tuketimi_m3", "temp", "rain_mean", "month", "weekday"] if c in home_df.columns]
    corr_df = home_df[corr_cols].dropna().corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Korelasyon Haritası",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_corr, width='stretch', config=PLOTLY_CONFIG)

    st.info("Lütfen sol menüden bir model seçerek analize başlayın.")
    st.stop()

selected_model_spec = MODEL_SPECS[selected_model_name]
selected_model_info = load_model_info(selected_model_name)

if st.sidebar.button("Ana Sayfa", width='stretch'):
    st.session_state["model_selection"] = HOME_OPTION
    st.rerun()

st.sidebar.caption(f"Notebook: {selected_model_spec['notebook']}")

scenario_days = 30
scenario_temp_shift = 0.0
scenario_rain_scale = 1.0

if selected_model_spec["kind"] == "feature_bundle":
    scenario_days = st.sidebar.slider("Senaryo Ufku (gün)", 7, 60, 30)
    scenario_temp_shift = st.sidebar.slider("Sıcaklık Sapması (°C)", -5.0, 5.0, 0.0, 0.5)
    scenario_rain_scale = st.sidebar.slider("Yağış Çarpanı", 0.0, 2.0, 1.0, 0.05)
elif selected_model_spec["kind"] == "sequence":
    scenario_days = st.sidebar.slider("Senaryo Ufku (gün)", 7, 60, 30)
    st.sidebar.caption("Not: GRU/LSTM tek değişkenli olduğu için sıcaklık ve yağış slider'ları bu modelde kullanılmaz.")
elif selected_model_spec["kind"] == "sarimax":
    st.sidebar.caption("Not: SARIMAX için ileri tahminler notebook'tan hazır artefakt olarak okunur.")

model = None
scaler = None
feature_names = []
look_back = None

if selected_model_spec["kind"] == "feature_bundle":
    try:
        model_bundle = load_model_bundle(selected_model_spec["artifact"])
    except FileNotFoundError as exc:
        st.error(f"Model dosyası bulunamadı: {exc}")
        st.stop()

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["features"]
elif selected_model_spec["kind"] == "sequence":
    model, scaler = load_sequence_artifacts(selected_model_name)
    look_back = int(selected_model_info.get("look_back", 60))

st.sidebar.header("Tarih Filtreleri")
start_date = st.sidebar.date_input("Başlangıç Tarihi", data["date"].min())
end_date = st.sidebar.date_input("Bitiş Tarihi", data["date"].max())
threshold = st.sidebar.slider(
    "Acil Durum Tüketim Eşiği",
    int(data["Su_Tuketimi_m3"].min()),
    int(data["Su_Tuketimi_m3"].max()),
    int(data["Su_Tuketimi_m3"].quantile(0.90)),
)

filtered = data[
    (data["date"] >= pd.to_datetime(start_date))
    & (data["date"] <= pd.to_datetime(end_date))
].copy()

if filtered.empty:
    st.warning("Seçilen tarih aralığında veri bulunamadı.")
    st.stop()

live_prediction_available = selected_model_spec["kind"] in {"feature_bundle", "sequence"}

if selected_model_spec["kind"] == "feature_bundle":
    X = build_feature_frame(filtered, feature_names)
    X_scaled_arr = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=feature_names, index=X.index)
    filtered["prediction"] = model.predict(X_scaled)
elif selected_model_spec["kind"] == "sequence":
    sequence_predictions = build_sequence_prediction_frame(data, model, scaler, look_back)
    filtered = filtered.merge(sequence_predictions[["date", "prediction"]], on="date", how="left")
else:
    filtered["prediction"] = np.nan

risk_df = build_risk_fields(filtered, threshold)
filtered["rain_mean"] = risk_df["rain_mean"]
filtered["temp"] = risk_df["temp"]
filtered["error"] = filtered["prediction"] - filtered["Su_Tuketimi_m3"]

if live_prediction_available and filtered["prediction"].notna().any():
    _eval = filtered[["Su_Tuketimi_m3", "prediction"]].dropna()
    mae = mean_absolute_error(_eval["Su_Tuketimi_m3"], _eval["prediction"])
    mse = mean_squared_error(_eval["Su_Tuketimi_m3"], _eval["prediction"])
    rmse = np.sqrt(mse)
    r2 = r2_score(_eval["Su_Tuketimi_m3"], _eval["prediction"])

    start_perf = time.perf_counter()
    if selected_model_spec["kind"] == "feature_bundle":
        _ = model.predict(X_scaled[: min(100, len(X_scaled))])
    else:
        tail_values = data[["Su_Tuketimi_m3"]].tail(look_back).astype(float)
        tail_scaled = scaler.transform(tail_values).reshape(1, look_back, 1)
        _ = model.predict(tail_scaled, verbose=0)
    predict_latency_ms = (time.perf_counter() - start_perf) * 1000
else:
    mae = selected_model_info["metrics"].get("MAE", np.nan)
    rmse = selected_model_info["metrics"].get("RMSE", np.nan)
    r2 = selected_model_info["metrics"].get("R2", np.nan)
    predict_latency_ms = np.nan

risk_status = summarize_risk(risk_df)
risk_count = int(risk_df["high_demand"].sum())

if live_prediction_available:
    if selected_model_spec["kind"] == "feature_bundle":
        scenario_forecast = generate_scenario_forecast(
            source_df=filtered,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            days=scenario_days,
            temp_shift=scenario_temp_shift,
            rain_scale=scenario_rain_scale,
        )
    else:
        scenario_forecast = generate_sequence_forecast(
            source_df=filtered,
            sequence_model=model,
            scaler=scaler,
            look_back=look_back,
            days=scenario_days,
        )
elif selected_model_spec["kind"] == "sarimax":
    scenario_forecast = load_future_forecast_csv(selected_model_spec["forecast_file"]).rename(
        columns={"sarimax_tahmin_m3": "prediction"}
    )
else:
    scenario_forecast = pd.DataFrame(columns=["date", "prediction", "scenario_temp", "rain_mean"])

if live_prediction_available and filtered["prediction"].notna().any():
    baseline_mean = float(filtered["prediction"].tail(min(scenario_days, len(filtered))).mean())
else:
    baseline_mean = float(filtered["Su_Tuketimi_m3"].tail(min(scenario_days, len(filtered))).mean())

scenario_mean = float(scenario_forecast["prediction"].mean()) if not scenario_forecast.empty else baseline_mean
scenario_delta_pct = ((scenario_mean - baseline_mean) / max(baseline_mean, 1.0)) * 100

drought_risk_pct = float(risk_df["low_rain"].mean() * 100)
drought_delta_vs_threshold = drought_risk_pct - 30.0

avg_actual_demand = float(filtered["Su_Tuketimi_m3"].mean())
if live_prediction_available and filtered["prediction"].notna().any():
    avg_predicted_demand = float(filtered["prediction"].mean())
else:
    avg_predicted_demand = avg_actual_demand
demand_change_pct = ((avg_predicted_demand - avg_actual_demand) / max(avg_actual_demand, 1.0)) * 100

if "Ort_Sicaklik" in data.columns:
    long_term_temp = float(data["Ort_Sicaklik"].mean())
else:
    long_term_temp = float(filtered["temp"].mean())
current_temp = float(filtered["temp"].mean())
temp_anomaly = current_temp - long_term_temp

comparison_df = build_model_comparison_table()
selected_theme = MODEL_THEMES.get(selected_model_name, MODEL_THEMES["Makine Öğrenmesi Modelleri"])

if comparison_df["RMSE"].notna().any():
    best_model_row = comparison_df.sort_values("RMSE", ascending=True).iloc[0]
else:
    best_model_row = pd.Series({"Model": "-", "RMSE": np.nan, "R2": np.nan, "MAE": np.nan})

model_titles = {
    "Makine Öğrenmesi Modelleri": "Çoklu Regresyon Tabanlı Su Talebi Analizi",
    "SARIMAX": "SARIMAX ile Zaman Serisi Su Talebi Analizi",
    "GRU": "GRU ile Derin Öğrenme Tabanlı Su Talebi Analizi",
    "LSTM": "LSTM ile Derin Öğrenme Tabanlı Su Talebi Analizi",
}

hero_title = model_titles.get(selected_model_name, "Su Talebi Analizi")
hero_subtitle = selected_model_spec["description"]

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {selected_theme['background']};
    }}
    [data-testid="stSidebar"] {{
        background: {selected_theme['sidebar']};
    }}
    .best-model-title {{
        color: {selected_theme['secondary']};
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {selected_theme['accent']}80;
    }}
    [data-testid="stMetricValue"] {{
        color: {selected_theme['accent']} !important;
        text-shadow: 0 0 10px {selected_theme['accent']}66;
    }}
    </style>
    <h1 class="main-title">{hero_title}</h1>
    <p class="subtitle">{hero_subtitle}</p>
    <p class="meta"><strong>Danışman:</strong> Doç. Dr. Canan Batur Şahin - Mühendislik ve Doğa Bilimleri Fakültesi</p>
    <p class="meta"><strong>Yusuf Yıldırım:</strong> Yazılım Mühendisliği</p>
    """,
    unsafe_allow_html=True,
)
st.divider()

chips = "".join(
    [
        f'<span class="model-chip">{name}</span>'
        for name in MODEL_SPECS.keys()
    ]
)
st.markdown(chips, unsafe_allow_html=True)

st.caption(
    f"Seçili model: {selected_model_name} | Notebook: {selected_model_spec['notebook']} | {selected_model_spec['description']}"
)

st.markdown("#### Hızlı Durum Özeti")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Kuraklık Riski",
    f"%{drought_risk_pct:.1f}",
    f"{drought_delta_vs_threshold:+.1f} puan (30% eşik)",
    delta_color="normal",
)
kpi2.metric(
    "Beklenen Su Talebi",
    f"{avg_predicted_demand:,.0f} m³",
    f"{demand_change_pct:+.1f}%",
    delta_color="normal",
)
kpi3.metric(
    "Sıcaklık Anomalisi",
    f"{current_temp:.1f} °C",
    f"{temp_anomaly:+.1f} °C",
    delta_color="normal",
)
kpi4.metric(
    "Eşik Üstü Gün",
    f"{risk_count}",
    f"{(risk_count / len(filtered)) * 100:+.1f}%",
    delta_color="normal",
)
st.divider()

hist_tab, model_tab, future_tab, scenario_tab, irrigation_tab, reservoir_tab, report_tab = st.tabs(
    [
        "Tarihsel Analiz",
        "Model Karşılaştırması",
        "Gelecek Tahminleri",
        "Senaryo",
        "Tarımsal Sulama Planı",
        "Baraj Doluluk Optimizasyonu",
        "Genel Değerlendirme",
    ]
)

with hist_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kayıt Sayısı", f"{len(filtered):,}")
    c2.metric("Ort. Tüketim", f"{filtered['Su_Tuketimi_m3'].mean():,.0f} m³")
    c3.metric("Ort. Yağış", f"{filtered['rain_mean'].mean():.1f} mm")
    c4.metric("Riskli Gün", f"{risk_count}")

    if risk_status == "Düşük":
        st.success("Genel Risk Durumu: Düşük")
    elif risk_status == "Orta":
        st.warning("Genel Risk Durumu: Orta")
    else:
        st.error("Genel Risk Durumu: Yüksek")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig_consumption = px.line(
            filtered,
            x="date",
            y="Su_Tuketimi_m3",
            title="Tarihsel Su Tüketimi",
            labels={"date": "Tarih", "Su_Tuketimi_m3": "Tüketim (m³)"},
            template=PLOTLY_TEMPLATE,
        )
        fig_consumption = style_time_series_figure(fig_consumption, "Tüketim (m³)")
        st.plotly_chart(fig_consumption, width='stretch', config=PLOTLY_CONFIG)

    with chart_right:
        fig_rain = px.line(
            filtered,
            x="date",
            y="rain_mean",
            title="Tarihsel Yağış Ortalaması",
            labels={"date": "Tarih", "rain_mean": "Yağış (mm)"},
            template=PLOTLY_TEMPLATE,
        )
        fig_rain = style_time_series_figure(fig_rain, "Yağış (mm)")
        st.plotly_chart(fig_rain, width='stretch', config=PLOTLY_CONFIG)

    with st.expander("İlk 100 satırı göster"):
        st.dataframe(filtered.head(100), width='stretch')

with model_tab:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", "-" if pd.isna(mae) else f"{mae:,.0f}")
    m2.metric("RMSE", "-" if pd.isna(rmse) else f"{rmse:,.0f}")
    m3.metric("R2", "-" if pd.isna(r2) else f"{r2:.3f}")
    m4.metric("Yanıt Süresi", "-" if pd.isna(predict_latency_ms) else f"{predict_latency_ms:.1f} ms")

    info_left, info_right = st.columns([1.4, 1])
    with info_left:
        st.markdown("##### Model Bilgisi")
        st.write(f"Model: {selected_model_name}")
        st.write(f"Notebook: {selected_model_spec['notebook']}")
        st.write(f"Tür: {selected_model_spec['kind']}")
        if selected_model_spec["kind"] == "feature_bundle":
            st.write(f"Özellik sayısı: {selected_model_info['feature_count']}")
        if "look_back" in selected_model_info:
            st.write(f"Look-back: {selected_model_info['look_back']} gün")
        if "order" in selected_model_info:
            st.write(f"Order: {selected_model_info['order']}")
            st.write(f"Seasonal order: {selected_model_info['seasonal_order']}")

    with info_right:
        st.markdown("##### Artefaktlar")
        st.dataframe(
            pd.DataFrame({"Dosya": selected_model_info["artifacts"]}),
            width='stretch',
            hide_index=True,
        )

    st.markdown("##### Tüm Modellerin Metrik Karşılaştırması")
    st.dataframe(comparison_df, width='stretch', hide_index=True)

    comparison_chart = comparison_df.melt(
        id_vars=["Model", "Tür"],
        value_vars=["MAE", "RMSE", "R2"],
        var_name="Metrik",
        value_name="Değer",
    ).dropna()
    fig_compare = px.bar(
        comparison_chart,
        x="Model",
        y="Değer",
        color="Metrik",
        barmode="group",
        title="Model Performans Karşılaştırması",
        template=PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_compare, width='stretch', config=PLOTLY_CONFIG)

    if live_prediction_available:
        top_row_left, top_row_right = st.columns(2)
        with top_row_left:
            fig_pred_line = px.line(
                filtered,
                x="date",
                y=["Su_Tuketimi_m3", "prediction"],
                title="Gerçek vs Tahmin",
                labels={"value": "Tüketim (m³)", "date": "Tarih", "variable": "Seri"},
                template=PLOTLY_TEMPLATE,
                color_discrete_map={"Su_Tuketimi_m3": "#1f77b4", "prediction": "#ff7f0e"},
            )
            fig_pred_line = style_time_series_figure(fig_pred_line, "Tüketim (m³)")
            st.plotly_chart(fig_pred_line, width='stretch', config=PLOTLY_CONFIG)

        with top_row_right:
            error_frame = filtered.dropna(subset=["prediction", "error"])
            fig_error = px.scatter(
                error_frame,
                x="date",
                y="error",
                title="Tahmin Hatası (Zamana Göre)",
                labels={"date": "Tarih", "error": "Hata (m³)"},
                template=PLOTLY_TEMPLATE,
            )
            fig_error = style_time_series_figure(fig_error, "Hata (m³)")
            st.plotly_chart(fig_error, width='stretch', config=PLOTLY_CONFIG)

        extra_left, extra_right = st.columns(2)
        with extra_left:
            compare_frame = filtered.dropna(subset=["prediction"]).copy()
            fig_scatter = px.scatter(
                compare_frame,
                x="Su_Tuketimi_m3",
                y="prediction",
                title="Gerçek vs Tahmin Dağılımı",
                labels={"Su_Tuketimi_m3": "Gerçek (m³)", "prediction": "Tahmin (m³)"},
                template=PLOTLY_TEMPLATE,
                opacity=0.55,
            )
            min_axis = float(min(compare_frame["Su_Tuketimi_m3"].min(), compare_frame["prediction"].min()))
            max_axis = float(max(compare_frame["Su_Tuketimi_m3"].max(), compare_frame["prediction"].max()))
            fig_scatter.add_shape(
                type="line",
                x0=min_axis,
                y0=min_axis,
                x1=max_axis,
                y1=max_axis,
                line=dict(color="#F39C12", dash="dash"),
            )
            st.plotly_chart(fig_scatter, width='stretch', config=PLOTLY_CONFIG)

        with extra_right:
            monthly_eval = compare_frame.copy()
            monthly_eval["month_period"] = monthly_eval["date"].dt.to_period("M").dt.to_timestamp()
            monthly_eval = (
                monthly_eval.groupby("month_period", as_index=False)[["Su_Tuketimi_m3", "prediction"]]
                .mean()
                .rename(columns={"month_period": "date"})
            )
            fig_monthly = px.line(
                monthly_eval,
                x="date",
                y=["Su_Tuketimi_m3", "prediction"],
                title="Aylık Ortalama Gerçek vs Tahmin",
                labels={"value": "Ortalama Tüketim (m³)", "date": "Tarih", "variable": "Seri"},
                template=PLOTLY_TEMPLATE,
            )
            fig_monthly = style_time_series_figure(fig_monthly, "Ortalama Tüketim (m³)")
            st.plotly_chart(fig_monthly, width='stretch', config=PLOTLY_CONFIG)

        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            fig_imp = px.bar(
                importance.reset_index(),
                x="index",
                y=0,
                title="Özellik Önemi",
                labels={"index": "Özellik", "0": "Önem"},
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_imp, width='stretch')
    else:
        st.info(
            "Bu seçim notebook'ta üretilen modeli ve saklanan metrikleri gösterir. Canlı Streamlit tahmin akışı şu an yalnızca 'Makine Öğrenmesi Modelleri' için etkindir."
        )

with future_tab:
    st.subheader("Gelecek Dönem Operasyonel Uyarılar")

    if live_prediction_available:
        a1, a2, a3 = st.columns(3)
        a1.metric("Sulama Uyarısı", "Yüksek" if risk_df["high_demand"].iloc[-1] else "Normal")
        a2.metric("Haşere Riski", risk_df["pest_risk"].mode().iloc[0])
        a3.metric("Hastalık Riski", risk_df["disease_risk"].mode().iloc[0])

        proj = filtered[["date", "prediction", "Su_Tuketimi_m3", "rain_mean", "temp"]].dropna(subset=["prediction"]).copy()
        proj = proj.sort_values("date", ascending=False).head(30).sort_values("date")
        proj["durum"] = np.where(proj["prediction"] >= threshold, "Riskli", "Normal")

        fig_future = px.line(
            proj,
            x="date",
            y="prediction",
            color="durum",
            title="Son 30 Gün İçin Tahmin Eğilimi",
            labels={"date": "Tarih", "prediction": "Tahmin (m³)", "durum": "Durum"},
            template=PLOTLY_TEMPLATE,
            color_discrete_map={"Riskli": "#d62728", "Normal": "#2ca02c"},
        )
        fig_future = style_time_series_figure(fig_future, "Tahmin (m³)")
        st.plotly_chart(fig_future, width='stretch', config=PLOTLY_CONFIG)

        risky_days = proj[proj["prediction"] >= threshold].sort_values("prediction", ascending=False)
        if risky_days.empty:
            st.success("Seçilen aralıkta eşik üstü gün bulunmadı.")
        else:
            st.warning(f"Eşik üstü gün sayısı: {len(risky_days)}")
            st.dataframe(risky_days, width='stretch')
    elif selected_model_spec["kind"] == "sarimax" and not scenario_forecast.empty:
        st.info("Bu görünüm SARIMAX notebook'unda üretilen 30 günlük ileri tahmini gösterir.")
        fig_future = px.line(
            scenario_forecast,
            x="date",
            y="prediction",
            title="SARIMAX 30 Günlük İleri Tahmin",
            labels={"date": "Tarih", "prediction": "Tahmin (m³)"},
            template=PLOTLY_TEMPLATE,
        )
        fig_future = style_time_series_figure(fig_future, "Tahmin (m³)")
        st.plotly_chart(fig_future, width='stretch', config=PLOTLY_CONFIG)
        st.dataframe(scenario_forecast, width='stretch')
    else:
        st.info("Bu model için dashboard tarafında canlı gelecek tahmini yerine notebook artefaktları ve metrikler gösteriliyor.")

with scenario_tab:
    st.subheader("Etkileşimli Senaryo Simülasyonu")

    if live_prediction_available:
        if selected_model_spec["kind"] == "feature_bundle":
            st.info(
                f"Seçili model: {selected_model_name}. Senaryo: önümüzdeki {scenario_days} gün için sıcaklık {scenario_temp_shift:+.1f}°C, yağış çarpanı {scenario_rain_scale:.2f}."
            )
            st.markdown(
                "Örnek soru: **Önümüzdeki ay sıcaklık normallerin 2°C üzerinde olursa ve hiç yağış almazsak su talebi ne olur?**"
            )
        else:
            st.info(
                f"Seçili model: {selected_model_name}. Bu model tek değişkenli olduğu için senaryo tahmini son {look_back} günlük pencere üzerinden recursive olarak üretilir."
            )

        s1, s2, s3 = st.columns(3)
        s1.metric("Senaryo Ortalama Talep", f"{scenario_mean:,.0f} m³", f"{scenario_delta_pct:+.1f}%")
        s2.metric("Referans Ortalama Talep", f"{baseline_mean:,.0f} m³")
        s3.metric("Senaryo İçinde Eşik Üstü Gün", f"{int((scenario_forecast['prediction'] >= threshold).sum())}")

        fig_scenario = px.line(
            scenario_forecast,
            x="date",
            y="prediction",
            title="Senaryo Bazlı Gelecek Su Talebi Tahmini",
            labels={"date": "Tarih", "prediction": "Tahmin (m³)"},
            template=PLOTLY_TEMPLATE,
        )
        fig_scenario.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#F39C12",
            annotation_text=f"Acil Eşik ({threshold:,.0f} m³)",
            annotation_position="top left",
        )
        fig_scenario = style_time_series_figure(fig_scenario, "Tahmin (m³)")
        st.plotly_chart(fig_scenario, width='stretch', config=PLOTLY_CONFIG)

        if not scenario_forecast.empty:
            scenario_table = scenario_forecast.copy()
            scenario_table["durum"] = np.where(scenario_table["prediction"] >= threshold, "Riskli", "Normal")
            st.dataframe(scenario_table, width='stretch')
    elif selected_model_spec["kind"] == "sarimax" and not scenario_forecast.empty:
        st.info(
            "SARIMAX için notebook'ta üretilen 30 günlük hazır tahmin kullanılıyor."
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Tahmin Ortalama Talep", f"{scenario_mean:,.0f} m³", f"{scenario_delta_pct:+.1f}%")
        s2.metric("Referans Ortalama Talep", f"{baseline_mean:,.0f} m³")
        s3.metric("Eşik Üstü Gün", f"{int((scenario_forecast['prediction'] >= threshold).sum())}")

        fig_scenario = px.line(
            scenario_forecast,
            x="date",
            y="prediction",
            title="SARIMAX Gelecek Tahmini",
            labels={"date": "Tarih", "prediction": "Tahmin (m³)"},
            template=PLOTLY_TEMPLATE,
        )
        fig_scenario = style_time_series_figure(fig_scenario, "Tahmin (m³)")
        st.plotly_chart(fig_scenario, width='stretch', config=PLOTLY_CONFIG)
        st.dataframe(scenario_forecast, width='stretch')
    else:
        st.info(
            "GRU ve LSTM seçimlerinde bu ekranda notebook kaynakları ve test metrikleri gösterilir. İstersen bir sonraki adımda bu modeller için canlı tahmin akışını da dashboard'a bağlayabilirim."
        )

with irrigation_tab:
    st.subheader("Tarımsal Sulama Planı (Model Tabanlı)")
    plan_df = build_irrigation_plan_frame(filtered, scenario_forecast, threshold)

    irrigated_area = float(filtered["Sulu_Tarim_Alani_Dekar"].mean()) if "Sulu_Tarim_Alani_Dekar" in filtered.columns else np.nan
    total_area = float(filtered["Toplam_Tarim_Alani_Dekar"].mean()) if "Toplam_Tarim_Alani_Dekar" in filtered.columns else np.nan
    sulu_oran = (100 * irrigated_area / total_area) if np.isfinite(total_area) and total_area > 0 and np.isfinite(irrigated_area) else np.nan

    p1, p2, p3 = st.columns(3)
    p1.metric("Seçili Model", selected_model_name)
    p2.metric("Sulu Tarım Alanı", "-" if not np.isfinite(irrigated_area) else f"{irrigated_area:,.0f} dekar")
    p3.metric("Sulu Alan Oranı", "-" if pd.isna(sulu_oran) else f"%{sulu_oran:.1f}")

    st.markdown(
        "Bu menüde su tüketimi tahminleri ile tarımsal alan verileri birlikte kullanılarak günlük sulama ihtiyacı planlanır."
    )

    left_plan, right_plan = st.columns(2)
    with left_plan:
        fig_plan = px.line(
            plan_df,
            x="date",
            y="prediction",
            color="Oncelik",
            title="Tahmin Edilen Su Talebi ve Sulama Önceliği",
            labels={"date": "Tarih", "prediction": "Tahmin (m³)", "Oncelik": "Öncelik"},
            template=PLOTLY_TEMPLATE,
            color_discrete_map={"Yüksek": "#E74C3C", "Orta": "#F1C40F"},
        )
        fig_plan = style_time_series_figure(fig_plan, "Tahmin (m³)")
        st.plotly_chart(fig_plan, width='stretch', config=PLOTLY_CONFIG)

    with right_plan:
        fig_per_dekar = px.bar(
            plan_df.tail(30),
            x="date",
            y="Sulama_Ihtiyaci_m3_dekar",
            color="Yagis_Etkisi",
            title="Birim Alan Başına Sulama İhtiyacı (m³/dekar)",
            labels={"date": "Tarih", "Sulama_Ihtiyaci_m3_dekar": "m³/dekar", "Yagis_Etkisi": "Yağış Etkisi"},
            template=PLOTLY_TEMPLATE,
            color_discrete_map={"Azalt": "#2ECC71", "Dengele": "#3498DB", "Artır": "#E67E22"},
        )
        st.plotly_chart(fig_per_dekar, width='stretch', config=PLOTLY_CONFIG)

    st.markdown("##### Operasyon Önerileri")
    st.write("1. `Yüksek` öncelik günlerinde sulama vardiyaları artırılmalı, düşük yağış günlerinde su kaynağı planı güçlendirilmelidir.")
    st.write("2. Yağış etkisi `Azalt` olan günlerde sulama süresi düşürülerek su tasarrufu sağlanabilir.")
    st.write("3. Bu plan, seçili modelin tahminine göre otomatik güncellenir; model değiştiğinde sulama planı da değişir.")

    cols = ["date", "prediction", "rain_mean", "Oncelik", "Yagis_Etkisi", "Sulama_Ihtiyaci_m3_dekar"]
    show_cols = [c for c in cols if c in plan_df.columns]
    st.dataframe(plan_df[show_cols].tail(45), width='stretch')


with reservoir_tab:
    st.subheader("Kayıp-Kaçak ve İklim Senaryosu ile Baraj Doluluk Artırma")

    try:
        leakage_df = load_leakage_data()
        climate_df = load_climate_scenario_data()
    except FileNotFoundError as exc:
        st.warning(f"Bu menü için gerekli veri dosyası bulunamadı: {exc}")
        st.stop()

    if leakage_df.empty or climate_df.empty:
        st.warning("Kayıp-kaçak veya iklim senaryosu verisi boş görünüyor.")
        st.stop()

    min_year = int(climate_df["year"].min())
    max_year = int(climate_df["year"].max())

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        years = st.slider("Senaryo Yıl Aralığı", min_year, max_year, (min_year, min(min_year + 10, max_year)))
    with rc2:
        leakage_reduction_pp = st.slider("Kayıp-Kaçak Azaltımı (puan)", 0.0, 25.0, 8.0, 0.5)
    with rc3:
        demand_saving_pct = st.slider("Talep Yönetimi Tasarrufu (%)", 0.0, 25.0, 6.0, 0.5)

    demand_pressure_pct = max(0.0, float(scenario_delta_pct))
    opt_df = build_reservoir_optimization_frame(
        climate_df=climate_df,
        leakage_df=leakage_df,
        start_year=years[0],
        end_year=years[1],
        leakage_reduction_pp=leakage_reduction_pp,
        demand_saving_pct=demand_saving_pct,
        demand_pressure_pct=demand_pressure_pct,
    )

    if opt_df.empty:
        st.warning("Seçilen yıl aralığında optimize edilecek senaryo verisi bulunamadı.")
        st.stop()

    base_mean = float(opt_df["Baz_Doluluk_%"].mean())
    improved_mean = float(opt_df["Iyilestirilmis_Doluluk_%"].mean())
    gain_mean = improved_mean - base_mean
    critical_before = int((opt_df["Baz_Doluluk_%"] < 35).sum())
    critical_after = int((opt_df["Iyilestirilmis_Doluluk_%"] < 35).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ortalama Baz Doluluk", f"%{base_mean:.1f}")
    m2.metric("Ortalama İyileştirilmiş Doluluk", f"%{improved_mean:.1f}", f"{gain_mean:+.1f} puan")
    m3.metric("Kritik Ay Sayısı (<35%)", f"{critical_after}", f"{critical_after - critical_before:+d}")
    m4.metric("Model Talep Baskısı", f"%{demand_pressure_pct:.1f}")

    left_res, right_res = st.columns(2)
    with left_res:
        fig_res = px.line(
            opt_df,
            x="date",
            y=["Baz_Doluluk_%", "Iyilestirilmis_Doluluk_%"],
            title="Gelecek Baraj Doluluk Senaryosu: Baz vs İyileştirilmiş",
            labels={"value": "Doluluk (%)", "date": "Tarih", "variable": "Senaryo"},
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_res, width='stretch', config=PLOTLY_CONFIG)

    with right_res:
        month_gain = (
            opt_df.groupby("month", as_index=False)["Doluluk_Kazanimi_%"].mean()
            .sort_values("month")
        )
        fig_gain = px.bar(
            month_gain,
            x="month",
            y="Doluluk_Kazanimi_%",
            title="Aylık Ortalama Doluluk Kazanımı",
            labels={"month": "Ay", "Doluluk_Kazanimi_%": "Kazanım (puan)"},
            template=PLOTLY_TEMPLATE,
            color="Doluluk_Kazanimi_%",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(fig_gain, width='stretch', config=PLOTLY_CONFIG)

    st.markdown("##### Stratejik Eylem Planı")
    if gain_mean >= 4:
        st.success("Kayıp-kaçak azaltımı + talep yönetimi kombinasyonu baraj doluluğunda anlamlı artış sağlıyor.")
    elif gain_mean >= 2:
        st.info("Dolulukta orta düzey iyileşme var. Ek olarak damla sulama yaygınlaştırması ve gece sulaması önerilir.")
    else:
        st.warning("İyileşme sınırlı. Kayıp-kaçak azaltım hedefi yükseltilmeli ve talep yönetimi güçlendirilmelidir.")

    st.write("1. Kayıp-kaçak oranı yüksek bölgelerde (özellikle eski şebeke) altyapı yenileme önceliklendirilmeli.")
    st.write("2. Su talebi artış baskısında ürün deseni ve sulama programı su verimliliği odaklı revize edilmeli.")
    st.write("3. İklim senaryosu yağış düşüşü gösterdiği aylarda baraj işletme kuralı daha korumacı uygulanmalı.")

    show_cols = [
        "date",
        "Toplam_Yagis",
        "Kayip_Kacak_Orani_%",
        "Baz_Doluluk_%",
        "Iyilestirilmis_Doluluk_%",
        "Doluluk_Kazanimi_%",
    ]
    st.dataframe(opt_df[[c for c in show_cols if c in opt_df.columns]].head(120), width='stretch')


with report_tab:
    st.subheader("Genel Değerlendirme ve Modele Özel Sonuç Raporu")

    model_rank = comparison_df.copy()
    if model_rank["RMSE"].notna().any():
        model_rank = model_rank.sort_values("RMSE", ascending=True).reset_index(drop=True)
        selected_rank = int(model_rank.index[model_rank["Model"] == selected_model_name][0] + 1)
    else:
        selected_rank = np.nan

    profile_map = {
        "Makine Öğrenmesi Modelleri": {
            "horizon": "Kısa-Orta vade (günlük/haftalık)",
            "need_info": "Yağış, sıcaklık, takvim etkisi, tarımsal alan ve kayıp-kaçak verisi",
            "field_use": "Günlük sulama vardiyası ve tarla bazlı önceliklendirme",
            "limitation": "Yeni iklim rejimlerinde periyodik yeniden eğitim gerekir",
            "improvements": [
                "Tarımsal parsel bazlı veri eklentisi ile bölgesel hata düşürülebilir.",
                "Kayıp-kaçak oranı aylık seviyede modele dışsal değişken olarak eklenmeli.",
                "Ürün deseni (mısır, kayısı vb.) bilgisi ile su talebi daha gerçekçi ayrıştırılmalı.",
            ],
        },
        "SARIMAX": {
            "horizon": "Orta vade (aylık/mevsimsel)",
            "need_info": "Mevsimsellik, yağış rejimi, sıcaklık anomali serisi, exog tutarlılığı",
            "field_use": "Baraj işletme planı ve dönemsel su tahsis kararları",
            "limitation": "Ani yapısal kırılmalarda parametre güncellemesi gerekir",
            "improvements": [
                "Exog tarafına sulu alan oranı ve kayıp-kaçak trendi eklenmeli.",
                "Kuru sezon için ayrı parametre kalibrasyonu yapılmalı.",
                "Aylık tahminler operasyonel eşiklerle (kritik doluluk) birlikte izlenmeli.",
            ],
        },
        "GRU": {
            "horizon": "Kısa vade (ardışık günlük tahmin)",
            "need_info": "Yüksek frekanslı tüketim serisi, temizlenmiş uç değerler",
            "field_use": "Ani talep sıçramalarında hızlı erken uyarı",
            "limitation": "Açıklanabilirlik düşüktür, tek değişkenli kullanımda dış etkenleri sınırlı görür",
            "improvements": [
                "Çok değişkenli GRU yapısına yağış/sıcaklık girişi eklenmeli.",
                "Aşırı günleri temsil eden ağırlıklı kayıp fonksiyonu denenmeli.",
                "Gerçek zamanlı akışta haftalık yeniden kalibrasyon yapılmalı.",
            ],
        },
        "LSTM": {
            "horizon": "Kısa-Orta vade (uzun bağımlılık)",
            "need_info": "Uzun dönem düzenli geçmiş veri, mevsimsel döngü tutarlılığı",
            "field_use": "Uzun dönemli talep desenlerini yakalayarak planlı sulama",
            "limitation": "Eğitim/veri maliyeti yüksektir, açıklanabilirlik sınırlıdır",
            "improvements": [
                "LSTM girişine sosyoekonomik sinyaller eklenmeli (nüfus, sulu alan).",
                "Look-back optimizasyonu ile gereksiz gecikme azaltılmalı.",
                "Model çıktıları eşik bazlı karar motoru ile otomatik aksiyona bağlanmalı.",
            ],
        },
    }
    selected_profile = profile_map.get(selected_model_name, profile_map["Makine Öğrenmesi Modelleri"])

    if pd.isna(r2):
        perf_state = "Performans dış artefakt metrikleri ile izleniyor; sahada güvenlik payı bırakılmalı."
    elif r2 >= 0.85:
        perf_state = "Performans güçlü; operasyonel planlama için güvenilir aralıkta."
    elif r2 >= 0.70:
        perf_state = "Performans orta; planlamada model + uzman yorumu birlikte kullanılmalı."
    else:
        perf_state = "Performans düşük; doğrudan karar yerine uyarı sinyali olarak kullanılmalı."

    speed_score = float(np.clip(100 - (0 if pd.isna(predict_latency_ms) else predict_latency_ms), 10, 100))
    explain_score = 85 if selected_model_spec["kind"] in {"feature_bundle", "sarimax"} else 55
    scenario_score = 90 if selected_model_spec["kind"] == "feature_bundle" else (70 if selected_model_spec["kind"] == "sarimax" else 60)
    data_need_score = 65 if selected_model_spec["kind"] in {"feature_bundle", "sarimax"} else 50

    r1, r2c, r3, r4 = st.columns(4)
    r1.metric("Seçili Model", selected_model_name)
    r2c.metric("Model Sıralaması (RMSE)", "-" if pd.isna(selected_rank) else f"{selected_rank}/{len(model_rank)}")
    r3.metric("Kuraklık Riski", f"%{drought_risk_pct:.1f}")
    r4.metric("Senaryo Talep Farkı", f"{scenario_delta_pct:+.1f}%")

    card_left, card_right = st.columns(2)
    with card_left:
        st.markdown(
            f"""
            <div class="best-model-card">
                <div class="best-model-title">Modele Özel Değerlendirme</div>
                <div><strong>Karar Ufku:</strong> {selected_profile['horizon']}</div>
                <div style="margin-top:0.45rem;"><strong>İhtiyaç Duyulan Bilgiler:</strong> {selected_profile['need_info']}</div>
                <div style="margin-top:0.45rem;"><strong>Saha Kullanımı:</strong> {selected_profile['field_use']}</div>
                <div style="margin-top:0.45rem;"><strong>Sınırlama:</strong> {selected_profile['limitation']}</div>
                <div style="margin-top:0.55rem;">{perf_state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_right:
        score_df = pd.DataFrame(
            {
                "Boyut": ["Yanıt Hızı", "Açıklanabilirlik", "Senaryo Uyum", "Veri Hazırlık"],
                "Skor": [speed_score, explain_score, scenario_score, data_need_score],
            }
        )
        fig_score = px.bar(
            score_df,
            x="Boyut",
            y="Skor",
            color="Skor",
            title="Modelin Operasyonel Uygunluk Profili",
            template=PLOTLY_TEMPLATE,
            color_continuous_scale="Blues",
            range_y=[0, 100],
        )
        st.plotly_chart(fig_score, width='stretch', config=PLOTLY_CONFIG)

    left_eval, right_eval = st.columns(2)
    with left_eval:
        summary_frame = pd.DataFrame(
            {
                "Gösterge": ["Kuraklık Riski", "Talep Değişimi", "Senaryo Etkisi", "Eşik Üstü Gün Oranı"],
                "Değer": [drought_risk_pct, demand_change_pct, scenario_delta_pct, 100 * risk_count / max(len(filtered), 1)],
            }
        )
        fig_summary = px.bar(
            summary_frame,
            x="Gösterge",
            y="Değer",
            color="Değer",
            title="İhtiyaç Duyulan Operasyonel Göstergeler",
            template=PLOTLY_TEMPLATE,
            color_continuous_scale="Sunset",
        )
        st.plotly_chart(fig_summary, width='stretch', config=PLOTLY_CONFIG)

    with right_eval:
        valid_rank = model_rank[["Model", "RMSE", "R2"]].dropna(subset=["RMSE"]).copy()
        if not valid_rank.empty:
            fig_rank = px.scatter(
                valid_rank,
                x="RMSE",
                y="R2",
                color="Model",
                size=np.maximum(0.001, valid_rank["R2"].fillna(0.1)),
                title="Model Konumlandırma Haritası (RMSE-R2)",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_rank, width='stretch', config=PLOTLY_CONFIG)
        else:
            st.info("Model konumlandırma grafiği için yeterli metrik bulunamadı.")

    st.markdown("##### Modele Özel İyileştirme Adımları")
    for i, item in enumerate(selected_profile["improvements"], start=1):
        st.write(f"{i}. {item}")

    common_roadmap = pd.DataFrame(
        [
            {"Öncelik": "1", "Alan": "Veri", "Eylem": "Kayıp-kaçak ve tarımsal alan verisini aylık eşleştirme ile güncel tut", "Beklenen Etki": "Tahminlerin saha gerçekliğine yaklaşması"},
            {"Öncelik": "2", "Alan": "Operasyon", "Eylem": "Eşik üstü günlerde sulama vardiya planını otomatik tetikle", "Beklenen Etki": "Kritik günlerde hızlı yanıt"},
            {"Öncelik": "3", "Alan": "Altyapı", "Eylem": "Kayıp-kaçak odaklı bölgesel şebeke yenileme planı", "Beklenen Etki": "Baraj doluluk kazanımı"},
        ]
    )
    st.markdown("##### Ortak Yol Haritası")
    st.dataframe(common_roadmap, width='stretch', hide_index=True)


