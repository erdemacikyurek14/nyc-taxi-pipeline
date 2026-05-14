import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Sayfa Ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi ML Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 13, "axes.labelsize": 11})

# ── Başlık ──────────────────────────────────────────────────────────────────
st.title("🚖 NYC Yellow Taxi — Büyük Veri & ML Dashboard")
st.markdown(
    "**Kafka → Spark → Delta Lake → MLflow** pipeline'ından gelen verilerle "
    "eğitilen regresyon modellerinin interaktif analizi."
)
st.divider()

# ── Veri Yükleme ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Gold katmanından veri okunuyor…")
def load_data():
    gold_path = os.getenv("DELTA_GOLD_PATH", "/app/data/delta/gold/fare_features")
    try:
        from pyspark.sql import SparkSession

        DELTA_PKG = "io.delta:delta-spark_2.12:3.2.0"
        spark = (
            SparkSession.builder.appName("NYC_Taxi_Dashboard")
            .config("spark.jars.packages", DELTA_PKG)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        df = spark.read.format("delta").load(gold_path).toPandas()
        spark.stop()
        return df
    except Exception as exc:
        st.sidebar.warning(f"Delta okunamadı → sentetik veri kullanılıyor.\n\n`{exc}`")
        rng = np.random.default_rng(42)
        n = 8000
        df = pd.DataFrame(
            {
                "trip_distance": rng.uniform(0.5, 20, n),
                "pickup_hour": rng.integers(0, 24, n),
                "pickup_day_of_week": rng.integers(0, 7, n),
                "passenger_count": rng.integers(1, 5, n),
                "is_airport_trip": rng.integers(0, 2, n),
                "RatecodeID": rng.integers(1, 6, n),
                "payment_type": rng.integers(1, 5, n),
                "pickup_borough": rng.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n),
                "dropoff_borough": rng.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n),
            }
        )
        df["label"] = (
            3.0
            + df["trip_distance"] * 2.5
            + df["pickup_hour"] * 0.3
            + rng.normal(0, 2, n)
        )
        return df


df_raw = load_data()

# Hedef kolon
TARGET = "label" if "label" in df_raw.columns else "fare_amount"


# ── Model Eğitimi (cache) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Modeller eğitiliyor…")
def train_models(df: pd.DataFrame):
    df_enc = pd.get_dummies(df, drop_first=True, dtype=float)
    target = "label" if "label" in df_enc.columns else "fare_amount"
    X = df_enc.drop(target, axis=1)
    y = df_enc[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosted Trees": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "Ridge (GLR)": Ridge(alpha=1.0),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        results[name] = {
            "model": mdl,
            "y_test": y_te,
            "y_pred": y_pred,
            "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "r2": float(r2_score(y_te, y_pred)),
            "features": X.columns.tolist(),
        }
    return results


results = train_models(df_raw)

# ── Sekme Düzeni ────────────────────────────────────────────────────────────
tab_eda, tab_model, tab_reg, tab_extra = st.tabs(
    ["📊 EDA & Dağılımlar", "🤖 Model Karşılaştırma", "🎯 Regresyon Analizleri", "🔍 Ek Analizler"]
)

# ════════════════════════════════════════════════════════════════════════════
# SEKME 1 — EDA & VERİ DAĞILIMI
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("Keşifsel Veri Analizi (EDA)")

    c1, c2 = st.columns(2)

    # Histogram — Mesafe
    with c1:
        st.subheader("🗺️ Yolculuk Mesafesi Dağılımı")
        fig, ax = plt.subplots(figsize=(6, 4))
        clip_dist = df_raw["trip_distance"].clip(upper=df_raw["trip_distance"].quantile(0.99))
        sns.histplot(clip_dist, bins=40, kde=True, ax=ax, color="#4C9BE8")
        ax.set_xlabel("Mesafe (Mil)")
        ax.set_ylabel("Frekans")
        ax.set_title("Yolculuk Mesafesi Histogramı")
        st.pyplot(fig)
        plt.close()

    # Pie Chart — Yolcu Sayısı
    with c2:
        st.subheader("👥 Yolcu Sayısı Dağılımı")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_raw["passenger_count"].value_counts().sort_index()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=sns.color_palette("pastel"), startangle=90)
        ax.set_title("Yolcu Sayısı (Pie Chart)")
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)

    # Line Chart — Saatlik ortalama ücret
    with c3:
        st.subheader("⏰ Saatlik Ortalama Ücret Trendi")
        hourly = df_raw.groupby("pickup_hour")[TARGET].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hourly.index, hourly.values, marker="o", color="#E8634C", linewidth=2)
        ax.fill_between(hourly.index, hourly.values, alpha=0.2, color="#E8634C")
        ax.set_xlabel("Günün Saati")
        ax.set_ylabel("Ort. Ücret ($)")
        ax.set_title("Saatlik Ortalama Ücret (Line Chart)")
        st.pyplot(fig)
        plt.close()

    # Günlük yolculuk sayısı
    with c4:
        st.subheader("📅 Haftanın Günlerine Göre Yolculuklar")
        day_labels = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
        day_counts = df_raw["pickup_day_of_week"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        # Spark dayofweek returns 1-7 (Sun=1), mod 7 maps both 0-6 and 1-7 safely
        bars = ax.bar(
            [day_labels[int(i) % 7] for i in day_counts.index],
            day_counts.values,
            color=sns.color_palette("viridis", len(day_counts)),
        )
        ax.set_xlabel("Gün")
        ax.set_ylabel("Yolculuk Sayısı")
        ax.set_title("Günlük Yolculuk Dağılımı")
        st.pyplot(fig)
        plt.close()

    # Eksik değer analizi
    st.divider()
    st.subheader("🔍 Eksik Değer Analizi")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Kolon", "Eksik Değer"]
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        st.dataframe(missing, hide_index=True, use_container_width=True)
        if missing["Eksik Değer"].sum() == 0:
            st.success("✅ Tüm eksik değerler Silver/Gold katmanlarında temizlendi!")
        else:
            st.warning("⚠️ Veri setinde eksik değerler mevcut.")
    with mc2:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(data=missing, x="Kolon", y="Eksik Değer", palette="Reds_r", ax=ax)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# SEKME 2 — MODEL KARŞILAŞTIRMA
# ════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.header("Makine Öğrenmesi Model Performansları")

    # MLflow'dan metrik çekme
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    metrics_rows = []
    try:
        exp = mlflow.get_experiment_by_name("NYC_Taxi_Fare_Prediction")
        if exp:
            runs = mlflow.search_runs([exp.experiment_id])
            if not runs.empty and "params.model_type" in runs.columns:
                latest = runs.sort_values("start_time", ascending=False).drop_duplicates("params.model_type")
                for _, row in latest.iterrows():
                    metrics_rows.append({
                        "Model": row["params.model_type"],
                        "RMSE": row.get("metrics.rmse", 0),
                        "MAE": row.get("metrics.mae", 0),
                        "R²": row.get("metrics.r2_score", 0),
                    })
    except Exception:
        pass

    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        st.success(f"✅ MLflow'dan {len(mdf)} model metriği çekildi.")
    else:
        # Eğitim sonuçlarından al
        mdf = pd.DataFrame([
            {"Model": n, "RMSE": v["rmse"], "MAE": v["mae"], "R²": v["r2"]}
            for n, v in results.items()
        ])
        st.info("ℹ️ Yerel eğitim sonuçları gösteriliyor.")

    # KPI kartları
    best = mdf.loc[mdf["R²"].idxmax()]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏆 En İyi Model", best["Model"])
    k2.metric("📉 En İyi RMSE", f"{best['RMSE']:.3f}")
    k3.metric("📉 En İyi MAE", f"{best['MAE']:.3f}")
    k4.metric("📈 En İyi R²", f"{best['R²']:.4f}")

    st.divider()

    # Grouped bar chart — 3 metrik yan yana
    st.subheader("📊 5 Model Performans Karşılaştırması (Grouped Bar Chart)")
    mdf_melt = mdf.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R²"],
                        var_name="Metrik", value_name="Değer")
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=mdf_melt, x="Model", y="Değer", hue="Metrik",
                palette=["#E74C3C", "#F39C12", "#2ECC71"], ax=ax)
    ax.set_title("5 Modelin RMSE / MAE / R² Karşılaştırması")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    ax.legend(loc="upper right")
    st.pyplot(fig)
    plt.close()

    # Feature Importance — Random Forest
    st.divider()
    st.subheader("🌟 Feature Importance — Random Forest (Horizontal Bar)")
    rf_res = results["Random Forest"]
    fi = pd.Series(rf_res["model"].feature_importances_, index=rf_res["features"]).nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("teal", len(fi)))
    ax.set_xlabel("Önem Skoru")
    ax.set_title("En Önemli 10 Özellik (Random Forest)")
    st.pyplot(fig)
    plt.close()

    # Tablo
    st.subheader("📋 Model Metrik Tablosu")
    st.dataframe(
        mdf.style.highlight_min(subset=["RMSE", "MAE"], color="#FADBD8")
               .highlight_max(subset=["R²"], color="#D5F5E3")
               .format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# SEKME 3 — REGRESYON ANALİZLERİ
# ════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.header("Regresyon Detay Analizleri")
    selected = st.selectbox("Model seçin:", list(results.keys()))
    res = results[selected]
    y_test, y_pred = res["y_test"], res["y_pred"]
    residuals = y_test - y_pred

    c1, c2 = st.columns(2)

    # Scatter — Gerçek vs Tahmin
    with c1:
        st.subheader("🎯 Gerçek vs Tahmin (Scatter Plot)")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#3498DB")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Mükemmel Tahmin")
        ax.set_xlabel("Gerçek Ücret ($)")
        ax.set_ylabel("Tahmin ($)")
        ax.set_title(f"{selected} — Gerçek vs Tahmin")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # Residual Histogram
    with c2:
        st.subheader("📉 Residual (Artık) Dağılımı")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(residuals, bins=50, kde=True, color="#9B59B6", ax=ax)
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Hata ($)")
        ax.set_ylabel("Frekans")
        ax.set_title(f"{selected} — Artık Dağılımı")
        st.pyplot(fig)
        plt.close()

    # Residuals vs Fitted
    st.subheader("🔬 Artıklar vs Tahmin Edilen Değerler")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(y_pred, residuals, alpha=0.3, s=8, color="#E67E22")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Tahmin Edilen Değer ($)")
    ax.set_ylabel("Artık ($)")
    ax.set_title("Artıklar vs Tahminler (Heteroscedasticity Kontrolü)")
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# SEKME 4 — EK ANALİZLER (EDA)
# ════════════════════════════════════════════════════════════════════════════
with tab_extra:
    st.header("🔍 Ek Keşifsel Analizler")

    c1, c2 = st.columns(2)

    # Ücret dağılımı histogramı
    with c1:
        st.subheader("💵 Ücret Dağılımı (Histogram)")
        fare_clip = df_raw[TARGET].clip(upper=df_raw[TARGET].quantile(0.99))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(fare_clip, bins=40, kde=True, color="#27AE60", ax=ax)
        ax.set_xlabel("Ücret ($)")
        ax.set_title("Yolculuk Ücreti Dağılımı")
        st.pyplot(fig)
        plt.close()

    # Havalimanı seyahatleri pie
    with c2:
        st.subheader("✈️ Havalimanı Seyahati Dağılımı")
        if "is_airport_trip" in df_raw.columns:
            airport_counts = df_raw["is_airport_trip"].value_counts()
            labels = ["Havalimanı Dışı", "Havalimanı"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(airport_counts, labels=[labels[i] for i in airport_counts.index],
                   autopct="%1.1f%%", colors=["#3498DB", "#E74C3C"], startangle=90)
            ax.set_title("Havalimanı Seyahat Oranı")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("is_airport_trip kolonu mevcut değil.")

    # Korelasyon Isı Haritası
    st.subheader("🌡️ Korelasyon Matrisi (Heatmap)")
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    corr = df_raw[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Sayısal Değişkenler Korelasyon Matrisi")
    st.pyplot(fig)
    plt.close()

    c3, c4 = st.columns(2)

    # Mesafe vs Ücret scatter
    with c3:
        st.subheader("📏 Mesafe — Ücret İlişkisi")
        sample = df_raw.sample(min(2000, len(df_raw)), random_state=42)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sample["trip_distance"].clip(upper=25), sample[TARGET].clip(upper=80),
                   alpha=0.3, s=8, color="#8E44AD")
        ax.set_xlabel("Mesafe (Mil)")
        ax.set_ylabel("Ücret ($)")
        ax.set_title("Mesafe vs Ücret (Scatter)")
        st.pyplot(fig)
        plt.close()

    # İlçelere göre ortalama ücret
    with c4:
        if "pickup_borough" in df_raw.columns:
            st.subheader("🏙️ İlçelere Göre Ortalama Ücret")
            borough_fare = df_raw.groupby("pickup_borough")[TARGET].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(borough_fare.index, borough_fare.values,
                          color=sns.color_palette("Set2", len(borough_fare)))
            ax.set_xlabel("İlçe")
            ax.set_ylabel("Ort. Ücret ($)")
            ax.set_title("Pickup İlçesine Göre Ort. Ücret")
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig)
            plt.close()
