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

# ── Sayfa Ayarları & Tema ──────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi Fare Prediction Dashboard",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estetik ayarlar
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120, 
    "axes.titlesize": 14, 
    "axes.labelsize": 12,
    "axes.titleweight": "bold"
})

# ── Ana Başlık ve Açıklama ──────────────────────────────────────────────────
st.title("🚖 NYC Yellow Taxi — Büyük Veri & Regresyon Analizi Dashboard'u")
st.markdown("""
Bu interaktif dashboard, **Adım 7** isterleri doğrultusunda hazırlanmıştır. 
Pipeline üzerinden akan veriler kullanılarak Keşifsel Veri Analizi (EDA) yapılmış ve 
5 farklı regresyon modelinin performans metrikleri görselleştirilmiştir.
""")
st.divider()

# ── Veri Yükleme ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Veriler Delta Lake (Gold) katmanından yükleniyor...")
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
        st.sidebar.error(f"Delta tablosu okunamadı. Örnek veri üretiliyor.\nHata: {exc}")
        rng = np.random.default_rng(42)
        n = 8000
        df = pd.DataFrame({
            "trip_distance": rng.uniform(0.5, 20, n),
            "pickup_hour": rng.integers(0, 24, n),
            "pickup_day_of_week": rng.integers(0, 7, n),
            "passenger_count": rng.integers(1, 5, n),
            "is_airport_trip": rng.integers(0, 2, n),
            "RatecodeID": rng.integers(1, 6, n),
            "payment_type": rng.integers(1, 5, n),
            "pickup_borough": rng.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n),
            "dropoff_borough": rng.choice(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], n),
        })
        df["fare_amount"] = (
            3.0 + df["trip_distance"] * 2.5 + df["pickup_hour"] * 0.3 + rng.normal(0, 2, n)
        )
        return df

df_raw = load_data()
TARGET = "label" if "label" in df_raw.columns else "fare_amount"

# ── Model Eğitimi (Cache) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Makine Öğrenmesi modelleri eğitiliyor...")
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
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
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

# ── Sekme Yapısı ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Zorunlu Görseller (Dağılım & Trend)", 
    "🤖 Model Karşılaştırmaları", 
    "🎯 Regresyon Analizi & Ek EDA"
])

# =============================================================================
# SEKME 1: ZORUNLU GÖRSELLER (Dağılım ve Trend)
# =============================================================================
with tab1:
    st.header("1. Veri Dağılımı ve Zaman Serisi Trendleri")
    st.markdown("Proje isterlerinde belirtilen zorunlu **histogram**, **pie chart** ve **line chart** grafikleri bu bölümde sunulmaktadır.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Veri Dağılımı: Yolculuk Mesafesi (Histogram)")
        st.markdown("Yolculukların ne kadarlık mesafelerde yoğunlaştığını gösteren frekans grafiği.")
        fig, ax = plt.subplots(figsize=(6, 4))
        clip_dist = df_raw["trip_distance"].clip(upper=df_raw["trip_distance"].quantile(0.98))
        sns.histplot(clip_dist, bins=30, kde=True, ax=ax, color="#2980B9", edgecolor="white")
        ax.set_xlabel("Mesafe (Mil)")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Veri Dağılımı: Yolcu Sayısı (Pie Chart)")
        st.markdown("Taksi yolculuklarındaki yolcu sayılarının oransal dağılımı.")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_raw["passenger_count"].value_counts().sort_index()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"), startangle=140, wedgeprops={'edgecolor': 'white'})
        st.pyplot(fig)
        plt.close()

    st.divider()

    st.subheader("Zaman Serisi Trend Grafiği: Saatlik Ortalama Ücret (Line Chart)")
    st.markdown("Günün farklı saatlerinde taksi ücretlerindeki değişimi ve trendi gösteren çizgi grafik.")
    hourly = df_raw.groupby("pickup_hour")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly.index, hourly.values, marker="o", color="#E74C3C", linewidth=2.5, markersize=8)
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#E74C3C")
    ax.set_xlabel("Günün Saati (0-23)")
    ax.set_ylabel("Ortalama Ücret ($)")
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig)
    plt.close()


# =============================================================================
# SEKME 2: MODEL KARŞILAŞTIRMALARI
# =============================================================================
with tab2:
    st.header("2. 5 Modelin Performans Karşılaştırması ve Özellik Önemleri")
    
    # Metrikleri DataFrame'e Çevirme
    mdf = pd.DataFrame([
        {"Model": n, "RMSE": v["rmse"], "MAE": v["mae"], "R²": v["r2"]}
        for n, v in results.items()
    ])

    best_model = mdf.loc[mdf["R²"].idxmax()]
    st.success(f"**En İyi Model:** {best_model['Model']} (R²: {best_model['R²']:.4f})")

    st.subheader("5 Modelin Performans Karşılaştırma Grafiği (Grouped Bar Chart)")
    st.markdown("Regresyon problemleri için modellerin **RMSE**, **MAE** ve **R²** skorlarının yan yana gruplanmış bar grafik ile karşılaştırması.")
    
    mdf_melt = mdf.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R²"], var_name="Metrik", value_name="Skor")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=mdf_melt, x="Model", y="Skor", hue="Metrik", palette=["#E74C3C", "#F39C12", "#2ECC71"], ax=ax)
    ax.set_xlabel("Modeller", fontweight='bold')
    ax.set_ylabel("Skor Değeri", fontweight='bold')
    plt.xticks(rotation=15)
    ax.legend(title="Metrikler", loc="upper right")
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 8), textcoords = 'offset points', fontsize=9)
    st.pyplot(fig)
    plt.close()

    st.divider()

    st.subheader("Feature Importance Grafiği (Horizontal Bar Chart)")
    st.markdown("En iyi sonuç veren ağaç tabanlı modelimizin (Random Forest) taksi ücretini belirlerken hangi özelliklere en çok ağırlık verdiğinin sıralaması.")
    
    rf_res = results["Random Forest"]
    fi = pd.Series(rf_res["model"].feature_importances_, index=rf_res["features"]).nlargest(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    fi.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(fi)))
    ax.set_xlabel("Özellik Önem Skoru", fontweight='bold')
    ax.set_ylabel("Özellikler (Features)", fontweight='bold')
    st.pyplot(fig)
    plt.close()


# =============================================================================
# SEKME 3: REGRESYON ANALİZİ & EK EDA GÖRSELLERİ
# =============================================================================
with tab3:
    st.header("3. Regresyon Analiz Grafikleri ve Ek EDA Görselleri")
    
    st.markdown("### Regresyon Problemleri İçin Ek İsterler")
    selected_model = st.selectbox("Grafikleri çizmek için bir model seçin:", list(results.keys()), index=2)
    res = results[selected_model]
    y_test, y_pred = res["y_test"], res["y_pred"]
    residuals = y_test - y_pred

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gerçek vs Tahmin (Scatter Plot)")
        st.markdown("Modelin tahmin ettiği ücret ile gerçek ücretin dağılımı. Kırmızı çizgi mükemmel tahmini temsil eder.")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="#8E44AD")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=2, label="Mükemmel Tahmin (y=x)")
        ax.set_xlabel("Gerçek Ücret ($)")
        ax.set_ylabel("Tahmin Edilen Ücret ($)")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Residual (Artık) Dağılım Grafiği")
        st.markdown("Hataların (Gerçek - Tahmin) dağılımı. İdeal durumda 0 etrafında normal dağılması beklenir.")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(residuals, bins=40, kde=True, color="#E67E22", ax=ax)
        ax.axvline(0, color="red", linestyle="--", lw=2)
        ax.set_xlabel("Hata Miktarı ($)")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        plt.close()

    st.divider()

    st.markdown("### EDA Bulgularını Özetleyen Ek 3 Görselleştirme (Zorunlu İster)")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("Ek 1: Ücret Dağılımı")
        st.markdown("Genel ücret dağılımının histogramı.")
        fare_clip = df_raw[TARGET].clip(upper=df_raw[TARGET].quantile(0.98))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(fare_clip, bins=30, kde=True, color="#1ABC9C", ax=ax)
        ax.set_xlabel("Ücret ($)")
        ax.set_ylabel("Frekans")
        st.pyplot(fig)
        plt.close()

    with c2:
        if "pickup_borough" in df_raw.columns:
            st.subheader("Ek 2: İlçelere Göre Ücret")
            st.markdown("Başlangıç ilçesine göre ortalama kazançlar.")
            borough_fare = df_raw.groupby("pickup_borough")[TARGET].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=borough_fare.index, y=borough_fare.values, palette="rocket", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Ortalama Ücret ($)")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("İlçe verisi bulunamadı.")

    with c3:
        st.subheader("Ek 3: Korelasyon Isı Haritası")
        st.markdown("Sayısal öznitelikler arası ilişki matrisi.")
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        corr = df_raw[num_cols].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False, cbar=True, ax=ax)
        st.pyplot(fig)
        plt.close()
