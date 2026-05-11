import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. VERİ OKUMA VE HAZIRLIK ---
# df = pd.read_parquet("arkadasinin_hazirladigi_veri.parquet")

print("Veri yükleniyor...")

# Hedef değişken (y) ve özellikler (X)
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']
features = X.columns

# Eğitim (%80) ve Test (%20) ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. MODELLERİN TANIMLANMASI ---
# PDF Adım 6'da istenen 5 regresyon modeli:
models = {
    "Linear_Regression": LinearRegression(),
    "Decision_Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random_Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "Gradient_Boosted_Trees": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "Generalized_Linear_Ridge": Ridge(alpha=1.0)
}

# --- 3. MLFLOW ENTEGRASYONU VE EĞİTİM ---
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow sunucu adresi
mlflow.set_experiment("NYC_Taxi_Fare_Prediction")

# Grafikleri geçici kaydetmek için klasör
os.makedirs("temp_plots", exist_ok=True)

print("Model eğitimleri ve MLflow loglaması başlıyor...")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"-> {model_name} eğitiliyor...")

        # Modeli Eğit ve Tahmin Yap
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # PDF Adım 6 - Zorunlu Metrikler
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Metrikleri ve Parametreleri MLflow'a Logla
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # --- PDF Adım 7 - Zorunlu Grafikler ---

        # 1. Gerçek vs Tahmin Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Gerçek Ücret ($)')
        plt.ylabel('Tahmin Edilen Ücret ($)')
        plt.title(f'{model_name} - Gerçek vs Tahmin')
        plot_path_scatter = f"temp_plots/{model_name}_scatter.png"
        plt.savefig(plot_path_scatter)
        mlflow.log_artifact(plot_path_scatter)
        plt.close()

        # 2. Residual (Artık) Dağılım Grafiği
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, bins=50, kde=True, color='purple')
        plt.xlabel('Hata Miktarı ($)')
        plt.ylabel('Frekans')
        plt.title(f'{model_name} - Residual (Artık) Dağılımı')
        plot_path_residual = f"temp_plots/{model_name}_residual.png"
        plt.savefig(plot_path_residual)
        mlflow.log_artifact(plot_path_residual)
        plt.close()

        # 3. Feature Importance (Sadece Ağaç tabanlı modeller için)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(model.feature_importances_, index=features)
            feat_importances.nlargest(5).plot(kind='barh', color='teal')
            plt.title(f'{model_name} - Feature Importance')
            plot_path_fi = f"temp_plots/{model_name}_feature_importance.png"
            plt.savefig(plot_path_fi)
            mlflow.log_artifact(plot_path_fi)
            plt.close()

        # Eğitilmiş Modeli MLflow'a Kaydet
        mlflow.sklearn.log_model(model, "model")
        print(f"   [Tamamlandı] RMSE: {rmse:.2f} | R2: {r2:.2f}")

print("\nTüm modeller başarıyla eğitildi ve MLflow'a kaydedildi!")
print(
    "Sonuçları görmek için terminalde 'mlflow ui' komutunu çalıştırın ve tarayıcıda http://localhost:5000 adresine gidin.")
