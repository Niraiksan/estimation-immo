import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Chargement des données
df = pd.read_csv("Data/92700-appart-encoded.csv", sep=";")
X = df.drop("prix_m2", axis=1)
y = df["prix_m2"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

# Modèles à évaluer
models = {
    "Régression Linéaire": make_pipeline(StandardScaler(), LinearRegression()),
    "Arbre de Décision": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

summary_rows = []

print("=== Évaluation croisée sur TRAIN + test final sur TEST ===\n")

for name, model in models.items():
    print(f"{name}")

    # Évaluation croisée sur le train uniquement
    r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    print(f"   R² CV train     : {r2_mean:.3f} (± {r2_std:.3f})")

    # Entraînement sur tout le train
    model.fit(X_train, y_train)

    # Prédictions sur le test
    y_pred_test = model.predict(X_test)

    # Métriques sur test
    r2_test = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    print(f"   R² test         : {r2_test:.3f}")
    print(f"   RMSE test       : {rmse:.2f}")
    print(f"   MSE test        : {mse:.2f}")
    print(f"   MAE test        : {mae:.2f}")

    # Importance des variables
    top_vars = []
    try:
        if name == "Régression Linéaire":
            coefs = model.named_steps["linearregression"].coef_
            importance = np.abs(coefs)
            importance_pct = importance / importance.sum()
            importance_df = pd.DataFrame({
                "Variable": X.columns,
                "Importance": importance_pct
            }).sort_values(by="Importance", ascending=False)
        else:
            if name == "CatBoost":
                importances = model.feature_importances_ / 100
                importance_df = pd.DataFrame({
                    "Variable": X.columns,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)
            else:
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Variable": X.columns,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)

        top_vars = importance_df.head(5).apply(
            lambda row: f"{row['Variable']}\nImportance : {row['Importance']:.2f}", axis=1
        ).tolist()

    except AttributeError:
        top_vars = ["Non disponible"] * 5

    summary_rows.append({
        "Modèle": name,
        "R² CV train": f"{r2_mean:.3f} ± {r2_std:.3f}",
        "R² test": f"{r2_test:.3f}",
        "RMSE test": f"{rmse:.2f}",
        "MSE test": f"{mse:.2f}",
        "MAE test": f"{mae:.2f}",
        "Top 1": top_vars[0],
        "Top 2": top_vars[1],
        "Top 3": top_vars[2],
        "Top 4": top_vars[3],
        "Top 5": top_vars[4],
    })

    # Visualisation des prédictions vs vrai
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Vrai prix au m²")
    plt.ylabel("Prix prédit")
    plt.title(f"Prédictions vs Réel - {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Colombes-PrixM2-PrédVSRéel-{name}.png", dpi=300)
    plt.show()

    print("-" * 60)

# Résumé sous forme de tableau
summary_df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(20, 5 + len(summary_df) * 0.6))
ax.axis('off')

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.5, 1.5)

plt.title("Résumé des performances (CV sur train + test final)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("résumé_modèles.png", dpi=300)
plt.show()
