import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from seaborn import boxplot
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def attribuer_type_appart(surface):
    if pd.isna(surface):
        return None
    if surface < 30:
        return "T1"
    elif surface < 45:
        return "T2"
    elif surface < 65:
        return "T3"
    elif surface < 85:
        return "T4"
    elif surface < 100:
        return "T5"
    else:
        return "T6+"

###############################
## SELECTION PRINCIPAL DATA ###
###############################

df = pd.read_csv("Data/dvf-92.csv", delimiter=';', encoding="utf-8")


### GRAPHIQUE : nombre d'occurrences par nom_commune
# ville_counts = df['nom_commune'].value_counts()

# plt.figure(figsize=(8, 8))
# plt.pie(ville_counts, labels=ville_counts.index, autopct='%1.1f%%', startangle=90)
# plt.title('Répartition par nom_commune')
# plt.axis('equal')
# plt.show()

# df = df[df['nom_commune'] == 'Colombes']


### GRAPHIQUE : nombre d'occurrences par type_local
# categorie_counts = df['type_local'].value_counts()

# plt.figure(figsize=(8, 8))
# plt.pie(categorie_counts, labels=categorie_counts.index, autopct='%1.1f%%', startangle=90)
# plt.title('Répartition des lignes par type_local')
# plt.axis('equal')
# plt.show()

df = df[df['type_local'] == 'Appartement']


### GRAPHIQUE : nombre d'occurrences par nature_mutation
# nature_mutation_counts = df['nature_mutation'].value_counts()

# plt.figure(figsize=(8, 8))
# plt.pie(nature_mutation_counts, labels=nature_mutation_counts.index, autopct='%1.1f%%', startangle=90)
# plt.title('Répartition des lignes par nature_mutation')
# plt.axis('equal')
# plt.show()

df = df[df['nature_mutation'].isin(["Vente","Vente en l'état futur d'achèvement"])]


df = df[df['code_nature_culture'].isna()]

df = df[df['valeur_fonciere'].notna()]
df = df[df['longitude'].notna()]
df = df[df['latitude'].notna()]




####################
## AJOUT COLONNE ###
####################

carrez_cols = [
    "lot1_surface_carrez",
    "lot2_surface_carrez",
    "lot3_surface_carrez",
    "lot4_surface_carrez",
    "lot5_surface_carrez"
]

df[carrez_cols] = df[carrez_cols].apply(pd.to_numeric, errors='coerce')

df["nb_lots_renseignes"] = df[carrez_cols].notna().sum(axis=1)
df["lots_surface_total"] = df[carrez_cols].sum(axis=1, skipna=True).fillna(0)
df["lots_surface_logement"] = df[carrez_cols].max(axis=1, skipna=True)
df["lots_surface_autre_lots"] = df["lots_surface_total"] - df["lots_surface_logement"]

mask = df["nb_lots_renseignes"] == 0
df.loc[mask, "lots_surface_total"] = df.loc[mask, "surface_reelle_bati"]
df.loc[mask, "lots_surface_logement"] = df.loc[mask, "surface_reelle_bati"]
df.loc[mask, "lots_surface_autre_lots"] = 0

df = df[df['lots_surface_logement'].notna()]


df['prix_m2'] = df['valeur_fonciere'] / df['lots_surface_logement']

df['code_secteur_ville'] = df['id_parcelle'].astype(str).str[:-4]

df['construction_recente'] = (df['nature_mutation'] == "Vente en l'état futur d'achèvement").astype(int)

df["type_appart"] = df["lots_surface_logement"].apply(attribuer_type_appart)


### GRAPHIQUE : nombre d'occurrences par type_appart
# type_appart_counts = df['type_appart'].value_counts()

# plt.figure(figsize=(8, 8))
# plt.pie(type_appart_counts, labels=type_appart_counts.index, autopct='%1.1f%%', startangle=90)
# plt.title('Répartition des lignes par type_appart')
# plt.axis('equal')
# plt.show()



#######################
## VALEUR ABERRANTE ###
#######################

print("AVANT TRAITEMENT ABERRANT")
print(df.shape)


#### GESTION id_mutation doublon ####
mutation_counts = df['id_mutation'].value_counts()
double_mutation = mutation_counts[mutation_counts.values>1]
index_double_mutation = double_mutation.index 

df = df[~df['id_mutation'].isin(index_double_mutation)]

print("APRES TRAITEMENT id_mutation doublon")
print(df.shape)



#### SURFACE ABERRANTE #####

# sns.boxplot(y=df["lots_surface_logement"])
# plt.show()

for i in range(5) :
    Q1 = df["lots_surface_logement"].quantile(0.25)
    Q3 = df["lots_surface_logement"].quantile(0.75)
    IQR = Q3 - Q1

    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    df = df[(df["lots_surface_logement"] >= borne_basse) & (df["lots_surface_logement"] <= borne_haute)]

    # sns.boxplot(y=df["lots_surface_logement"])
    # plt.show()


print("APRES TRAITEMENT SURFACE ABERRANTE")
print(df.shape)



#### PRIX_M2 ABERRANTE #####

# sns.boxplot(y=df["prix_m2"])
# plt.show()

for i in range(5) :
    Q1 = df["prix_m2"].quantile(0.25)
    Q3 = df["prix_m2"].quantile(0.75)
    IQR = Q3 - Q1

    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    df = df[(df["prix_m2"] >= borne_basse) & (df["prix_m2"] <= borne_haute)]

    # sns.boxplot(y=df["prix_m2"])
    # plt.show()


print("APRES TRAITEMENT PRIX_M2 ABERRANTE")
print(df.shape)



#### VALEUR FONCIERE ABERRANTE #####

# sns.boxplot(y=df["valeur_fonciere"])
# plt.show()

for i in range(5) :
    Q1 = df["valeur_fonciere"].quantile(0.25)
    Q3 = df["valeur_fonciere"].quantile(0.75)
    IQR = Q3 - Q1

    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    df = df[(df["valeur_fonciere"] >= borne_basse) & (df["valeur_fonciere"] <= borne_haute)]

    # sns.boxplot(y=df["valeur_fonciere"])
    # plt.show()


print("APRES TRAITEMENT valeur_fonciere ABERRANTE")
print(df.shape)




print("APRES TOUS LES TRAITEMENTS ABERRANTS")
print(df.shape)


# ## GRAPHIQUE
# sns.boxplot(y=df["valeur_fonciere"])
# plt.show()


# ## GRAPHIQUE
# sns.boxplot(y=df["lots_surface_logement"])
# plt.show()


# ## GRAPHIQUE
# sns.boxplot(y=df["prix_m2"])
# plt.show()


# # ## GRAPHIQUE
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x="type_appart", y="prix_m2", palette="Set2")

# plt.title("Distribution du prix au m² par type d'appartement")
# plt.xlabel("Type d'appartement")
# plt.ylabel("Prix au m² (€)")
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# ## GRAPHIQUE
# df_selected=df

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     df_selected['longitude'], df_selected['latitude'],
#     c=df_selected['prix_m2'], cmap='viridis', s=20
# )
# plt.colorbar(scatter, label='Prix')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Répartition géographique des biens')
# plt.show()





##########################
## COLONNE A CONSERVER ###
##########################

cols = [
    'prix_m2', 
    'nom_commune','longitude', 'latitude', 'code_secteur_ville', 'id_parcelle', 'adresse_code_voie',
    'construction_recente', 'type_appart'
    ]


df[cols].to_csv('Data/92-appart-type-appart.csv', sep=";", index=False, encoding='utf-8')







###########################################################################################################
###########################################################################################################
##########################################  TEST  #########################################################
###########################################################################################################
###########################################################################################################

# label_enc_id_parcelle = LabelEncoder()
# df['id_parcelle'] = label_enc_id_parcelle.fit_transform(df['id_parcelle'])

# label_enc_code_secteur_ville = LabelEncoder()
# df['code_secteur_ville'] = label_enc_code_secteur_ville.fit_transform(df['code_secteur_ville'])

# label_enc_adresse_code_voie = LabelEncoder()
# df['adresse_code_voie'] = label_enc_adresse_code_voie.fit_transform(df['code_secteur_ville'])

# label_enc_type_appart = LabelEncoder()
# df['type_appart'] = label_enc_type_appart.fit_transform(df['type_appart'])

# label_enc_nom_commune = LabelEncoder()
# df['nom_commune'] = label_enc_nom_commune.fit_transform(df['nom_commune'])

# print(df[cols].dtypes)

# correlation_matrix = df[cols].corr(numeric_only=True)
# plt.figure(figsize=(10,8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Matrice de Corrélation')
# plt.show()

# df_selected = df[cols]

# y = df_selected['prix_m2']
# X = df_selected.drop('prix_m2', axis=1)

# model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)

# scores = cross_val_score(model_xgb, X, y, cv=5, scoring='r2')
# print("Scores R2 pour chaque fold :", scores)
# print("Score R2 moyen :", scores.mean())

# model_xgb.fit(X, y)

# importances = model_xgb.feature_importances_
# features = X.columns

# feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# print(feature_importance_df)
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.gca().invert_yaxis()
# plt.title("Importance des variables (XGBoost)")
# plt.xlabel("Importance")
# plt.show()