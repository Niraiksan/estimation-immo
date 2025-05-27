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


df = pd.read_csv("Data/92700-appart-with-distance.csv", sep=";")

print(df.dtypes)


label_enc_id_parcelle = LabelEncoder()
df['id_parcelle'] = label_enc_id_parcelle.fit_transform(df['id_parcelle'])

label_enc_code_secteur_ville = LabelEncoder()
df['code_secteur_ville'] = label_enc_code_secteur_ville.fit_transform(df['code_secteur_ville'])

label_enc_adresse_code_voie = LabelEncoder()
df['adresse_code_voie'] = label_enc_adresse_code_voie.fit_transform(df['code_secteur_ville'])

label_enc_type_appart = LabelEncoder()
df['type_appart'] = label_enc_type_appart.fit_transform(df['type_appart'])


print(df.dtypes)



df.to_csv("Data/92700-appart-encoded.csv", sep=";", index=False)