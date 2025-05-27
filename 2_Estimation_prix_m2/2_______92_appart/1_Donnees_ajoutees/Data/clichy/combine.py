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



#######################################################################################
###  FUSION DES DATASETS DEPARTEMENTALE POUR AVOIR UNE DATASET SUR L'ILE-DE-FRANCE  ###
#######################################################################################

fichiers_csv = ['92-clichy-appart-with-distance.csv', '92-sans-clichy-appart-with-distance.csv']

dataframes = []

for fichier in fichiers_csv:
    df = pd.read_csv(fichier, sep=';', encoding='utf-8')
    dataframes.append(df)

df_combine = pd.concat(dataframes, ignore_index=True)

df_combine.to_csv('92-appart-with-distance.csv', sep=";", index=False, encoding='utf-8')