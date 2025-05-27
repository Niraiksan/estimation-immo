import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm
import csv
import seaborn as sns
import matplotlib.pyplot as plt


def compter_poi_proximite(lat_bien, lon_bien, df_poi, rayon_km):
    centre = (lat_bien, lon_bien)
    count = 0

    for _, row in df_poi.iterrows():
        dist = geodesic(centre, (row['lat'], row['lon'])).km
        if dist <= rayon_km:
            count += 1
    return count


def centre_ville_distance(lat_bien, lon_bien, df_poi, nom_commune):
    centre_bien = (lat_bien, lon_bien)
    
    df_poi_centreville = df_poi[
        (df_poi['poi'] == 'townhall') & (df_poi['city'] == nom_commune)
    ][['name', 'lat', 'lon']]
    
    if df_poi_centreville.empty:
        return None  # ou float('nan'), selon ce que tu veux
    
    df_poi_centreville = df_poi_centreville.head(1)
    
    mairie_lat = df_poi_centreville['lat'].iloc[0]
    mairie_lon = df_poi_centreville['lon'].iloc[0]
    
    dist = geodesic(centre_bien, (mairie_lat, mairie_lon)).km
    return dist


###################################################################
###################################################################
df_biens = pd.read_csv("Data/92-appart-type-appart.csv", sep=";")

df_biens = df_biens[df_biens['nom_commune']=="Clichy"]

df_poi_colombes = pd.read_csv("POI-data/92-poi.csv", sep=";")

distance_m = 500
distance_km = distance_m/1000

df_colombes_ecoles = df_poi_colombes[df_poi_colombes['poi'].isin(['school','college'])][['name','lat', 'lon']]
df_colombes_ecoles['lat'] = df_colombes_ecoles['lat'].astype(float)
df_colombes_ecoles['lon'] = df_colombes_ecoles['lon'].astype(float)
print(df_colombes_ecoles)


df_colombes_gares = df_poi_colombes[df_poi_colombes['poi'] == 'station'][['name','lat', 'lon']]
df_colombes_gares['lat'] = df_colombes_gares['lat'].astype(float)
df_colombes_gares['lon'] = df_colombes_gares['lon'].astype(float)
print(df_colombes_gares)


# df_colombes_transports = df_poi_colombes[df_poi_colombes['poi'].isin(['bus_stop', 'tram_stop', 'station'])][['name', 'lat', 'lon']]
# df_colombes_transports['lat'] = df_colombes_transports['lat'].astype(float)
# df_colombes_transports['lon'] = df_colombes_transports['lon'].astype(float)
# print(df_colombes_transports)


df_colombes_supermarket = df_poi_colombes[df_poi_colombes['poi'].isin(['supermarket'])][['name', 'lat', 'lon']]
df_colombes_supermarket['lat'] = df_colombes_supermarket['lat'].astype(float)
df_colombes_supermarket['lon'] = df_colombes_supermarket['lon'].astype(float)
print(df_colombes_supermarket)


df_colombes_centrecommercial = df_poi_colombes[df_poi_colombes['poi'].isin(['mall'])][['name', 'lat', 'lon']]
df_colombes_centrecommercial['lat'] = df_colombes_centrecommercial['lat'].astype(float)
df_colombes_centrecommercial['lon'] = df_colombes_centrecommercial['lon'].astype(float)
print(df_colombes_centrecommercial)


df_colombes_hospital = df_poi_colombes[df_poi_colombes['poi'].isin(['hospital'])][['name', 'lat', 'lon']]
df_colombes_hospital['lat'] = df_colombes_hospital['lat'].astype(float)
df_colombes_hospital['lon'] = df_colombes_hospital['lon'].astype(float)
print(df_colombes_hospital)



tqdm.pandas()
df_biens['centre_ville_distance'] = df_biens.progress_apply(
    lambda row: centre_ville_distance(row['latitude'], row['longitude'], df_poi_colombes, "Clichy-la-Garenne"), axis=1
)

tqdm.pandas()
nom_colonne = "nb_ecole_proximite_" + str(distance_m) + "m"
df_biens[nom_colonne] = df_biens.progress_apply(
    lambda row: compter_poi_proximite(row['latitude'], row['longitude'], df_colombes_ecoles, rayon_km=distance_km), axis=1
)

tqdm.pandas()
nom_colonne =  "nb_gare_proximite_" + str(distance_m) + "m"
df_biens[nom_colonne] = df_biens.progress_apply(
    lambda row: compter_poi_proximite(row['latitude'], row['longitude'], df_colombes_gares, rayon_km=distance_km), axis=1
)

# tqdm.pandas()
# nom_colonne = "nb_transports_proximite_" + str(distance_m) + "m"
# df_biens[nom_colonne] = df_biens.progress_apply(
#     lambda row: compter_poi_proximite(row['latitude'], row['longitude'], df_colombes_transports, rayon_km=distance_km), axis=1
# )

tqdm.pandas()
nom_colonne = "nb_supermarket_" + str(distance_m) + "m"
df_biens[nom_colonne] = df_biens.progress_apply(
    lambda row: compter_poi_proximite(row['latitude'], row['longitude'], df_colombes_supermarket, rayon_km=distance_km), axis=1
)

tqdm.pandas()
nom_colonne = "nb_centrecommercial_proximite_" + str(distance_m) + "m"
df_biens[nom_colonne] = df_biens.progress_apply(
    lambda row: compter_poi_proximite(row['latitude'], row['longitude'], df_colombes_centrecommercial, rayon_km=distance_km), axis=1
)

tqdm.pandas()
nom_colonne = "nb_hospital_" + str(distance_m) + "m"
df_biens[nom_colonne] = df_biens.progress_apply(
    lambda row: compter_poi_proximite(row['latitude'], row['longitude'], df_colombes_hospital, rayon_km=distance_km), axis=1
)


#################################### OBSERVATION #########################################


# # df = df_biens[(df_biens["prix_m2"] > 2000) & (df_biens["prix_m2"] <= 15000)]
df = df_biens

# print("")
# print("")
# print("")
print(df)

df.to_csv('Data/92-clichy-appart-with-distance.csv', sep=";", index=False, encoding='utf-8')


correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()