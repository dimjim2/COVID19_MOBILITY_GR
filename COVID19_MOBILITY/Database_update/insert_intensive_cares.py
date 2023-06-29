import pandas as pd
from utils.dataOperations import insert_data_to_table

'''
Εισάγει τον πανελλήνιο αριθμό ασθενών σε ΜΕΘ ανά ημερομηνία στον πίνακα intensive_care_cases της βάσης
'''
url = 'https://raw.githubusercontent.com/Covid-19-Response-Greece/covid19-data-greece/master/data/greece/general/intensive_care_cases.json'
df_intensive = pd.read_json(url)
df_intensive = pd.json_normalize(df_intensive['cases'])
print(df_intensive)
df_intensive["date"] = pd.to_datetime(df_intensive["date"]).dt.date
df_intensive.rename(columns={"intensive_care": "intensive_care_patients"}, inplace=True)
print(df_intensive)

print("Αρχικό μέγεθος dataset")
print(df_intensive.shape)
# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_intensive.dtypes)

# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns. Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_intensive = df_intensive.drop_duplicates()
print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
print(df_intensive.shape)
print(df_intensive.dtypes)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_intensive.isna().sum())
print("--------------------------------------------")
print(df_intensive.isnull().sum())

# Ανιχνεύει missing values
nan_values = df_intensive.isna()
print(nan_values)
# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()
# Αριθμός στηλών που έχουμε missing values
print("Αριθμός στηλών που έχουμε missing values")
print(nan_columns.sum())

# Γεμίζει με μηδενικές τιμές τα missing values
df_intensive.interpolate(method='linear', inplace=True)
df_intensive = df_intensive.astype({"intensive_care_patients": "int64"})
print(df_intensive["intensive_care_patients"])

'''
Συμπλήρωση ελλειπών τιμών με KNN imputer
# imputer = KNNImputer(n_neighbors=7, weights='uniform', metric='nan_euclidean')
# X = df_intensive["intensive_care"].values.reshape(-1, 1)
# print(X)
# print(X.shape)
# df_intensive["intensive_care"] = imputer.fit_transform(X)
Συμπλήρωση ελλειπών τιμών με Iterative imputer
# it_imputer = IterativeImputer(random_state=0, skip_complete=True)
# X = df_intensive["intensive_care"].values.reshape(-1, 1)
# print(X)
# print(X.shape)
# df_intensive["intensive_care"] = it_imputer.fit_transform(X)
'''

print("Μέγιστος αριθμός ασθενών σε ΜΕΘ")
print(df_intensive.max())

# find_best_distribution_distfit(df_intensive["intensive_care_patients"].values, "Intensive care cases")
#
# find_best_distribution_fitter(df_intensive["intensive_care_patients"].values, "Intensive care cases")

insert_data_to_table(df_intensive, "intensive_care_cases", "(date)")