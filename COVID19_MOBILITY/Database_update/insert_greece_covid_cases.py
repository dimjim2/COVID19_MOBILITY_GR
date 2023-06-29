import pandas as pd
from utils.dataOperations import insert_data_to_table

'''
Εισάγει τα πανελλήνια καθημερινά κρούσματα ανά ημερομηνία στον πίνακα covid_cases_greece της βάσης
'''
df_covid_cases_greece_iMedD = pd.read_csv('https://raw.githubusercontent.com/iMEdD-Lab/open-data/master/COVID-19/greeceTimeline.csv')
print(df_covid_cases_greece_iMedD)

df_covid_cases_greece_iMedD.drop(["Province/State", "Country/Region"], axis=1, inplace=True)
df_covid_cases_greece_iMedD = df_covid_cases_greece_iMedD[df_covid_cases_greece_iMedD['Status'].isin(['cases', 'total cases'])]
df_covid_cases_greece_iMedD = df_covid_cases_greece_iMedD.T
df_covid_cases_greece_iMedD.reset_index(inplace=True)

# Αλλαγή ονομάτων των στηλών
df_covid_cases_greece_iMedD.rename(columns={'index': 'date', 0: 'cases', 21: 'total_cases'}, inplace=True)
df_covid_cases_greece_iMedD = df_covid_cases_greece_iMedD[1:]
print(df_covid_cases_greece_iMedD)

df_covid_cases_greece_iMedD["date"] = pd.to_datetime(df_covid_cases_greece_iMedD["date"]).dt.date
print(df_covid_cases_greece_iMedD)

df_covid_cases_greece = df_covid_cases_greece_iMedD
df_covid_cases_greece = df_covid_cases_greece.astype({'cases': 'float', 'total_cases': 'float'})
print(df_covid_cases_greece)

# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_covid_cases_greece.dtypes)
# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns. Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_covid_cases_greece = df_covid_cases_greece.drop_duplicates()
print(df_covid_cases_greece.shape)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print("--------------------------------------------")
print(df_covid_cases_greece.isnull().sum())

# Ανιχνεύει missing values
nan_values = df_covid_cases_greece.isna()
# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()

# Αριθμός στηλών που έχουμε missing values
print("Αριθμός στηλών που έχουμε missing values")
print(nan_columns.sum())

# Συμπληρώνει με 0 τη στήλη των κρουσμάτων όταν εμφανίζονται missing values
df_covid_cases_greece['cases'].fillna(0, inplace=True)
print(df_covid_cases_greece.isna().sum())
print(df_covid_cases_greece.columns)

print(df_covid_cases_greece)
print(df_covid_cases_greece.shape)
df_covid_cases_greece = df_covid_cases_greece.astype({"cases": "int64", "total_cases": "int64"})

# find_best_distribution_distfit(df_deaths["cases"].values, "Cases Greece")
#
# find_best_distribution_fitter(df_deaths["cases"].values, "Cases Greece")
insert_data_to_table(df_covid_cases_greece, "covid_cases_greece", "(date)")
