import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.dataOperations import insert_data_to_table
'''
Εισάγει τον πανελλήνιο αριθμό θανάτων ανά ημερομηνία στον πίνακα deaths της βάσης
'''

df_deaths = pd.read_csv('https://raw.githubusercontent.com/iMEdD-Lab/open-data/master/COVID-19/greeceTimeline.csv')
df_deaths.drop(["Province/State", "Country/Region"], axis=1, inplace=True)
df_deaths = df_deaths[df_deaths['Status'].isin(['deaths', 'deaths_cum'])]
# Αναστροφή πίνακα
df_deaths = df_deaths.T

df_deaths.reset_index(inplace=True)
# Μεταονομασία των στηλών
df_deaths.rename(columns={'index': 'date', 3: 'deaths', 4: 'deaths_cum'}, inplace=True)
df_deaths = df_deaths[1:]
df_deaths = df_deaths.reset_index(drop=True)
print(df_deaths)

df_deaths["date"] = pd.to_datetime(df_deaths["date"]).dt.date

# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_deaths.dtypes)
# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns.Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_deaths = df_deaths.drop_duplicates()
print(df_deaths.shape)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print("--------------------------------------------")
print(df_deaths.isnull().sum())

# Ανιχνεύει missing values
nan_values = df_deaths.isna()
# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()
# Αριθμός στηλών που έχουμε missing values
print("Αριθμός στηλών που έχουμε missing values")
print(nan_columns.sum())

# Όταν λείπουν τιμές στους θανάτους των εγγραφών τις συμπληρώνει με τον τρέχοντα αριθμό θανάτων που έχει παρατηρηθεί
# στην περίπτωση των αθροιστικών δεδομένων ενώ με 0 στον ημερίσιο δείκτη
df_deaths['deaths_cum'].fillna(df_deaths['deaths_cum'].max(), inplace=True)
df_deaths['deaths'].fillna(0, inplace=True)
print(df_deaths.isna().sum())

print(df_deaths.columns)
# Αφαιρούμε την εγγραφή στις 2022-05-16 λόγω σφάλματος που προέκυψε από τον τρόπο υπολογισμού των θανάτων
df_deaths['date'] = pd.to_datetime(df_deaths['date'], format='%Y-%m-%d')
df_deaths = df_deaths.loc[(df_deaths['date'] != '2022-05-16')]
df_deaths["date"] = pd.to_datetime(df_deaths["date"]).dt.date

print(df_deaths)
print(df_deaths.shape)

df_deaths = df_deaths.astype({"deaths": "int64", "deaths_cum": "int64"})

sns.displot(df_deaths["deaths"])
plt.show()

# find_best_distribution_distfit(df_deaths["deaths"].values, "Deaths")
#
# find_best_distribution_fitter(df_deaths["deaths"].values, "Deaths")


insert_data_to_table(df_deaths, "deaths", "(date)")