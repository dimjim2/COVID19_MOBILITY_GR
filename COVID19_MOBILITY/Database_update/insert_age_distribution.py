from utils.dataOperations import insert_data_to_table
import pandas as pd
'''
Εισάγει τα δημογραφικά δεδομένα στον πίνακα age_distribution της βάσης.
Τα δημογραφικά δεδομένα αποτελούνται τα κρούσματα, τους θανάτους και τους εμβολιασμούς των ανδρών
και των γυναικών στα ηλικιακά γκρουπ 0-17, 18-39, 40-64, 65+
'''

df_men = pd.read_csv('https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/demography_men_details.csv')
print("Άντρες")
print(df_men)

# Μετασχηματίζει το DataFrame με βάσει το index / column values.
df_men = df_men.pivot_table(['cases', 'deaths', 'intensive'], 'date', 'category')
df_men.columns = "men_" + df_men.columns.get_level_values(0) + '_' + df_men.columns.get_level_values(1)
df_men.reset_index(inplace=True)

print(df_men)
print(df_men.columns)
print(df_men.dtypes)


df_women = pd.read_csv('https://raw.githubusercontent.com/Sandbird/covid19-Greece/master/demography_women_details.csv')
print("Γυναίκες")
print(df_women)
df_women = df_women.pivot_table(['cases', 'deaths', 'intensive'], 'date', 'category')

df_women.columns = "women_" + df_women.columns.get_level_values(0) + '_' + df_women.columns.get_level_values(1)
print(df_women)
print(df_women.columns)

# Συγχωνεύει τα dataframes που περιλαμβάνουν υγειονομικά δεδομένα των ανδρών και των γυναικών
# σε ένα ενιαίο με βάση την ημερομηνία
df_demographics = pd.merge(df_men, df_women, on='date')
print("Δημογραφικά")
print(df_demographics)
print(df_demographics.columns)
df_demographics["date"] = pd.to_datetime(df_demographics["date"]).dt.date

print("Αρχικό μέγεθος dataset")
print(df_demographics.shape)

# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_demographics.dtypes)
# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns.Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_demographics = df_demographics.drop_duplicates()
print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
print(df_demographics.shape)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_demographics.isna().sum())
print("--------------------------------------------")
print(df_demographics.isnull().sum())

# Ανιχνεύει missing values
nan_values = df_demographics.isna()

# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()

print("Αριθμός στηλών που έχουμε missing values")
# Αριθμός στηλών που έχουμε missing values
print(nan_columns.sum())

# Επειδή υπάρχει μεγάλη χρονική απόσταση μεταξύ των δεδομένων διαγράφουμε τις δύο πρώτες εγγραφές
# Έχουμε μόνο δύο εγγραφές στην αρχή 2020-01-25,2020-01-26 για τον Γενάρη και μετά από Απρίλιο
df_demographics = df_demographics.iloc[2:, :]
print(df_demographics)

insert_data_to_table(df_demographics, "age_distribution", "(date)")