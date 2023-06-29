import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from utils.dataOperations import insert_data_to_table
'''
Εισάγει τα συνολικά κρούσματα ανά περιοχή της Ελλάδας στον πίνακα covid_area_cumulative_cases της βάσης
'''
url = 'https://raw.githubusercontent.com/Covid-19-Response-Greece/covid19-data-greece/master/data/greece/regional/regions_cumulative.csv'
df_cases_by_area_cumulative = pd.read_csv(url)
df_cases_by_area_cumulative["last_updated_at"] = pd.to_datetime(df_cases_by_area_cumulative["last_updated_at"]).dt.date
print(df_cases_by_area_cumulative)

df_cases_by_area_cumulative.drop(["total_deaths", "area_en", "region_gr", "region_en",
                              "geo_department_en", "longtitude", "latitude"], axis=1, inplace=True)

print(df_cases_by_area_cumulative)
print("Αρχικό μέγεθος dataset")
print(df_cases_by_area_cumulative.shape)
# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_cases_by_area_cumulative.dtypes)
df_cases_by_area_cumulative.loc[df_cases_by_area_cumulative['area_gr'] == 'ΑΓΙΟ ΟΡΟΣ', ['geo_department_gr']] = 'Αυτόνομη Μοναστική Πολιτεία'
# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns. Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_cases_by_area_cumulative = df_cases_by_area_cumulative.drop_duplicates()
print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
print(df_cases_by_area_cumulative.shape)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_cases_by_area_cumulative.isna().sum())
print("--------------------------------------------")
print(df_cases_by_area_cumulative.isnull().sum())

# Ανιχνεύει missing values
nan_values = df_cases_by_area_cumulative.isna()
# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()
# Αριθμός στηλών που έχουμε missing values
print("Αριθμός στηλών που έχουμε missing values")
print(nan_columns.sum())

sns.displot(df_cases_by_area_cumulative["total_cases"])
plt.show()

insert_data_to_table(df_cases_by_area_cumulative, "covid_area_cumulative_cases", "(area_gr)")