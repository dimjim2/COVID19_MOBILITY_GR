from datetime import datetime

import pandas as pd
from utils.dataOperations import insert_data_to_table

'''
Εισάγει τα συνολικά κρούσματα ανά περιοχή της Ελλάδας και ημερομηνία στον πίνακα covid_by_area_history_cases
'''

df_covid_cases_by_area = pd.read_csv('https://raw.githubusercontent.com/Covid-19-Response-Greece/covid19-data-greece/master/data/greece/regional/regions_history_cases.csv')

cols = ["area_gr", "area_en", "region_gr", "region_en", "geo_department_gr", "geo_department_en",
        "last_updated_at", "longtitude", "latitude", "population"]
df_covid_cases_by_area = df_covid_cases_by_area.melt(id_vars=cols,
                                                     var_name="date",
                                                     value_name="cases")
# Αφαιρούμε τις ακόλουθες στήλες από το dataframe
cols_del = ["area_en", "region_gr", "region_en", "geo_department_gr", "geo_department_en", "last_updated_at", "longtitude",
        "latitude", "population"]
df_covid_cases_by_area.drop(labels=cols_del, axis=1, inplace=True)
df_covid_cases_by_area["date"] = pd.to_datetime(df_covid_cases_by_area["date"]).dt.date

print(df_covid_cases_by_area)
print("Αρχικό μέγεθος dataset")
print(df_covid_cases_by_area.shape)
# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_covid_cases_by_area.dtypes)

# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns.Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_covid_cases_by_area = df_covid_cases_by_area.drop_duplicates()
print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
print(df_covid_cases_by_area.shape)

print(df_covid_cases_by_area)
# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_covid_cases_by_area.isna().sum())
print("--------------------------------------------")
print(df_covid_cases_by_area.isnull().sum())

# Ανιχνεύει missing values
nan_values = df_covid_cases_by_area.isna()

# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()
# Αριθμός στηλών που έχουμε missing values
print("Αριθμός στηλών που έχουμε missing values")
print(nan_columns.sum())

# Εμφανίζει τις πρώτες 200 γραμμές των κρουσμάτων στην Αττική
print(df_covid_cases_by_area[df_covid_cases_by_area["area_gr"] == "ΑΤΤΙΚΗΣ"].iloc[:200])

# Επειδή δεν είχε εμφανιστεί ο covid σε αυτές τις περιοχές τα κρούσματα είναι 0
df_covid_cases_by_area["cases"].fillna(0, inplace=True)
print("------------------------------------")
print(df_covid_cases_by_area.isna().sum())

df_covid_cases_by_area = df_covid_cases_by_area.loc[df_covid_cases_by_area["date"] >= datetime.strptime('2020-03-20', "%Y-%m-%d").date()]
df_covid_cases_by_area['date'] = pd.to_datetime(df_covid_cases_by_area['date'], format='%Y-%m-%d')

# Αφαιρούμε τις εγγραφές 2021-06-16 και 2021-06-17 καθώς εμφανίζουν πρόβλημα με τα κριτήρια υπολογισμού των κρουσμάτων
df_covid_cases_by_area = df_covid_cases_by_area.loc[(df_covid_cases_by_area['date'] <= '2021-06-15')
                     | (df_covid_cases_by_area['date'] >= '2021-06-18')]
df_covid_cases_by_area["date"] = pd.to_datetime(df_covid_cases_by_area["date"]).dt.date
print(df_covid_cases_by_area)
df_covid_cases_by_area = df_covid_cases_by_area.astype({"cases": "int64"})
'''
Εύρεση κατάλληλης κατανομής
# find_best_distribution_distfit(df_covid_cases_by_area["cases"].values, "Cases")
#
# find_best_distribution_fitter(df_covid_cases_by_area["cases"].values, "Cases")
'''

insert_data_to_table(df_covid_cases_by_area, "covid_by_area_history_cases", "(area_gr,date)")