import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataOperations import insert_data_to_table

'''
Εισάγει τους συνολικούς εμβολιασμούς καθώς και αυτούς με πρώτη, δεύτερη και αναμνηστική δόση ανά περιοχή της Ελλάδας
και ανά ημερομηνία στον πίνακα vaccinations_data_history_per_area της βάσης
'''

df_vaccinations_area = pd.read_json('https://raw.githubusercontent.com/Covid-19-Response-Greece/covid19-data-greece'
                                      '/master/data/greece/vaccines/vaccinations_data_history.json')
print(df_vaccinations_area)

df_vaccinations_area.drop(labels=["area_en"], axis=1, inplace=True)
df_vaccinations_area.rename(columns={"referencedate": "date"}, inplace=True)
print(df_vaccinations_area.columns)
# Μετατρέπουμε τη στήλη της περιοχής σε μορφή νομού
df_vaccinations_area['area_gr'].replace({'ΜΥΚΟΝΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΜΗΛΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΠΑΡΟΥ': 'ΚΥΚΛΑΔΩΝ',
                                           'ΝΑΞΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΤΗΝΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΣΥΡΟΥ': 'ΚΥΚΛΑΔΩΝ',
                                           'ΑΝΔΡΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΚΕΑΣ-ΚΥΘΝΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΘΗΡΑΣ': 'ΚΥΚΛΑΔΩΝ',
                                           'ΛΗΜΝΟΥ': 'ΛΕΣΒΟΥ', 'ΚΩ': 'ΔΩΔΕΚΑΝΗΣΩΝ', 'ΡΟΔΟΥ': 'ΔΩΔΕΚΑΝΗΣΩΝ',
                                           'ΚΑΡΠΑΘΟΥ': 'ΔΩΔΕΚΑΝΗΣΩΝ', 'ΚΑΛΥΜΝΟΥ': 'ΔΩΔΕΚΑΝΗΣΩΝ', 'ΚΟΡΙΝΘΙΑΣ': 'ΚΟΡΙΝΘΟΥ',
                                           'ΣΠΟΡΑΔΩΝ': 'ΜΑΓΝΗΣΙΑΣ', 'ΚΕΦΑΛΛΗΝΙΑΣ': 'ΚΕΦΑΛΛΟΝΙΑΣ', 'ΙΘΑΚΗΣ': 'ΚΕΦΑΛΛΟΝΙΑΣ',
                                           'ΙΚΑΡΙΑΣ': 'ΣΑΜΟΥ', 'ΘΑΣΟΣ': 'ΚΑΒΑΛΑΣ', 'ΚΕΝΤΡΙΚΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ',
                                           'ΝΟΤΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ', 'ΝΗΣΩΝ': 'ΑΤΤΙΚΗΣ', 'ΠΕΙΡΑΙΩΣ': 'ΑΤΤΙΚΗΣ',
                                           'ΔΥΤΙΚΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ', 'ΔΥΤΙΚΗΣ ΑΤΤΙΚΗΣ': 'ΑΤΤΙΚΗΣ',
                                           'ΒΟΡΕΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ', 'ΑΝΑΤΟΛΙΚΗΣ ΑΤΤΙΚΗΣ': 'ΑΤΤΙΚΗΣ'}, inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # Εγγραφές με την ίδια τιμή στην περιοχή αθροίζονται και προκύπτει μια ενιαία εγγραφή
    df_vaccinations_area = df_vaccinations_area.groupby(by=['area_gr', 'date']).sum()
    df_vaccinations_area.sort_values(by=['date', 'area_gr'], inplace=True)
    df_vaccinations_area.reset_index(inplace=True)

print(df_vaccinations_area)
df_vaccinations_area["date"] = pd.to_datetime(df_vaccinations_area["date"]).dt.date

print("Αρχικό μέγεθος dataset")
print(df_vaccinations_area.shape)
# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_vaccinations_area.dtypes)

df_vaccinations_area.rename(columns={"daytotal": "dailyvaccinations"}, inplace=True)

# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns. Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_vaccinations_area = df_vaccinations_area.drop_duplicates()
print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
print(df_vaccinations_area.shape)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_vaccinations_area.isna().sum())
print("--------------------------------------------")
print(df_vaccinations_area.isnull().sum())

sns.displot(df_vaccinations_area["dailyvaccinations"])
plt.title("Daily total vaccinations", fontsize=7)
plt.show()

# find_best_distribution_distfit(df_vaccinations_area["dailyvaccinations"].values, "Daily total accinations")
#
# find_best_distribution_fitter(df_vaccinations_area["dailyvaccinations"].values, "Daily Total vaccinations")

insert_data_to_table(df_vaccinations_area, "vaccinations_data_history_per_area", "(area_gr,date)")