import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.dataOperations import insert_data_to_table

'''
Εισάγει τους συνολικούς εμβολιασμούς καθώς και αυτούς με πρώτη, δεύτερη και αναμνηστική δόση  ανά περιοχή της Ελλάδας
στον πίνακα cumulative_per_area_vaccinations
'''

url = 'https://raw.githubusercontent.com/Covid-19-Response-Greece/covid19-data-greece/master/data/greece/vaccines/cumulative_per_area_vaccinations.json'
df_cumulative_area_vaccinations = pd.read_json(url)

print(df_cumulative_area_vaccinations)
print(df_cumulative_area_vaccinations.columns)

df_cumulative_area_vaccinations.rename(columns={"referencedate": "date"}, inplace=True)
df_cumulative_area_vaccinations["date"] = pd.to_datetime(df_cumulative_area_vaccinations["date"]).dt.date
# Μετατρέπουμε τη στήλη της περιοχής σε μορφή νομού
df_cumulative_area_vaccinations['area_gr'].replace({'ΜΥΚΟΝΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΜΗΛΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΠΑΡΟΥ': 'ΚΥΚΛΑΔΩΝ',
                                                    'ΝΑΞΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΤΗΝΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΣΥΡΟΥ': 'ΚΥΚΛΑΔΩΝ',
                                                    'ΑΝΔΡΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΚΕΑΣ-ΚΥΘΝΟΥ': 'ΚΥΚΛΑΔΩΝ', 'ΘΗΡΑΣ': 'ΚΥΚΛΑΔΩΝ',
                                                    'ΛΗΜΝΟΥ': 'ΛΕΣΒΟΥ', 'ΚΩ': 'ΔΩΔΕΚΑΝΗΣΩΝ', 'ΡΟΔΟΥ': 'ΔΩΔΕΚΑΝΗΣΩΝ',
                                                    'ΚΑΡΠΑΘΟΥ': 'ΔΩΔΕΚΑΝΗΣΩΝ', 'ΚΑΛΥΜΝΟΥ': 'ΔΩΔΕΚΑΝΗΣΩΝ', 'ΚΟΡΙΝΘΙΑΣ': 'ΚΟΡΙΝΘΟΥ',
                                                    'ΣΠΟΡΑΔΩΝ': 'ΜΑΓΝΗΣΙΑΣ', 'ΚΕΦΑΛΛΗΝΙΑΣ': 'ΚΕΦΑΛΛΟΝΙΑΣ', 'ΙΘΑΚΗΣ': 'ΚΕΦΑΛΛΟΝΙΑΣ',
                                                    'ΙΚΑΡΙΑΣ': 'ΣΑΜΟΥ', 'ΘΑΣΟΣ': 'ΚΑΒΑΛΑΣ', 'ΚΕΝΤΡΙΚΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ',
                                                    'ΝΟΤΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ', 'ΝΗΣΩΝ': 'ΑΤΤΙΚΗΣ', 'ΠΕΙΡΑΙΩΣ': 'ΑΤΤΙΚΗΣ',
                                           'ΔΥΤΙΚΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ':' ΑΤΤΙΚΗΣ', 'ΔΥΤΙΚΗΣ ΑΤΤΙΚΗΣ': 'ΑΤΤΙΚΗΣ',
                                                    'ΒΟΡΕΙΟΥ ΤΟΜΕΑ ΑΘΗΝΩΝ': 'ΑΤΤΙΚΗΣ', 'ΑΝΑΤΟΛΙΚΗΣ ΑΤΤΙΚΗΣ':'ΑΤΤΙΚΗΣ'},
                                                   inplace=True)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # Εγγραφές με την ίδια τιμή στην περιοχή αθροίζονται και προκύπτει μια ενιαία εγγραφή
    df_cumulative_area_vaccinations = df_cumulative_area_vaccinations.groupby(by=['area_gr', 'date']).sum()
    df_cumulative_area_vaccinations.sort_values(by=['date', 'area_gr'], inplace=True)
    df_cumulative_area_vaccinations.reset_index(inplace=True)

print(df_cumulative_area_vaccinations)
print("Αρχικό μέγεθος dataset")
print(df_cumulative_area_vaccinations.shape)
# Επιστρέφει τους τύπους των στηλών του Dataframe
print(df_cumulative_area_vaccinations.dtypes)

# Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns.Η πρώτη γραμμή by default είναι αυτή που διατηρείται
df_cumulative_area_vaccinations = df_cumulative_area_vaccinations.drop_duplicates()
print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
print(df_cumulative_area_vaccinations.shape)

# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_cumulative_area_vaccinations.isna().sum())
print("--------------------------------------------")
print(df_cumulative_area_vaccinations.isnull().sum())

# Ανιχνεύευει missing values
nan_values = df_cumulative_area_vaccinations.isna()
# Βρίσκει τις στήλες που έχουμε missing values
nan_columns = nan_values.any()
# Αριθμός στηλών που έχουμε missing values
print("Αριθμός στηλών που έχουμε missing values")
print(nan_columns.sum())

sns.displot(df_cumulative_area_vaccinations["totalvaccinations"], bins=30)
plt.show()

insert_data_to_table(df_cumulative_area_vaccinations, "cumulative_per_area_vaccinations", "(area_gr)")