import pandas as pd

from utils.dataOperations import insert_data_to_table
from Plots.plots import *
'''
Εισάγει τα δεδομένα τα δεδομένα της Apple στον πίνακα apple_mobility_trends της βάσης ενώ ταυτόχρονα παράγει χρήσιμα γραφήματα
Οι στήλες που εισάγονται αφορούν τον γεωγραφικό τύπο (πχ πόλη ή χώρα), την περιοχή, την ημερομηνία,
την οδήγηση και το περπάτημα.
'''
def insert_data_to_database_from_apple():
    df_apple_greece = get_apple_dataset()
    print(df_apple_greece)

    df_apple_greece.drop(['alternative_name', 'sub-region', 'country'], axis=1, inplace=True)

    print("Αρχικό μέγεθος dataset")
    print(df_apple_greece.shape)
    # Επιστρέφει τους τύπους των στηλών του Dataframe
    print(df_apple_greece.dtypes)
    # Παράγει στατιστικά
    print(df_apple_greece.describe())

    # Αφαιρούμε όλες τις διπλότυπες γραμμές βάσει όλων των columns. Η πρώτη γραμμή by default είναι αυτή που διατηρείται
    df_apple_greece = df_apple_greece.drop_duplicates()
    print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
    print(df_apple_greece.shape)
    # Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
    print(df_apple_greece.isna().sum())
    print("--------------------------------------------")
    print(df_apple_greece.isnull().sum())

    nan_values = df_apple_greece.isna()
    # Βρίσκει τις στήλες που έχουμε missing values
    nan_columns = nan_values.any()
    # Αριθμός στηλών που έχουμε missing values
    print("Αριθμός στηλών που έχουμε missing values")
    print(nan_columns.sum())

    # Βρίσκουμε τα ονόματα των στηλών που έχουμε missing values
    columns_with_nan = df_apple_greece.columns[nan_columns].tolist()
    print("Στήλες με nan values")
    print(columns_with_nan)

    # Πετάμε το παλιό index του Dataframe και το νέο Index θα είναι auto increment
    df_apple_greece.reset_index(drop=True, inplace=True)

    '''
    Μετατρέπουμε τις στήλες που αφορούν ημερομηνίες και τις τιμές τους σε γραμμές 
    id_vars-> Στήλες που παραμένουν αμετάβλητες
    var_name-> Όνομα για την νέα στήλη. Οι τιμές της εδώ είναι οι ημερομηνίες
    value_name-> Οι τιμές των ημερομηνιών που αντιστοιχούν πλέον στην νέα στήλη
    '''

    df_apple_greece = df_apple_greece.melt(
        id_vars=["geo_type", "region", "transportation_type"],
        var_name="Date",
        value_name="Value")

    # Μετατρέπουμε το transportation_type σε στήλες driving and walking και θέτουμε τις τιμές τους
    df_apple_greece = df_apple_greece.set_index(
        ["geo_type", "region", "Date", "transportation_type"])[
        'Value'].unstack()

    df_apple_greece = df_apple_greece.reset_index().rename_axis(None).rename_axis(None, axis=1)
    print(df_apple_greece)
    print(df_apple_greece.columns)

    cols = ["driving", "walking"]

    df_apple_greece.rename(columns={'Date': 'date'}, inplace=True)


    df_apple_greece["date"] = pd.to_datetime(df_apple_greece["date"]).dt.date

    # Κάνουμε sort με βάση την ημερομηνία
    df_apple_greece.sort_values("date", inplace=True)
    df_apple_greece.reset_index(drop=True, inplace=True)

    print(df_apple_greece.dtypes)

    make_hist(df_apple_greece, cols, 50, "Ιστόγραμμα Apple mobility dataset πριν  από καθαρισμό")

    print(df_apple_greece.isna().sum())
    print("--------------------------------------------")
    print(df_apple_greece.isnull().sum())
    df_apple_greece["date"] = pd.to_datetime(df_apple_greece["date"]).dt.date

    # Συμπλήρωση ελλειπών τιμών με linear interpolation
    df_apple_greece.interpolate(method='linear', inplace=True)

    '''
    # Συμπλήρωση τιμών που λείπουν με μέση τιμή
    # df_apple_greece["driving"].fillna((df_apple_greece.groupby("region")["driving"].transform('mean')), inplace=True)
    # df_apple_greece["walking"].fillna((df_apple_greece.groupby("region")["walking"].transform('mean')), inplace=True)
    # Συμπλήρωση τιμών που λείπουν με πολυωνυμικό interpolation
    #df_apple_greece.interpolate(method='polynomial', order=2, inplace=True)
    
    #Συμπλήρωση τιμών που λείπουν με KNN imputation
    # imputer = KNNImputer(n_neighbors=7, weights='uniform', metric='nan_euclidean')
    # df_apple_greece[cols] = imputer.fit_transform(df_apple_greece[cols])
    
    Συμπλήρωση με Iterative Imputation
    # it_imputer = IterativeImputer(random_state=0, skip_complete=True)
    # df_apple_greece[cols] = it_imputer.fit_transform(df_apple_greece[cols])
    # print(it_imputer.n_features_with_missing_)
    # df_apple_greece[cols] = df_apple_greece[cols].fillna(df_apple_greece[cols].rolling(window=30, min_periods=1).mean())
    '''

    print(df_apple_greece.isna().sum())
    print("--------------------------------------------")
    print(df_apple_greece.isnull().sum())

    print(df_apple_greece.dtypes)

    '''
    Βρίσκουμε την πιο πιθανή κατανομή των δεδομένων μας
    # find_best_distribution_distfit(df_apple_greece["driving"].values, "Driving")
    #
    # find_best_distribution_fitter(df_apple_greece["driving"].values, "Driving")
    #
    # find_best_distribution_distfit(df_apple_greece["walking"].values, "Walking")
    #
    # find_best_distribution_fitter(df_apple_greece["walking"].values, "Walking")
    '''

    insert_data_to_table(df_apple_greece, "apple_mobility_trends", "(region,date)")

# Λαμβάνει τα δεδομένα από το Apple dataset που αφορούν μόνο την Ελλάδα
def get_apple_dataset():
    df_apple = pd.read_csv('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/apple_reports/applemobilitytrends.csv')
    print(df_apple)
    # Εκτυπώνουμε τις στήλες
    print(df_apple.columns)
    # Επιλέγουμε τις εγγραφές του Apple mobility Dataset που αφορούν μόνο την Ελλάδα
    df_region_greece_1 = df_apple.loc[df_apple["region"] == "Greece"]
    df_region_greece_2 = df_apple.loc[df_apple["country"] == "Greece"]
    # Συνενώνουμε τα δύο dataframes objects που αφορούν την Ελλάδα
    df_apple_greece = pd.concat([df_region_greece_1, df_region_greece_2])
    return df_apple_greece


insert_data_to_database_from_apple()
