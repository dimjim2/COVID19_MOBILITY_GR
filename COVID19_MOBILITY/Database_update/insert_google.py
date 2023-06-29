import urllib.request
from zipfile import ZipFile
import pandas as pd

from Plots.plots import make_boxplot, make_hist, make_google_plot, make_google_plot_in_the_same_axes, \
    make_google_scatter_plot
from utils.dataOperations import insert_data_to_table
'''
Εισάγει τους δείκτες κινητικότητας της Google στον πίνακα google_region_mobility της βάσης
'''

def insert_data_to_database_from_google():
    df_google = get_google_dataset()
    print(df_google.head())
    df_google["date"] = pd.to_datetime(df_google["date"])
    print(df_google.dtypes)

    cols = ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']

    make_hist(df_google, cols, 10, "Ιστόγραμμα για το αρχικό dataset πριν εφαρμοστούν τεχνικές preprocessing")
    # Δημιουργία plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία
    make_google_plot(df_google, "Plot πριν από τον καθαρισμό")
    # Δημιουργία scatter plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία
    make_google_scatter_plot(df_google, "Scatter plot πριν από τον καθαρισμό")
    make_boxplot(df_google, cols, "Boxplot στο dataset google mobility  before data cleaning")

    df_weekly_google = df_google.set_index("date").resample('W').apply('mean')
    df_weekly_google.reset_index(inplace=True)
    make_google_plot_in_the_same_axes(df_weekly_google, "Εβδομαδιαία μέση κινητικότητα Google πριν το preprocessing")

    print("Αρχικό μέγεθος dataset")
    print(df_google.shape)
    df_google.rename(columns={"sub_region_1": "sub_region"}, inplace=True)
    df_google = df_google.drop_duplicates()
    print("Μέγεθος dataset με την αφαίρεση διπλότυπων γραμμών")
    print(df_google.shape)

    # Τα outliers υπάρχουν εξ αρχής στο dataset και δεν οφείλεται σε σφάλμα του δικού μας κώδικα
    # Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values

    print("Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values")
    print(df_google.isna().sum())
    print("--------------------------------------------")
    print(df_google.isnull().sum())

    # Βάσει κατάλληλης αναζήτησης με τον κωδικό place id και το site https://developers.google.cn/maps/documentation/javascript/examples/geocoding-place-id
    df_google["sub_region"].fillna("Greece", inplace=True)
    print("--------------------------------------------")
    print(df_google.isna().sum())

    df_google.drop(["country_region_code", "sub_region_2", "place_id", "metro_area", "iso_3166_2_code", "census_fips_code"], axis=1, inplace=True)
    print("Μέγεθος μετά την αφαίρεση")
    print(df_google.shape)
    print("--------------------------------------------")

    print(df_google.isna().sum())
    print(df_google.isnull().sum())

    df_google["date"] = pd.to_datetime(df_google["date"]).dt.date
    df_google = df_google.astype({"retail_and_recreation_percent_change_from_baseline": "float64",
                                  "grocery_and_pharmacy_percent_change_from_baseline": "float64",
                                  "parks_percent_change_from_baseline": "float64",
                                  "transit_stations_percent_change_from_baseline": "float64"})
    print(df_google.dtypes)
    # Χρήση linear interpolation για missing values
    df_google.interpolate(method='linear', inplace=True)

    '''
    Συμπλήρωση ελλειπών τιμών με Q2
    # df_google['retail_and_recreation_percent_change_from_baseline'].fillna((df_google.groupby("sub_region")['retail_and_recreation_percent_change_from_baseline'].transform('median')), inplace = True)
    # df_google['grocery_and_pharmacy_percent_change_from_baseline'].fillna((df_google.groupby("sub_region")['grocery_and_pharmacy_percent_change_from_baseline'].transform('median')), inplace = True)
    # df_google["parks_percent_change_from_baseline"].fillna((df_google.groupby("sub_region")["parks_percent_change_from_baseline"].transform('median')), inplace = True)
    # df_google["transit_stations_percent_change_from_baseline"].fillna((df_google.groupby("sub_region")["transit_stations_percent_change_from_baseline"].transform('median')), inplace = True)
    Συμπλήρωση ελλειπών τιμών με τεχνική rolling window
    # df_google[cols] = df_google[cols].fillna(df_google[cols].rolling(window=30, min_periods=1).mean())
    Συμπλήρωση ελλειπών τιμών 
    # imputer = KNNImputer(n_neighbors=7, weights='uniform', metric='nan_euclidean')
    # df_google[cols] = imputer.fit_transform(df_google[cols])

    # it_imputer = IterativeImputer(random_state=0)
    # df_google[cols] = it_imputer.fit_transform(df_google[cols])
    '''

    print("--------------------------------------------")
    print(df_google.isna().sum())
    print(df_google.dtypes)

    insert_data_to_table(df_google, "google_region_mobility", "(sub_region,date)")

# Λαμβάνει τα δεδομένα από το Google dataset που αφορούν μόνο την Ελλάδα
def get_google_dataset():
    url = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
    # Αντιγράφει το network object στην τοποθεσία ../Region_Mobility_Report_CSVs.zip
    urllib.request.urlretrieve(url, '../Region_Mobility_Report_CSVs.zip')
    with ZipFile('../Region_Mobility_Report_CSVs.zip', 'r') as zipObj:

        # Λαμβάνει μια λίστα όλων των archived file names
        listOfFileNames = zipObj.namelist()
        for fileName in listOfFileNames:
            if fileName.endswith('.csv'):
                if (fileName == "2022_GR_Region_Mobility_Report.csv" or fileName == "2021_GR_Region_Mobility_Report.csv"
                        or fileName == "2020_GR_Region_Mobility_Report.csv"):
                    # κάνει extract το αρχείο στο current working directory
                    zipObj.extract(fileName)

    df_google_2020 = pd.read_csv('2020_GR_Region_Mobility_Report.csv')
    df_google_2021 = pd.read_csv('2021_GR_Region_Mobility_Report.csv')
    df_google_2022 = pd.read_csv('2022_GR_Region_Mobility_Report.csv')
    # Συνένωση σε ένα pandas dataframe
    df_google = pd.concat([df_google_2020, df_google_2021, df_google_2022])
    return df_google


insert_data_to_database_from_google()