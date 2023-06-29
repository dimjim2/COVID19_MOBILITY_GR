from datetime import datetime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Plots.plots import make_google_plot, make_plot, make_bar_plot
from utils import timeseries_conv
from Predictions_problems.models import GRU_model, Random_Forests, ConvLSTM, XGBoost_model
from Predictions_problems.model_utils import seperate_dataset_valid, calculate_predictions, \
    calculate_predictions_valid, convert_X_to_3D_valid, seperate_dataset, get_dates
from utils.dataOperations import load_data_from_table
from Plots.plots_predictions_from_model import plot_google_predictions_valid_with_real_data, plot_google_predictions_with_real_data

# Κατασκευάζουμε το dataset το οποίο αποτελείται τους δείκτες κινητικότητας Google και τα κρούσματα στη Δυτική Ελλάδα
def construct_dataset():
    df_google = load_data_from_table("google_region_mobility")
    df_google.drop(['country_region'], axis=1, inplace=True)
    prediction_columns_google = df_google.drop(['sub_region', 'date'], axis=1).columns.values.tolist()
    print(prediction_columns_google)

    # Επιλέγουμε τις εγγραφές που αφορούν τη Δυτική Ελλάδα
    df_google_Western_Greece = df_google[
        df_google["sub_region"] == "Decentralized Administration of Peloponnese, Western Greece and the Ionian"]
    df_google_Western_Greece.reset_index(drop=True, inplace=True)
    print(df_google_Western_Greece)

    df_cases = load_data_from_table("covid_by_area_history_cases")
    Western_regions = ["ΑΙΤΩΛΟΑΚΑΡΝΑΝΙΑΣ", "ΑΡΓΟΛΙΔΑΣ", "ΑΡΚΑΔΙΑΣ", "ΑΧΑΪΑΣ", "ΗΛΕΙΑΣ", "ΚΕΡΚΥΡΑΣ", "ΚΕΦΑΛΛΟΝΙΑΣ",
                       "ΚΟΡΙΝΘΟΥ", "ΛΕΥΚΑΔΑΣ", "ΜΕΣΣΗΝΙΑΣ", "ΖΑΚΥΝΘΟΥ"]
    # Επιλέγουμε μια εγγραφή εφόσον βρίσκεται σε μια από τις παραπάνω περιοχές
    df_cases = df_cases[df_cases["area_gr"].isin(Western_regions)]
    df_cases = df_cases.replace(Western_regions, 'Δυτική Ελλάδα')
    print(df_cases)

    # Αθροίζουμε τις εγγραφές ανά ημερομηνία
    df_cases_Western = df_cases.groupby(by=['area_gr', 'date']).sum()
    df_cases_Western.sort_values(by=['date', 'area_gr'], inplace=True)
    df_cases_Western['cases'].iloc[1:] = df_cases_Western['cases'].diff().iloc[1:]
    df_cases_Western.reset_index(inplace=True)
    print("Κρούσματα Δυτικής Ελλάδας")
    print(df_cases_Western)

    # Συνενώνουμε τις εγγραφές των δύο dataframes σε ένα
    df_Western_Greece = pd.merge(df_cases_Western, df_google_Western_Greece, on='date')
    df_Western_Greece = df_Western_Greece.query("date <= '2022-07-10'")
    df_Western_Greece.drop(['area_gr', 'sub_region'], axis=1, inplace=True)

    return df_Western_Greece, prediction_columns_google


df_Western_Greece, prediction_columns_google = construct_dataset()

print("Dataset Western Greece")
print(df_Western_Greece)
print(df_Western_Greece.head(5))
print(df_Western_Greece.columns)

make_google_plot(df_Western_Greece, "Ημερήσια κινητικότητα Google στην Δυτική Ελλάδα")
make_plot(df_Western_Greece, "cases", "Ημερήσια κρούσματα στην Δυτική Ελλάδα", "ημερομηνία", "αριθμός κρουσμάτων")

# Μέγεθος παραθύρου και βήμα πρόβλεψης του μέλλοντος
steps_in = 7
steps_out = 1

dates = get_dates(df_Western_Greece, steps_in, steps_out)
dates.reset_index(drop=True, inplace=True)
print(dates)
df_Western_Greece = df_Western_Greece.drop(['date'], axis=1)

dataframe = timeseries_conv.convert_timeseries_to_supervised(df_Western_Greece, steps_in, steps_out, prediction_columns_google)
print(dataframe)
print(dataframe.columns)
# Χωρίζουμε το dataset σε χαρακτηριστικά εισόδου και labels
X, y = dataframe[[('%s(t-%d)' % (x, i)) for i in range(steps_in, 0, -1) for x in df_Western_Greece.columns]].values,\
       dataframe[[('%s(t)' % (x)) if i == 0 else ('%s(t+%d)' % (x, i)) for i in range(0, steps_out, 1)
                             for x in prediction_columns_google]].values

features_in = len(df_Western_Greece.columns)
features_out = len(prediction_columns_google)

print("Features in")
print(features_in)
print("Features out")
print(features_out)

print("X")
print(X)
print(X.shape)
print("y")
print(y)
print(y.shape)

# Μετασχηματισμός z-scaling
scaler = StandardScaler()
y_scaler = StandardScaler()

X = scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

print(X.shape)
print(y.shape)

size_of_dataset = X.shape[0]
train_size = int(size_of_dataset * .6)

X_train, y_train, X_test,  y_test = seperate_dataset(X, y, train_size)
print(X_train.shape)
print(y_train.shape)

print("----------------XGBoost--------------")
# Δημιουργία XGBoost μοντέλου
XGBoost_model = XGBoost_model(X_train, y_train)
original_data_XGBoost, predicted_data_XGBoost = calculate_predictions(XGBoost_model, X_train, X_test, y_train,
y_test, y_scaler, prediction_columns_google, steps_out)
date_test = dates[dataframe.index[train_size]]
date_test = date_test.to_pydatetime().date()
print("Ημερομηνία που ξεκινάει ο έλεγχος", date_test)

plot_google_predictions_with_real_data(dates, original_data_XGBoost, predicted_data_XGBoost, df_Western_Greece, len(X_train),
                 "Πρόβλεψη δείκτων κινητικότητας της Google στην Πελοπόννησο με XGBoost και με βάση "
                "τα κρούσματα και τους αντίστοιχους δείκτες")

