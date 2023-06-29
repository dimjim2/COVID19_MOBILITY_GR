import pandas as pd
from sklearn.preprocessing import StandardScaler

from Plots.plots_predictions_from_model import plot_predictions_apple_valid_with_real_data
from utils import timeseries_conv
from Predictions_problems.models import ConvLSTM, RNN_neural_network
from Predictions_problems.model_utils import seperate_dataset_valid, convert_X_to_3D_valid, \
    calculate_predictions_valid, get_dates
from utils.dataOperations import load_data_from_table

'''
Κατασκευάζουμε το dataset το οποίο αποτελείται τους πανελλήνιους δείκτες κινητικότητας Apple, τους εμβολιασμούς,
τους θανάτους, τον αριθμό των κρουσμάτων και των ασθενών σε ΜΕΘ στην Ελλάδα
'''
def construct_dataset():
    df_apple = load_data_from_table("apple_mobility_trends")
    print(df_apple.head())
    print(df_apple.columns)
    cols = ["date", "driving", "walking"]
    df_apple_greece = df_apple[df_apple["region"] == "Greece"]
    df_apple_greece = df_apple_greece[cols]
    print(df_apple_greece)

    df_vaccinations = load_data_from_table("vaccinations_data_history_per_area")
    print(df_vaccinations.groupby('date')["dailyvaccinations"].sum())
    df_vaccinations_greece = df_vaccinations[['date']].drop_duplicates().copy()
    df_vaccinations_greece["dailyvaccinations"] = df_vaccinations.groupby('date')["dailyvaccinations"].sum().values
    df_vaccinations_greece.reset_index(drop=True, inplace=True)
    print(df_vaccinations_greece)

    df_deaths = load_data_from_table("deaths")
    df_deaths.drop(['deaths_cum'], axis=1, inplace=True)
    print(df_deaths)

    df_cases_greece = load_data_from_table("covid_cases_greece")
    df_cases_greece.drop(['total_cases'], axis=1, inplace=True)
    print(df_cases_greece)

    df_intensive_care_cases = load_data_from_table("intensive_care_cases")
    print(df_intensive_care_cases)
    df_greece = pd.merge(pd.merge(df_apple_greece, pd.merge(df_vaccinations_greece,
                                                            pd.merge(df_intensive_care_cases, df_deaths, on='date'),
                                                            on='date'), on='date'), df_cases_greece, on='date')
    return df_greece


df_greece = construct_dataset()
print(df_greece.isna().sum())

# Μέγεθος παραθύρου παρελθόντος και βήμα πρόβλεψης του μέλλοντος
steps_in = 4
# steps_in = 90
steps_out = 1

dates = get_dates(df_greece, steps_in, steps_out)
dates.reset_index(drop=True, inplace=True)
print(dates)

df_greece = df_greece.drop(['date'], axis=1)
features_in = len(df_greece.columns)
prediction_col_names = ['deaths', 'intensive_care_patients']
features_out = len(prediction_col_names)

print("Features in")
print(features_in)
print("Features out")
print(features_out)


# Μετατροπή της χρονοσειράς σε πρόβλημα supervised learning
dataframe_supervised = timeseries_conv.convert_timeseries_to_supervised(df_greece, steps_in, steps_out, prediction_col_names)
print(dataframe_supervised)
print()

# Δημιουργούμε τα χαρακτηριστικά X και τις ετικέτες labels
X, y = dataframe_supervised[[('%s(t-%d)' % (x,i)) for i in range(steps_in, 0, -1) for x in df_greece.columns]].values,\
       dataframe_supervised[[('%s(t)' % (x)) if i == 0 else ('%s(t+%d)' % (x, i)) for i in range(0, steps_out, 1)
                             for x in prediction_col_names]].values

print(X.shape)
print(y.shape)

# Μετασχηματίζει τα χαρακτηριστικά κανονικοποιώντας με τον Standard Scaler
scaler = StandardScaler()
y_scaler = StandardScaler()

# Κάνει fit τα δεδομένα υπολογίζοντας το μέσο και την διακύμανση που θα χρησιμοποιηθεί στο scaling και
# έπειτα γίνεται ο μετασχηματισμός των δεδομένων.

X = scaler.fit_transform(X)
y = y_scaler.fit_transform(y)
len_data = X.shape[0]

'''
Τα training datasets των X και y θα περιλαμβάνουν το 50% των δεδομένων
Τα validation datasets των X και y θα περιλαμβάνουν το 30% των δεδομένων
Τα test datasets των X και y θα περιλαμβάνουν το 20% των δεδομένων
'''

train_size = int(len_data * .5)
valid_size = int(len_data * .3)
test_size = len_data - (train_size+valid_size)

print("Train size: %d" % train_size)
print("Validation size: %d" % valid_size)
print("Test size: %d" % (test_size))


X_train, y_train, X_valid, y_valid, X_test, y_test = seperate_dataset_valid(X, y, train_size, valid_size)
output_neurons = features_out * steps_out

batch_size = 1
X_train, X_valid, X_test = convert_X_to_3D_valid(X_train, X_valid, X_test, steps_in, features_in)


#model = ConvLSTM(X_train, y_train, X_valid, y_valid, 150, batch_size, features_out, steps_out)
model = RNN_neural_network(X_train, y_train, X_valid, y_valid, 300, batch_size, output_neurons)
original, predicted = calculate_predictions_valid(model, X_train, X_valid, X_test, y_train,
                                                  y_valid, y_test, y_scaler, prediction_col_names, steps_out)
print(original.shape)
print(predicted.shape)
print(dates.shape)
# Ημερομηνία που ξεκινάει το validation
date_validation = dates[df_greece.index[train_size]]
# Ημερομηνία που ξεκινάει το test
date_test = dates[df_greece.index[train_size + valid_size]]
print("Validation ", date_validation.to_pydatetime().date())
print("Test", date_test.to_pydatetime().date())

plot_predictions_apple_valid_with_real_data(dates, original, predicted, df_greece, train_size,
                                            valid_size, prediction_col_names, "Πρόβλεψη κρουσμάτων και ατόμων με την χρήση RNN")
