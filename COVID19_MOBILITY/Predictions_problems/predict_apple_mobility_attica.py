import pandas as pd
from sklearn.preprocessing import StandardScaler

from Plots.plots import make_two_subplots, make_plot, make_bar_plot
from utils import timeseries_conv
from Plots.plots_predictions_from_model import plot_predictions_apple_with_real_data, \
    plot_predictions_apple_valid_with_real_data
from Predictions_problems.models import RNN_neural_network, Support_vector_regressor, \
    LSTM_model, GRU_model
from Predictions_problems.model_utils import calculate_predictions_valid, seperate_dataset_valid, \
    convert_X_to_3D_valid, seperate_dataset, calculate_predictions, get_dates
from utils.dataOperations import load_data_from_table

# Κατασκευάζουμε το dataset το οποίο αποτελείται τους δείκτες κινητικότητας Apple και τους εμβολιασμούς στην Αττική
def construct_dataset():
    df_apple = load_data_from_table("apple_mobility_trends")
    print(df_apple.head())
    # Εκτυπώνουμε τις στήλες
    print(df_apple.columns)
    cols = ["date", "driving", "walking"]
    # Διαλέγουμε μόνο αυτά που είναι για την Αθήνα
    df_apple_athens = df_apple[df_apple["region"] == "Athens"]
    df_apple_athens = df_apple_athens[cols]
    print(df_apple_athens)

    df_vaccinations = load_data_from_table("vaccinations_data_history_per_area")
    cols = ["area_gr", "dailyvaccinations", "date"]
    df_vaccinations = df_vaccinations[cols]
    df_vaccinations_Attica = df_vaccinations[df_vaccinations["area_gr"].isin(["ΑΤΤΙΚΗΣ"])]

    df_athens = pd.merge(df_apple_athens, df_vaccinations_Attica, on='date')
    df_athens = df_athens[["date", "driving", "walking", "dailyvaccinations"]]
    print(df_athens)

    return df_athens


df_athens = construct_dataset()

make_two_subplots(df_athens, "driving", "walking", "Κινητικότητα Apple στην Αττική κατά την διάρκεια της πανδημίας",
                  'Οδήγηση', "Περπάτημα")
make_bar_plot(df_athens, "dailyvaccinations", "Ημερήσιοι συνολικοί εμβολιασμοί στην Αττική", "ημερομηνία", "αριθμός εμβολιασμών")

# Βήμα χρονικών στιγμών που χρησιμοποιούνται για την πρόβλεψη
steps_in = 4
# steps_in = 90
# steps_out = 2
# Βήμα πρόβλεψης του μέλλοντος
steps_out = 1


dates = get_dates(df_athens, steps_in, steps_out)
dates.reset_index(drop=True, inplace=True)
print(dates)

df_athens = df_athens.drop(['date'], axis=1)
features_in = len(df_athens.columns)
prediction_columns = ['walking', 'driving']
features_out = len(prediction_columns)

print("Features in")
print(features_in)
print("Features out")
print(features_out)


# Μετατροπή της χρονοσειράς σε πρόβλημα supervised
dataframe_supervised = timeseries_conv.convert_timeseries_to_supervised(df_athens, steps_in, steps_out, prediction_columns)

# Δημιουργούμε τα χαρακτηριστικά X και τις ετικέτες labels
X, y = dataframe_supervised[[('%s(t-%d)' % (x, i)) for i in range(steps_in, 0, -1) for x in df_athens.columns]].values,\
       dataframe_supervised[[('%s(t)' % (x)) if i == 0 else ('%s(t+%d)' % (x, i)) for i in range(0, steps_out, 1)
                             for x in prediction_columns]].values

print("Μετά το χώρισμα")
print("X")
print(X[:5])
print("Y")
print(y[:5])

print(X.shape)
print(y.shape)

# Μετασχηματίζει τα χαρακτηριστικά κανονικοποιώντας με τον Standard Scaler
scaler = StandardScaler()
y_scaler = StandardScaler()
batch_size = 2
# Κάνει fit τα δεδομένα υπολογίζοντας το μέσο και την διακύμανση που θα χρησιμοποιηθεί στο scaling και
# έπειτα γίνεται ο μετασχηματισμός των δεδομένων.

X = scaler.fit_transform(X)
y = y_scaler.fit_transform(y)
len_data = X.shape[0]

'''
Τα training datasets των X και y θα περιλαμβάνουν το 70% των δεδομένων
Τα validation datasets των X και y θα περιλαμβάνουν το 10% των δεδομένων
Τα test datasets των X και y θα περιλαμβάνουν το 20% των δεδομένων
'''
train_size = int(len_data * .7)
valid_size = int(len_data * .1)
test_size = len_data - (train_size + valid_size)

print("Train size: %d" % train_size)
print("Validation size: %d" % valid_size)
print("Test size: %d" % (test_size))


# Χωρίζουμε το dataset σε training  validation και test sets
X_train, y_train, X_valid, y_valid, X_test, y_test = seperate_dataset_valid(X, y, train_size, valid_size)
output_neurons = features_out * steps_out
# Μετατρέπουμε τα χαρακτηριστικά και τις ετικέτες σε 3D μορφή [samples, steps, features]
X_train, X_valid, X_test = convert_X_to_3D_valid(X_train, X_valid, X_test, steps_in, features_in)

print("----------------------GRU----------------------------")
model = GRU_model(X_train, y_train, X_valid, y_valid, batch_size, 150, output_neurons)

original_data_GRU, predicted_data_GRU = calculate_predictions_valid(model, X_train, X_valid, X_test, y_train,
                                                                    y_valid, y_test, y_scaler, prediction_columns, steps_out)
print(original_data_GRU.shape)
print(predicted_data_GRU.shape)
print(dates.shape)

# Ημερομηνία που ξεκινάει το validation
date_validation = dates[df_athens.index[train_size]]
# Ημερομηνία που ξεκινάει το test
date_test = dates[df_athens.index[train_size + valid_size]]
print("Validation ", date_validation.to_pydatetime().date())
print("Test", date_test.to_pydatetime().date())

plot_predictions_apple_valid_with_real_data(dates, original_data_GRU, predicted_data_GRU, df_athens, train_size, valid_size,
                                            prediction_columns, "Πρόβλεψη δεικτών της Οδήγησης και Περπατήματος με GRU και με βάση"
                                                " τους εμβολιασμούς και τους αντίστοιχους δείκτες στην Αττική")


