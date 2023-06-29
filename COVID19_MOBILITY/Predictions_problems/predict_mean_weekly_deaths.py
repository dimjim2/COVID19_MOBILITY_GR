import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from Plots.plots import make_plot, make_bar_plot, make_google_plot
from utils import timeseries_conv
from Plots.plots_predictions_from_model import plot_predictions_valid_with_real_data, \
    plot_correlation_matrix_prediction_column
from Predictions_problems.models import MLP_neural_netowrk_valid, XGBoost_model, \
    CNN_model, LSTM_model
from Predictions_problems.model_utils import seperate_dataset_valid, calculate_predictions_valid, \
    convert_X_to_3D_valid, get_dates
from utils.dataOperations import load_data_from_table

# Κατασκευάζουμε το dataset το οποίο αποτελείται τους μέσους εβδομαδιαίους δείκτες κινητικότητας Google,
# τους θανάτους και τον αριθμό ασθενών σε ΜΕΘ στην Ελλάδα
def construct_dataset():
    # Θάνατοι στην Ελλάδα
    df_deaths = load_data_from_table("deaths")
    df_deaths.drop(['deaths_cum'], axis=1, inplace=True)
    print(df_deaths)

    # Ασθενείς σε ΜΕΘ
    df_intensive_care_cases = load_data_from_table("intensive_care_cases")
    print(df_intensive_care_cases)

    # Δείκτες κινητικότητας του Google dataset
    df_google = load_data_from_table("google_region_mobility")
    df_google.drop(['country_region'], axis=1, inplace=True)
    df_google_greece = df_google[df_google["sub_region"] == "Greece"]
    df_google_greece.reset_index(drop=True, inplace=True)
    print(df_google_greece)

    # Συνενώνουμε όλα τα dataframes σε ένα
    df_greece = pd.merge(df_google_greece,  pd.merge(df_deaths, df_intensive_care_cases, on='date'), on='date')
    print(df_greece.columns)
    print(df_greece.isna().sum())

    # Υπολογίζουμε για τα χαρακτηριστικά τις μέσες εβδομαδιαίες τιμές τους
    df_greece_weekly = df_greece.set_index("date").resample('W').apply('mean')
    df_greece_weekly.reset_index(inplace=True)
    print(df_greece_weekly)
    return df_greece_weekly


df_greece_weekly = construct_dataset()

make_plot(df_greece_weekly, "deaths", "Εβδομαδιαίοι μέσοι Θάνατοι στην Ελλάδα", "ημερομηνία", "θάνατοι")
make_plot(df_greece_weekly, "intensive_care_patients", "Εβδομαδιαίος μέσος αριθμός ασθενών σε ΜΕΘ", "ημερομηνία", "άτομα σε ΜΕΘ")
make_google_plot(df_greece_weekly, "Εβδομαδιαία μέση κινητικότητα Google στην Ελλάδα")

# Μέγεθος παραθύρου και βήμα πρόβλεψης του μέλλοντος
# steps_in = 180
steps_in = 4
# steps_in = 2
steps_out = 1

dates = get_dates(df_greece_weekly, steps_in, steps_out)
dates.reset_index(drop=True, inplace=True)
print(dates)

df_greece_weekly.drop(['date'], axis=1, inplace=True)
print(df_greece_weekly)

prediction_col = 'deaths'
prediction_column = ['deaths']

# Μήτρα σύγχυσης
df_greece_weekly.columns = df_greece_weekly.columns.str.replace("_percent_change_from_baseline","")
correlation_matrix = df_greece_weekly.corr()[prediction_column]
print("---------------------------------------------")
print("Correlation matrix")
print(correlation_matrix)
plot_correlation_matrix_prediction_column(correlation_matrix, str(prediction_col))

features_in = len(df_greece_weekly.columns)
features_out = len(prediction_column)

print("Features in")
print(features_in)
print("Features out")
print(features_out)


# Μετατροπή της χρονοσειράς σε πρόβλημα supervised
df_weekly_supervised = timeseries_conv.convert_timeseries_to_supervised(df_greece_weekly, steps_in, steps_out, prediction_column)
print("-----------------------------------------------------")

# Εκτυπώνουμε τις στήλες του μετασχηματισμένου dataset.
print("Στήλες μετασχηματισμένου dataset")
print(df_weekly_supervised.columns)
print(df_greece_weekly.columns)



X, y = df_weekly_supervised[[('%s(t-%d)' % (x, i)) for i in range(steps_in, 0, -1) for x in df_greece_weekly.columns]].values, \
       df_weekly_supervised[[('%s(t)' % (x)) if i == 0 else ('%s(t+%d)' % (x, i)) for i in range(0, steps_out, 1) for x in
                             prediction_column]].values

print(X.shape, y.shape)

# Μετασχηματισμός z-scaling
scaler = StandardScaler()
y_scaler = StandardScaler()

print("Statistics")
print(DataFrame(X).describe())

X = scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

print("Scaled Statistics")
print(DataFrame(X).describe())
len_data = X.shape[0]


train_size = int(len_data * .6)
valid_size = int(len_data * .2)
test_size = len_data - (train_size + valid_size)

print("Train size: %d" % train_size)
print("Validation size: %d" % valid_size)
print("Test size: %d" % (test_size))

X_train, y_train, X_valid, y_valid, X_test, y_test = seperate_dataset_valid(X, y, train_size, valid_size)
output_neurons = features_out * steps_out
#batch_size = 4
batch_size = 1

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

X_train, X_valid, X_test = convert_X_to_3D_valid(X_train, X_valid, X_test, steps_in, features_in)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

print("----------------LSTM--------------")
# Δημιουργία LSTM μοντέλου
lstm_model = LSTM_model(X_train, y_train, X_valid, y_valid, batch_size, 300, features_out, steps_out)

original_data_lstm, predicted_data_lstm = calculate_predictions_valid(lstm_model, X_train, X_valid, X_test, y_train,
                                                                      y_valid, y_test, y_scaler, prediction_column, steps_out)
# Ημερομηνία που ξεκινάει το validation
date_validation = dates[df_greece_weekly.index[train_size]]
# Ημερομηνία που ξεκινάει το test
date_test = dates[df_greece_weekly.index[train_size + valid_size]]
print("Validation ", date_validation.to_pydatetime().date())
print("Test", date_test.to_pydatetime().date())

plot_predictions_valid_with_real_data(df_greece_weekly, dates, original_data_lstm, predicted_data_lstm, len(X_train), len(X_valid),
                 "Πρόβλεψη μέσων εβδομαδιαίων θανάτων πανελλαδικά με LSTM και +με βάση τον αριθμό ατόμων σε ΜΕΘ, "
                 "τους δείκτες κινητικότητας Google και τους θανάτους",
                 'Ημερομηνία', 'Αριθμός μέσων εβδομαδιαίων θανάτων πανελλαδικά')