import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import timeseries_conv
from Plots.plots_predictions_from_model import plot_google_predictions_valid_with_real_data
from Predictions_problems.models import LSTM_model, Support_vector_regressor, Random_Forests, XGBoost_model, \
    MLP_neural_netowrk_valid, CNN_model, RNN_neural_network, GRU_model, ConvLSTM
from Predictions_problems.model_utils import seperate_dataset_valid, convert_X_to_3D_valid, \
    calculate_predictions_valid, get_dates, seperate_dataset, calculate_predictions
from utils.dataOperations import load_data_from_table

'''
Κατασκευάζουμε το dataset το οποίο αποτελείται τους πανελλήνιους δείκτες κινητικότητας Google, τους εμβολιασμούς,
τους θανάτους, τον αριθμό των κρουσμάτων και των ασθενών σε ΜΕΘ 
'''
def construct_dataset():
    df_google = load_data_from_table("google_region_mobility")
    df_google.drop(['country_region'], axis=1, inplace=True)
    df_google_greece = df_google[df_google["sub_region"] == "Greece"]
    df_google_greece.reset_index(drop=True, inplace=True)
    print(df_google_greece)

    df_vaccinations = load_data_from_table("vaccinations_data_history_per_area")
    print(df_vaccinations.groupby('date')["dailyvaccinations"].sum())
    df_vaccinations_greece = df_vaccinations[['date']].drop_duplicates().copy()
    df_vaccinations_greece["dailyvaccinations"] = df_vaccinations.groupby('date')["dailyvaccinations"].sum().values
    df_vaccinations_greece.reset_index(drop=True, inplace=True)
    df_vaccinations_greece["dailyvaccinations"].iloc[1:] = df_vaccinations_greece["dailyvaccinations"].diff().iloc[1:]
    print(df_vaccinations_greece)

    df_deaths = load_data_from_table("deaths")
    df_deaths.drop(['deaths_cum'], axis=1, inplace=True)
    print(df_deaths)

    df_cases_greece = load_data_from_table("covid_cases_greece")
    df_cases_greece.drop(['total_cases'], axis=1, inplace=True)
    print(df_cases_greece)

    df_intensive_care_cases = load_data_from_table("intensive_care_cases")
    print(df_intensive_care_cases)

    df_greece = pd.merge(pd.merge(df_google_greece, pd.merge(df_vaccinations_greece,
                                                             pd.merge(df_intensive_care_cases, df_deaths, on='date'),
                                                             on='date'), on='date'), df_cases_greece, on='date')
    print(df_greece)
    print(df_greece.isna().sum())
    return df_greece


dataframe_greece = construct_dataset()

# Βήμα χρονικών στιγμών που χρησιμοποιούνται για την πρόβλεψη
# steps_in = 180
steps_in = 14
# steps_in = 2
# Βήμα πρόβλεψης του μέλλοντος
steps_out = 1
print(len(dataframe_greece))

dates = get_dates(dataframe_greece, steps_in, steps_out)
dates.reset_index(drop=True, inplace=True)
print(dates)

dataframe_greece = dataframe_greece.drop(['date', 'sub_region'], axis=1)
features_in = len(dataframe_greece.columns)


prediction_columns = ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline',
'residential_percent_change_from_baseline']

features_out = len(prediction_columns)
print("Features in")
print(features_in)
print("Features out")
print(features_out)

# Μετατροπή της χρονοσειράς σε πρόβλημα επιτηρούμενης μάθησης
tsdf = timeseries_conv.convert_timeseries_to_supervised(dataframe_greece, steps_in, steps_out, prediction_columns)
# Δημιουργούμε τα χαρακτηριστικά X και τις ετικέτες labels
X, y = tsdf[[('%s(t-%d)' % (x, i)) for i in range(steps_in, 0, -1) for x in dataframe_greece.columns]].values, tsdf[[('%s(t)' % (x)) if i == 0 else ('%s(t+%d)' % (x, i)) for i in range(0, steps_out, 1) for x in prediction_columns]].values

print("Dataset")
print(X.shape)
print(y.shape)

# Μετασχηματίζει τα χαρακτηριστικά κανονικοποιώντας με τον Standard Scaler
scaler = StandardScaler()
y_scaler = StandardScaler()

# Κάνει fit τα δεδομένα υπολογίζοντας το μέσο και την διακύμανση που θα χρησιμοποιηθεί στο scaling και
# έπειτα γίνεται ο μετασχηματισμός των δεδομένων.

X = scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

size_of_dataset = X.shape[0]

train_size = int(0.60 * size_of_dataset)


X_train, y_train, X_test,  y_test = seperate_dataset(X, y, train_size)
print(X_train.shape)
print(y_train.shape)

# Πρόβλεψη από SVR
print("----------------SVR--------------")
svr_model = Support_vector_regressor(X_train, y_train)
original_data_svr, predicted_data_svr = calculate_predictions(svr_model, X_train, X_test, y_train,
                                                              y_test, y_scaler, prediction_columns, steps_out)
print("----------------Random Forests--------------")
model = Random_Forests(X_train, y_train, 200, 5)
original_data_rf, predicted_data_rf = calculate_predictions(model, X_train, X_test, y_train,
                                                            y_test, y_scaler, prediction_columns, steps_out)


print("----------------XGBOOST--------------")
xgboost_model = XGBoost_model(X_train, y_train)
original_data_XGBOOST, predicted_data_XGBOOST = calculate_predictions(xgboost_model, X_train, X_test, y_train,
y_test, y_scaler, prediction_columns, steps_out)
#
valid_size = int(size_of_dataset * .2)
test_size = size_of_dataset - (train_size + valid_size)

print("Train size: %d" % train_size)
print("Validation size: %d" % valid_size)
print("Test size: %d" % (test_size))

X_train, y_train, X_valid, y_valid, X_test, y_test = seperate_dataset_valid(X, y, train_size, valid_size)
output_neurons = features_out * steps_out
batch_size = 4

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)
#
print("----------------MLP--------------")
mlp_model = MLP_neural_netowrk_valid(X_train, y_train, X_valid, y_valid, 200, 4, features_out * steps_out)
original_data_mlp, predicted_data_mlp = calculate_predictions_valid(mlp_model, X_train, X_valid, X_test, y_train, y_valid,
                                                              y_test, y_scaler, prediction_columns, steps_out)

X_train, X_valid, X_test = convert_X_to_3D_valid(X_train, X_valid, X_test, steps_in, features_in)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)
print("----------------CNN--------------")
CNN_model = CNN_model(X_train, y_train, X_valid, y_valid, batch_size, 50, steps_out, features_out)

original_data_CNN, predicted_data_CNN = calculate_predictions_valid(CNN_model, X_train, X_valid, X_test, y_train, y_valid, y_test, y_scaler,
                                                                    prediction_columns, steps_out)
print("----------------LSTM--------------")
lstm_model = LSTM_model(X_train, y_train, X_valid, y_valid, 4, 300, features_out, steps_out)

original_data_lstm, predicted_data_lstm = calculate_predictions_valid(lstm_model, X_train, X_valid, X_test, y_train,
                                                                      y_valid, y_test, y_scaler, prediction_columns, steps_out)

print("----------------RNN--------------")
model = RNN_neural_network(X_train, y_train, X_valid, y_valid, 300, batch_size, output_neurons)


original, predicted = calculate_predictions_valid(model, X_train, X_valid, X_test, y_train,
                                                  y_valid, y_test, y_scaler, prediction_columns, steps_out)
print("----------------GRU--------------")
model = GRU_model(X_train, y_train, X_valid, y_valid, batch_size, 150, output_neurons)

original_data_GRU, predicted_data_GRU = calculate_predictions_valid(model, X_train, X_valid, X_test, y_train,
                                                                    y_valid, y_test, y_scaler, prediction_columns, steps_out)
print("----------------CONVLSTM--------------")
model = ConvLSTM(X_train, y_train, X_valid, y_valid, 150, batch_size, features_out, steps_out)

original_data_ConvLSTM, predicted_data_ConvLSTM = calculate_predictions_valid(model, X_train, X_valid, X_test, y_train,
                                                                              y_valid, y_test, y_scaler, prediction_columns, steps_out)
