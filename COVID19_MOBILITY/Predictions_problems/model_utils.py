import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score

# Εξάγουμε τις ημερομηνίες από το dataframe έπειτα από τον μετασχηματισμό της χρονοσειράς
def get_dates(dataframe, steps_in, steps_out):
    if (steps_out == 1):
        dates = dataframe['date'][steps_in:]
    else:
        dates = dataframe['date'][steps_in: 1 - steps_out]
    return dates

# Χωρίζουμε το dataset σε training, validation και test sets
def seperate_dataset_valid(X, y, train_size, valid_size):
    xtr, ytr = X[:train_size], y[:train_size]
    xva, yva = X[train_size:train_size + valid_size, :], y[train_size:train_size + valid_size, :]
    xte, yte = X[train_size + valid_size:, :], y[train_size + valid_size:, :]
    print(xtr.shape, ytr.shape)
    print(xva.shape, yva.shape)
    print(xte.shape, yte.shape)
    return xtr, ytr, xva, yva, xte, yte

# Χωρίζουμε το dataset σε training και test sets
def seperate_dataset(X, y, train_size):
    xtr, ytr = X[:train_size], y[:train_size]
    xte, yte = X[train_size:, :], y[train_size:, :]
    print(xtr.shape, ytr.shape)
    print(xte.shape, yte.shape)
    return xtr, ytr, xte, yte

# Μετατρέπουμε τα διανύσματα X σε 3D μορφή [samples, steps, features]
def convert_X_to_3D_valid(xtr, xva, xte, steps_in, features_in):
    X_train = np.reshape(xtr, (xtr.shape[0], steps_in, features_in))
    print(X_train.shape)

    X_valid = np.reshape(xva, (xva.shape[0], steps_in, features_in))
    print(X_valid.shape)

    X_test = np.reshape(xte, (xte.shape[0], steps_in, features_in))
    print(X_test.shape)
    return X_train, X_valid, X_test

# Υπολογίζουμε τα αποτελέσματα των προβλέψεων στα training, validation και test sets από το μοντέλο σε διάφορες μετρικές
# και επιστρέφει τα δεδομένα y original και predicted

def calculate_predictions_valid(model, X_train, X_valid, X_test, y_train_original, y_valid_original,
                                y_test_original, y_scaler, prediction_columns, steps_out):
    features_out = len(prediction_columns)

    y_train_predicted, y_valid_predicted, y_test_predicted = calculate_model_predictions_valid(X_train, X_valid, X_test,
                                                                                               model)
    print("Y train shape")
    print(y_train_predicted.shape)

    y_test_original, y_test_predicted, y_train_original, y_train_predicted, y_valid_original, y_valid_predicted = reshape_y_labels_valid_to_2D(
        features_out, steps_out, y_test_original, y_test_predicted, y_train_original, y_train_predicted,
        y_valid_original, y_valid_predicted)

    y_train_original, y_train_predicted, y_valid_original, \
        y_valid_predicted, y_test_original, y_test_predicted \
        = inverse_y_labels_valid(y_scaler, y_train_original, y_train_predicted, y_valid_original,
                                 y_valid_predicted, y_test_original, y_test_predicted)
    print("Συνολικές μετρικές")

    calculate_metrics_valid(y_train_original, y_train_predicted, y_valid_original,
                            y_valid_predicted, y_test_original, y_test_predicted)

    calculate_metrics_valid_by_feature(features_out, prediction_columns, steps_out, y_test_original, y_test_predicted, y_train_original,
                                       y_train_predicted, y_valid_original, y_valid_predicted)

    y_original = np.concatenate((y_train_original, y_valid_original, y_test_original), axis=0)
    y_predicted = np.concatenate((y_train_predicted, y_valid_predicted, y_test_predicted), axis=0)
    return y_original, y_predicted

# Υπολογίζουμε τις μετρικές για το κάθε feature όταν χρησιμοποιείται και validation set
def calculate_metrics_valid_by_feature(features_out, prediction_columns, steps_out, y_test_original, y_test_predicted, y_train_original,
                                       y_train_predicted, y_valid_original, y_valid_predicted):
    if (steps_out > 1):
        for i in range(features_out):
            print("--------------------------------------------------")
            print("Για το χαρακτηριστικό " + prediction_columns[i])
            calculate_metrics_valid(y_train_original[:, i::steps_out], y_train_predicted[:, i::steps_out],
                                y_valid_original[:, i::steps_out],
                                y_valid_predicted[:, i::steps_out], y_test_original[:, i::steps_out],
                                y_test_predicted[:, i::steps_out])
    else:
        for i in range(features_out):
            print("--------------------------------------------------")
            print("Για το χαρακτηριστικό " + prediction_columns[i])
            calculate_metrics_valid(y_train_original[:, i], y_train_predicted[:, i],
                                y_valid_original[:, i],
                                y_valid_predicted[:, i], y_test_original[:, i],
                                y_test_predicted[:, i])

# Υπολογίζουμε τις μετρικές για το κάθε feature
def calculate_metrics_by_feature(features_out, prediction_columns, steps_out, y_test_original, y_test_predicted, y_train_original,
                                       y_train_predicted):
    if (steps_out > 1):
        for i in range(features_out):
            print("--------------------------------------------------")
            print("Για το χαρακτηριστικό " + prediction_columns[i])
            calculate_metrics(y_train_original[:, i::steps_out], y_train_predicted[:, i::steps_out], y_test_original[:, i::steps_out],
                                y_test_predicted[:, i::steps_out])
    else:
        for i in range(features_out):
            print("--------------------------------------------------")
            print("Για το χαρακτηριστικό " + prediction_columns[i])
            calculate_metrics(y_train_original[:, i], y_train_predicted[:, i], y_test_original[:, i],
                                y_test_predicted[:, i])

# Μετατρέπει τα y labels των training, validation και test sets σε 2D μορφή
def reshape_y_labels_valid_to_2D(features_out, steps_out, y_test_original, y_test_predicted, y_train_original,
                                 y_train_predicted, y_valid_original, y_valid_predicted):

    if (y_train_predicted.ndim == 3 or y_train_predicted.ndim == 1):
        y_train_original, y_valid_original, y_test_original = \
            reshape_y_labels_2D_valid(y_train_original, y_valid_original, y_test_original, features_out, steps_out)
        y_train_predicted, y_valid_predicted, y_test_predicted = \
            reshape_y_labels_2D_valid(y_train_predicted, y_valid_predicted, y_test_predicted, features_out, steps_out)
    return y_test_original, y_test_predicted, y_train_original, y_train_predicted, y_valid_original, y_valid_predicted

# Μετατρέπει τα y labels των training και test sets σε 2D μορφή
def reshape_y_labels_to_2D(features_out, steps_out, y_test_original, y_test_predicted, y_train_original,
                                 y_train_predicted):

    if (y_train_predicted.ndim == 3 or y_train_predicted.ndim == 1):
        y_train_original, y_test_original = \
            reshape_y_labels_2D(y_train_original, y_test_original, features_out, steps_out)
        y_train_predicted, y_test_predicted = \
            reshape_y_labels_2D(y_train_predicted, y_test_predicted, features_out, steps_out)
    return y_test_original, y_test_predicted, y_train_original, y_train_predicted

# Υπολογίζουμε τα αποτελέσματα των προβλέψεων στα training και test sets από το μοντέλο σε διάφορες μετρικές
# και επιστρέφει τα δεδομένα y original και predicted
def calculate_predictions(model, X_train, X_test, y_train_original,
                          y_test_original, y_scaler, prediction_columns, steps_out):

    features_out = len(prediction_columns)
    y_train_predicted, y_test_predicted = calculate_model_predictions(X_train, X_test, model)
    print("Y train shape")
    print(y_train_predicted.shape)

    y_test_original, y_test_predicted, y_train_original, y_train_predicted = reshape_y_labels_to_2D(
        features_out, steps_out, y_test_original, y_test_predicted, y_train_original, y_train_predicted)

    y_train_original, y_train_predicted, y_test_original, y_test_predicted = inverse_y_labels(y_scaler, y_train_original,
                        y_train_predicted, y_test_original, y_test_predicted)

    print(y_test_original.shape)
    print(y_test_predicted.shape)

    calculate_metrics(y_train_original, y_train_predicted, y_test_original, y_test_predicted)
    calculate_metrics_by_feature(features_out, prediction_columns, steps_out, y_test_original, y_test_predicted, y_train_original,
                                       y_train_predicted)

    y_original = np.concatenate((y_train_original, y_test_original), axis=0)
    y_predicted = np.concatenate((y_train_predicted, y_test_predicted), axis=0)
    return y_original, y_predicted

# Κάνουμε αναστροφή του μετασχηματισμού των δεδομένων που εφαρμόστηκε στα y labels των training, validation, test sets
def inverse_y_labels_valid(y_scaler, y_train_original, y_train_predicted, y_valid_original,
                           y_valid_predicted, y_test_original, y_test_predicted):

    y_train_predicted = y_scaler.inverse_transform(y_train_predicted)
    y_train_original = y_scaler.inverse_transform(y_train_original)
    y_valid_predicted = y_scaler.inverse_transform(y_valid_predicted)
    y_valid_original = y_scaler.inverse_transform(y_valid_original)
    y_test_predicted = y_scaler.inverse_transform(y_test_predicted)
    y_test_original = y_scaler.inverse_transform(y_test_original)
    return y_train_original, y_train_predicted, y_valid_original, y_valid_predicted, \
        y_test_original, y_test_predicted

# Κάνουμε αναστροφή του μετασχηματισμού των δεδομένων που εφαρμόστηκε στα y labels των training, test sets
def inverse_y_labels(y_scaler, y_train_original, y_train_predicted, y_test_original, y_test_predicted):

    y_train_predicted = y_scaler.inverse_transform(y_train_predicted)
    y_train_original = y_scaler.inverse_transform(y_train_original)
    y_test_predicted = y_scaler.inverse_transform(y_test_predicted)
    y_test_original = y_scaler.inverse_transform(y_test_original)
    return y_train_original, y_train_predicted, y_test_original, y_test_predicted

# Κάνουμε reshape των y labels στα training,validation και test sets σε 2D μορφή
def reshape_y_labels_2D_valid(y_train, y_valid, y_test, features_out, steps_out):

    y_train = np.reshape(y_train, (y_train.shape[0], features_out * steps_out))
    y_valid = np.reshape(y_valid, (y_valid.shape[0], features_out * steps_out))
    y_test = np.reshape(y_test, (y_test.shape[0], features_out * steps_out))
    return y_train, y_valid, y_test

# Κάνουμε reshape των y labels στα training και test sets σε 2D μορφή
def reshape_y_labels_2D(y_train, y_test, features_out, steps_out):

    y_train = np.reshape(y_train, (y_train.shape[0], features_out * steps_out))
    y_test = np.reshape(y_test, (y_test.shape[0], features_out * steps_out))
    return y_train, y_test


# Υπολογίζουμε τις προβλέψεις του μοντέλου στα training, validation, test sets των y ετικετών
def calculate_model_predictions_valid(X_train, X_valid, X_test, model):
    y_train_predicted = model.predict(X_train)
    y_valid_predicted = model.predict(X_valid)
    y_test_predicted = model.predict(X_test)
    return y_train_predicted, y_valid_predicted, y_test_predicted

# Υπολογίζουμε τις προβλέψεις του μοντέλου στα training και test sets των y ετικετών
def calculate_model_predictions(X_train, X_test, model):

    y_train_predicted = model.predict(X_train)
    y_test_predicted = model.predict(X_test)
    return y_train_predicted, y_test_predicted

# Υπολογίζουμε τις προβλέψεις του μοντέλου στα training, validation, test sets των y ετικετών
# στις διάφορες μετρικές
def calculate_metrics_valid(y_train_original, y_train_predicted, y_valid_original,
                            y_valid_predicted, y_test_original, y_test_predicted):

    metrics(y_train_original, y_train_predicted, "Train")
    metrics(y_valid_original, y_valid_predicted, "Valid")
    metrics(y_test_original, y_test_predicted, "Test")

# Υπολογίζουμε τις προβλέψεις του μοντέλου στα training και test sets των y ετικετών στις διάφορες μετρικές
def calculate_metrics(y_train_original, y_train_predicted, y_test_original, y_test_predicted):

    metrics(y_train_original, y_train_predicted, "Train")
    metrics(y_test_original, y_test_predicted, "Test")


# Υπολογίζουμε τις προβλέψεις του μοντέλου στις μετρικές του Μέσου Απόλυτου Σφάλματος,
# του Ποσοστό Μέσου Απόλυτου Σφάλματος, του Explained Variance Score και R2 συντελεστή
def metrics(y_original, y_predicted, dataset_type):

    print("-----------------------------------------------------")
    print("Μέσο Απόλυτο Σφάλμα " + dataset_type, mean_absolute_error(y_original, y_predicted))
    # Μέσο απόλυτο ποσοστό σφάλμα.
    print("Ποσοστό Μέσου Απόλυτου Σφάλματος " + dataset_type, mean_absolute_percentage_error(y_original, y_predicted))
    # Explained variance regression score function.Μέγιστη καλύτερη τιμή είναι το 1.
    print("Explained Variance Score " + dataset_type, explained_variance_score(y_original, y_predicted))
    # Ο r2 coefficient έχει τιμές από 0-1 και μπορούμε να αξιολογήσουμε σωστά την πρόβλεψη.
    print("R2 συντελεστής " + dataset_type, r2_score(y_original, y_predicted))
