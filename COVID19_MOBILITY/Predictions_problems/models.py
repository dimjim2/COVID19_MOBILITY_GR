# from keras import Sequential
# from keras import regularizers
import tensorflow as tf
from keras.layers import Conv2D
# from keras.regularizers import l1_l2, l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout, SimpleRNN, MaxPooling2D
from sklearn.svm import NuSVR, SVR
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.python.keras.layers import GRU, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Dense, RepeatVector
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l1_l2, l2
from xgboost import XGBRegressor
from tensorflow.python.keras import backend as K, regularizers
from tensorflow.python.keras.optimizers import adam_v2

# Υπολογισμός μέσου τετραγωνικού σφάλματος
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Δημιουργία πολυστρωματικού νευρωνικού δικτύου και χρήση validation set
def MLP_neural_netowrk_valid(X_train, y_train, X_valid, y_valid, number_of_epochs, batch_size, output_neurons):
    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    model.add(Dense(units=120, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(units=60, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(output_neurons))
    model.compile(loss=root_mean_squared_error, optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, callbacks=[callback], validation_data=(X_valid, y_valid), epochs=number_of_epochs,
                        batch_size=batch_size, verbose=1)
    return model

# Δημιουργία πολυστρωματικού νευρωνικού δικτύου
def MLP_neural_netowrk(X_train, y_train, number_of_epochs, batch_size, output_neurons):
    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15)
    model.add(Dense(units=100, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(output_neurons))
    model.compile(loss=root_mean_squared_error, optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, callbacks=[callback], epochs=number_of_epochs,
                        batch_size=batch_size, verbose=1)
    return model

# Δημιουργία αναδρομικού νευρωνικού δικτύου
def RNN_neural_network(X_train, y_train, X_valid, y_valid, number_of_epochs, batch_size, output_neurons):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=15, mode='min')
    # Προσθήκη και στατιστικών του μοντέλου στο Tensorboard
    callbacks_tensorboard = tf.keras.callbacks.TensorBoard(log_dir="AppleMobilityAtticaRNN/logs/fit", histogram_freq=1)

    model = Sequential()
    model.add(SimpleRNN(units=40, input_shape=(X_train.shape[1], X_train.shape[2]),  kernel_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(SimpleRNN(units=25, kernel_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation="relu"))
    model.add(Dense(output_neurons))

    # model.compile(loss=root_mean_squared_error, optimizer='adam', metrics=root_mean_squared_error)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics="mean_squared_error")
    model.fit(X_train, y_train, epochs=number_of_epochs, validation_data=(X_valid, y_valid), callbacks=[early_stop, callbacks_tensorboard],
              batch_size=batch_size, verbose=1)
    print(model.summary())
    return model

# Δημιουργία διανυσματικής μηχανής γισ προβλήματα regression
def Support_vector_regressor(X_train, y_train):
    svr = NuSVR(nu=0.6, C=1.5)
    svr = MultiOutputRegressor(svr)
    svr.fit(X_train, y_train)
    return svr

# Δημιουργία Gated recurrent unit
def GRU_model(X_train, y_train, X_valid, y_valid, batch_size, number_of_epochs, output_neurons):

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=15)
    model = Sequential()
    model.add(GRU(units=70, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, kernel_regularizer=regularizers.l2(l=0.01),
                  bias_regularizer=regularizers.l2(l=0.01), recurrent_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(GRU(units=40, kernel_regularizer=regularizers.l2(l=0.01), recurrent_regularizer=l2(0.01),
                  bias_regularizer=regularizers.l2(l=0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_neurons))

    # model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.fit(X_train, y_train, epochs=number_of_epochs, callbacks=[early_stop], validation_data=(X_valid, y_valid),
                        batch_size=batch_size, verbose=1)
    return model

# Δημιουργία δικτύου που συνδυάζει αρχιτεκτονικές συνελικτικού νευρωνικού δικτύου και LSTM
def ConvLSTM(X_train, y_train, X_valid, y_valid, epochs, batch_size, features_out, steps_out):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=15, mode='min')
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=0.01),
                     kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dropout(0.2))
    # model.add(RepeatVector(30))
    # model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, kernel_regularizer=regularizers.l2(l=0.01), activation='relu')))
    model.add(Dense(features_out * steps_out))
    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), batch_size=batch_size,
              callbacks=[early_stop], verbose=1)
    print(model.summary())
    return model

# Δημιουργία συνελεκτικού νευρωνικού δικτύου
def CNN_model(X_train, y_train, X_valid, y_valid, batch_size, epochs, steps_out, features_out):
    # Σταματάει την εκπαίδευση όταν η απώλεια σφάλαματος στο σύνολο validation έχει σταματήσει να μειώνεται
    # (μείωση του σφάλματος θεωρείται αποδεκτή αν είναι μεγαλύτερη ή ίση του 0.005)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=15, min_delta=0.005, mode='min')
    # Μειώνει κατά 20% το learning rate όταν το val_loss έχει σταματήσει να μειώνεται έπειτα από 15 εποχές
    rlrop_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, mode='min', patience=15)
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', kernel_regularizer=l1_l2(0.01, 0.01),
                     bias_regularizer=l1_l2(0.01, 0.01), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(RepeatVector(steps_out))
    model.add(TimeDistributed(Dense(30, activation='relu')))
    model.add(TimeDistributed(Dense(features_out)))

    # model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.fit(X_train, y_train, epochs=150, validation_data=(X_valid, y_valid), batch_size=batch_size,
                        callbacks=[early_stop, rlrop_callback], verbose=1)
    print(model.summary())
    return model

# Δημιουργία LSTM μοντέλου
def LSTM_model(X_train, y_train, X_valid, y_valid, batch_size, epochs, features_out, steps_out):
    callbacks_tensorboard = tf.keras.callbacks.TensorBoard(log_dir="AppleMobilityAtticaLSTM/logs/fit", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=15)
    model = Sequential()
    model.add(LSTM(80, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.01),
                   recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(40, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(30))
    model.add(Dense(features_out * steps_out))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # model.compile(loss=root_mean_squared_error, optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=[early_stop, callbacks_tensorboard],
                        batch_size=batch_size, verbose=1)
    return model

# Δημιουργία τυχαίου δάσους για επίλυση προβλημάτων regression
def Random_Forests(X_train, y_train, number_estimators, maximum_depth):
    # πλήθος χαρακτηριστικών σε κάθε δέντρο ίσο με τον λογάριθμο του αριθμού των χαρακτηριστικών του συνόλου δεδομένων
    # Χρησιμοποιεί τα δείγματα που δε συμμετείχαν στη διαδικασία εκπαίδευσης
    # ενός δέντρου απόφασης για να υπολογίσει generalization error (out-of-bag error)
    model = RandomForestRegressor(n_estimators=number_estimators, criterion='squared_error', oob_score=True, max_features='log2', max_depth=maximum_depth)
    model.fit(X_train, y_train)
    return model

# Δημιουργία XGBOOST μοντέλου
def XGBoost_model(X_train, y_train):
    # model = XGBRegressor(eta=0.05, max_depth=5, alpha=2, gamma=5)
    model = XGBRegressor(eta=0.05, n_estimators=5000, max_depth=3, alpha=2, gamma=0, min_child_weight=6, subsample=0.8, colsample_bytree=0.8)
    model = MultiOutputRegressor(model)
    xgbr = model.fit(X_train, y_train, eval_metric='rmse')
    return xgbr



