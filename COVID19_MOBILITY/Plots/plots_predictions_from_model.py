from matplotlib import pyplot as plt, pyplot
import seaborn as sns

# Δημιουργία correlation matrix σχετικά με τη στήλη πρόβλεψης
def plot_correlation_matrix_prediction_column(correlation_matrix, prediction_feature):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, ax=ax, linewidth=.5)
    plt.title("Μήτρα σύγχυσης αναφορικά με το χαρακτηριστικό " + prediction_feature)
    # plt.figure(figsize=(2,2))
    plt.show()

# Προβολή των προβλέψεων από το μοντέλο μαζί με τα πραγματικά δεδομένα όταν χρησιμοποιείται και validation dataset
def plot_predictions_valid_with_real_data(dataframe, dates, original, predicted, train_size, valid_size, title,
                                          x_label, y_label):
    plt.rcParams["figure.figsize"] = (17, 17)
    plt.plot(dates, original, 'g', label="Πραγματική")
    plt.plot(dates, predicted, 'r', label="Προβλεπόμενη")
    plt.legend()
    plt.title(title)

    '''
    Η axvline προσθέτει κάθετη γραμμή στους άξονες.
    Όρισματα 
    x-> θέση στην οποία θα τραβήξουμε την γραμμή
    c-> χρώμα της γραμμής
    '''

    print("Ημερομηνία που ξεκινάει το validation", dates[dataframe.index[train_size]].date())
    print("Ημερομηνία που ξεκινάει το testing", dates[dataframe.index[train_size + valid_size]].date())

    plt.xlabel(x_label, labelpad=30)
    plt.ylabel(y_label, labelpad=30)
    plt.axvline(dates[dataframe.index[train_size]], c="b")
    plt.axvline(dates[dataframe.index[train_size + valid_size]], c="m")
    plt.show()

# Προβολή των προβλέψεων από το μοντέλο μαζί με τα πραγματικά δεδομένα
def plot_predictions_with_real_data(dataframe, dates, original, predicted, train_size, title,
                                    x_label, y_label):
    plt.rcParams["figure.figsize"] = (17, 17)
    plt.plot(dates, original, 'g', label="Πραγματική")
    plt.plot(dates, predicted, 'r', label="Προβλεπόμενη")
    plt.legend()
    plt.title(title)
    '''
    Η axvline προσθέτει κάθετη γραμμή στους άξονες.
    Όρισματα 
    x-> θέση στην οποία θα τραβήξουμε την γραμμή
    c-> χρώμα της γραμμής
    '''

    print("Ημερομηνία που ξεκινάει το testing", dates[dataframe.index[train_size]].date())

    plt.xlabel(x_label, labelpad=30)
    plt.ylabel(y_label, labelpad=30)
    plt.axvline(dates[dataframe.index[train_size]], c="b")
    plt.show()

# Προβολή με barplot της σημαντικότητας των μεταβλητών
def plot_importance_variables(importances,features):
    print("importances", importances)
    print("features", features)

    plt.style.use('fivethirtyeight')
    plt.rcParams["figure.figsize"] = (12, 10)

    # Θέσεις της x μεταβλητής
    x_values = list(range(len(importances)))
    # Δημιουργία bar chart
    plt.bar(x_values, importances, orientation='vertical')
    plt.gcf()
    plt.xticks(x_values, features, fontsize=8)

    plt.ylabel('Επίπεδο σημαντικότητας')
    plt.title('Σημαντικότητα μεταβλητών')
    plt.show()

# Προβολή των προβλέψεων για τις Apple κστηγορίες από
# το μοντέλο μαζί με τα πραγματικά δεδομένα
def plot_predictions_apple_with_real_data(dates, original, predicted, dataframe, train_size, prediction_columns, message):
    fig, axs = plt.subplots(2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()
    axs[0].plot(dates, original[:, 0], 'g', label="Πραγματική")
    axs[0].plot(dates, predicted[:, 0], 'r', label="Προβλεπόμενη")
    axs[0].set_title(prediction_columns[0])
    axs[0].legend(loc="upper right")
    axs[0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0].set_xlabel("Ημερομηνία", labelpad=30)
    axs[0].set_ylabel(prediction_columns[0], labelpad=30)

    axs[1].plot(dates, original[:, 1], 'g', label="Πραγματική")
    axs[1].plot(dates, predicted[:, 1], 'r', label="Προβλεπόμενη")
    axs[1].set_title(prediction_columns[1])
    axs[1].legend(loc="upper right")
    axs[1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1].set_xlabel("Ημερομηνία", labelpad=30)
    axs[1].set_ylabel(prediction_columns[1], labelpad=30)

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

# Προβολή των προβλέψεων για τις Apple κατηγορίες από
# το μοντέλο μαζί με τα πραγματικά δεδομένα όταν χρησιμοποιείται και validation dataset
def plot_predictions_apple_valid_with_real_data(dates, original, predicted, dataframe, train_size, valid_size, prediction_columns, message):
    fig, axs = plt.subplots(2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()
    axs[0].plot(dates, original[:, 0], 'g', label="Πραγματική")
    axs[0].plot(dates, predicted[:, 0], 'r', label="Προβλεπόμενη")
    axs[0].set_title(prediction_columns[0])

    axs[0].legend(loc="upper right")
    axs[0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")
    axs[0].set_xlabel("Ημερομηνία", labelpad=30)
    axs[0].set_ylabel(prediction_columns[0], labelpad=30)

    axs[1].plot(dates, original[:, 1], 'g', label="Πραγματική")
    axs[1].plot(dates, predicted[:, 1], 'r', label="Προβλεπόμενη")
    axs[1].set_title(prediction_columns[1])
    axs[1].legend(loc="upper right")
    axs[1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")
    axs[1].set_xlabel("Ημερομηνία", labelpad=30)
    axs[1].set_ylabel(prediction_columns[1], labelpad=30)

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

'''
Προβολή των προβλέψεων για τις Google προβλέψεις από
το μοντέλο μαζί με τα πραγματικά δεδομένα όταν χρησιμοποιείται και validation dataset
'''
def plot_google_predictions_valid_with_real_data(dates, original, predicted, dataframe, train_size, valid_size, message):
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()
    axs[0, 0].plot(dates, original[:, 0], 'g', label="Πραγματική")
    axs[0, 0].plot(dates, predicted[:, 0], 'r', label="Προβλεπόμενη")
    axs[0, 0].set_title('Retail and recreation')
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0, 0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[0, 1].plot(dates, original[:, 1], 'g', label="Πραγματική")
    axs[0, 1].plot(dates, predicted[:, 1], 'r', label="Προβλεπόμενη")
    axs[0, 1].set_title('Grocery and pharmacy')
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0, 1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[1, 0].plot(dates, original[:, 2], 'g', label="Πραγματική")
    axs[1, 0].plot(dates, predicted[:, 2], 'r', label="Προβλεπόμενη")
    axs[1, 0].set_title('Parks')
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1, 0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[1, 1].plot(dates, original[:, 3], 'g', label="Πραγματική")
    axs[1, 1].plot(dates, predicted[:, 3], 'r', label="Προβλεπόμενη")
    axs[1, 1].set_title('Transit stations')
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1, 1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[2, 0].plot(dates, original[:, 4], 'g', label="Πραγματική")
    axs[2, 0].plot(dates, predicted[:, 4], 'r', label="Προβλεπόμενη")
    axs[2, 0].set_title('Workplaces')
    axs[2, 0].legend(loc="upper right")
    axs[2, 0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[2, 0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[2, 1].plot(dates, original[:, 5], 'g', label="Πραγματική")
    axs[2, 1].plot(dates, predicted[:, 5], 'r', label="Προβλεπόμενη")
    axs[2, 1].set_title('Residential')
    axs[2, 1].legend(loc="upper right")
    axs[2, 1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[2, 1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

'''
Προβολή των προβλέψεων για τις Google προβλέψεις από
το μοντέλο μαζί με τα πραγματικά δεδομένα
'''
def plot_google_predictions_with_real_data(dates, original, predicted, dataframe, train_size, message):
    # Δημιουργία plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία
    fig, axs = plt.subplots(3, 2, figsize=(20,20))
    pyplot.suptitle(message)
    fig.tight_layout()
    axs[0, 0].plot(dates, original[:, 0], 'g', label="Πραγματική")
    axs[0, 0].plot(dates, predicted[:, 0], 'r', label="Προβλεπόμενη")
    axs[0, 0].set_title('Retail and recreation')
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].axvline(dates[dataframe.index[train_size]], c="b")

    axs[0, 1].plot(dates, original[:, 1], 'g', label="Πραγματική")
    axs[0, 1].plot(dates, predicted[:, 1], 'r', label="Προβλεπόμενη")
    axs[0, 1].set_title('Grocery and pharmacy')
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].axvline(dates[dataframe.index[train_size]], c="b")

    axs[1, 0].plot(dates, original[:, 2], 'g', label="Πραγματική")
    axs[1, 0].plot(dates, predicted[:, 2], 'r', label="Προβλεπόμενη")
    axs[1, 0].set_title('Parks')
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].axvline(dates[dataframe.index[train_size]], c="b")

    axs[1, 1].plot(dates, original[:, 3], 'g', label="Πραγματική")
    axs[1, 1].plot(dates, predicted[:, 3], 'r', label="Προβλεπόμενη")
    axs[1, 1].set_title('Transit stations')
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].axvline(dates[dataframe.index[train_size]], c="b")

    axs[2, 0].plot(dates, original[:, 4], 'g', label="Πραγματική")
    axs[2, 0].plot(dates, predicted[:, 4], 'r', label="Προβλεπόμενη")
    axs[2, 0].set_title('Workplaces')
    axs[2, 0].legend(loc="upper right")
    axs[2, 0].axvline(dates[dataframe.index[train_size]], c="b")

    axs[2, 1].plot(dates, original[:, 5], 'g', label="Πραγματική")
    axs[2, 1].plot(dates, predicted[:, 5], 'r', label="Προβλεπόμενη")
    axs[2, 1].set_title('Residential')
    axs[2, 1].legend(loc="upper right")
    axs[2, 1].axvline(dates[dataframe.index[train_size]], c="b")

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

# Προβολή των προβλέψεων από το μοντέλο
def plot_predictions_from_model(dataframe, dates, original, predicted, train_size, valid_size, title,
                           x_label, y_label):
    plt.rcParams["figure.figsize"] = (17, 17)

    plt.plot(dates[:dataframe.index[train_size]+valid_size], original, 'g', label="Πραγματική")
    plt.plot(dates[dataframe.index[train_size]+valid_size:], predicted, 'r', label="Προβλεπόμενη")
    plt.legend()
    plt.title(title)

    '''
    Η axvline προσθέτει κάθετη γραμμή στους άξονες.
    Όρισματα 
    x-> θέση στην οποία θα τραβήξουμε την γραμμή
    c-> χρώμα της γραμμής
    '''
    print("Ημερομηνία που ξεκινάει το validation", dates[dataframe.index[train_size]].date())
    print("Ημερομηνία που ξεκινάει το testing", dates[dataframe.index[train_size + valid_size]].date())

    plt.xlabel(x_label, labelpad=30)
    plt.ylabel(y_label, labelpad=30)
    plt.axvline(dates[dataframe.index[train_size]], c="b")
    plt.axvline(dates[dataframe.index[train_size + valid_size]], c="m")
    plt.show()

# Προβολή των προβλέψεων για τις Google κατηγορίες από το μοντέλο
def plot_google_predictions(dates, original, predicted, dataframe, train_size, valid_size, message):
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()
    axs[0, 0].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 0], 'g', label="Πραγματική")
    axs[0, 0].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 0], 'r', label="Προβλεπόμενη")
    axs[0, 0].set_title('Retail and recreation')
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0, 0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[0, 1].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 1], 'g', label="Πραγματική")
    axs[0, 1].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 1], 'r', label="Προβλεπόμενη")
    axs[0, 1].set_title('Grocery and pharmacy')
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0, 1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[1, 0].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 2], 'g', label="Πραγματική")
    axs[1, 0].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 2], 'r', label="Προβλεπόμενη")
    axs[1, 0].set_title('Parks')
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1, 0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[1, 1].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 3], 'g', label="Πραγματική")
    axs[1, 1].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 3], 'r', label="Προβλεπόμενη")
    axs[1, 1].set_title('Transit stations')
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1, 1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[2, 0].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 4], 'g', label="Πραγματική")
    axs[2, 0].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 4], 'r', label="Προβλεπόμενη")
    axs[2, 0].set_title('Workplaces')
    axs[2, 0].legend(loc="upper right")
    axs[2, 0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[2, 0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    axs[2, 1].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 5], 'g', label="Πραγματική")
    axs[2, 1].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 5], 'r', label="Προβλεπόμενη")
    axs[2, 1].set_title('Residential')
    axs[2, 1].legend(loc="upper right")
    axs[2, 1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[2, 1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

# Προβολή των προβλέψεων για τις Apple κατηγορίες από το μοντέλο
def plot_apple_predictions(dates, original, predicted, dataframe, train_size, valid_size, prediction_columns, message):
    fig, axs = plt.subplots(2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()
    axs[0].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 0], 'g', label="Πραγματική")
    axs[0].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 0], 'r', label="Προβλεπόμενη")
    axs[0].set_title(prediction_columns[0])
    axs[0].legend(loc="upper right")
    axs[0].axvline(dates[dataframe.index[train_size]], c="b")
    axs[0].axvline(dates[dataframe.index[train_size + valid_size]], c="m")
    axs[0].set_xlabel("Ημερομηνία", labelpad=30)
    axs[0].set_ylabel(prediction_columns[0], labelpad=30)

    axs[1].plot(dates[:dataframe.index[train_size]+valid_size], original[:, 1], 'g', label="Πραγματική")
    axs[1].plot(dates[dataframe.index[train_size]+valid_size:], predicted[:, 1], 'r', label="Προβλεπόμενη")
    axs[1].set_title(prediction_columns[1])
    axs[1].legend(loc="upper right")
    axs[1].axvline(dates[dataframe.index[train_size]], c="b")
    axs[1].axvline(dates[dataframe.index[train_size + valid_size]], c="m")
    axs[1].set_xlabel("Ημερομηνία", labelpad=30)
    axs[1].set_ylabel(prediction_columns[1], labelpad=30)

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

# Προβολή της ακρίβειας και της απώλειας του μοντέλου
def plot_history_loss_accuracy(history, model_message):
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.title("Απώλεια " + model_message)
    plt.plot(history.history['loss'], label="Απώλεια training")
    plt.plot(history.history['val_loss'], label="Απώλεια validation")
    plt.legend(loc="upper left")
    plt.subplot(2, 1, 2)
    plt.title("Ακρίβεια")
    plt.plot(history.history['accuracy'], label="Ακρίβεια training")
    plt.plot(history.history['val_accuracy'], label="Ακρίβεια validation")
    plt.legend(loc="upper left")
    plt.show()

# Προβολή της απώλειας του μοντέλου
def plot_history_loss(history, model_message):
    plt.figure(figsize=(20, 10))
    plt.title("Απώλεια για το μοντέλο " + model_message)
    plt.ylabel('Απώλεια')
    plt.xlabel('Εποχή')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.plot(history.history['loss'], label="Απώλεια training")
    plt.plot(history.history['val_loss'], label="Απώλεια validation")
    plt.legend(loc="upper left")
    plt.show()