import pandas as pd

# Μετατρέπει μια χρονοσειρά σε supervised μορφή
def convert_timeseries_to_supervised(sdf, steps_in, steps_out, col_names):
    df = pd.DataFrame()
    # Χρήση των n_in προηγούμενων τιμών των δεικτών για να προβλέψουμε το μέλλον
    for i in range(steps_in, 0, -1):
        # Κάνουμε shift τον index axis κατά i περιόδους στη θετική κατεύθυνση
        # Κάνουμε copy του αντικείμενου συμπεριλαμβανομένων των indices και των data
        sdf_shifted = sdf.shift(i).copy()
        '''
        Αλλάζουμε τα ονόματα των στηλών .
        columns : Δέχεται σαν dictionary τα τροποποιημένα ονόματα των στηλών. Τα ονόματα των στηλών θα τροποποιηθούν
        με βάση σε ποια χρονική στιγμή του παρελθόντος εκφράζουν.
        inplace : Εκφράζει αν θα επιστρέψουμε ένα νεό Dataframe object. Επειδή οι τιμές είναι true δεν επιστρέφεται νέο 
        object και οι αλλαγές γίνονται στο ίδιο αντικείμενο dataframe.
        '''
        sdf_shifted.rename(columns=lambda x: ('%s(t-%d)' % (x, i)), inplace=True)
        '''
        Κάνουμε concatenate και ενώνουμε τα dataframes object με βάση έναν συγκεκριμένο άξονα
        Ορίσματα
        objs : Μια λίστα με τα dataframes object τα οποία επιθυμούμε να ενώσουμε
        axis : ο άξονας που θα χρησιμοποιηθεί για την συνένωση.Χρησιμοποιούμε άξονα 1 γιατί αναφερόμαστε σε στήλες.
        '''
        df = pd.concat([df, sdf_shifted], axis=1)

    for i in range(0, steps_out):
        # Θα κάνουμε shift μόνο για τις τιμές των δεικτών που θέλουμε να προβλέψουμε οι οποίες προσδιορίζονται από
        # το col_names
        sdf_shifted = sdf[col_names].shift(-i).copy()
        if i == 0:
            # Αλλάζουμε τα ονόματα των στηλών εξόδου που θέλουμε να προβλέψουμε
            sdf_shifted.rename(lambda x: ('%s(t)' % (x)), axis='columns', inplace=True)
        else:
            sdf_shifted.rename(lambda x: ('%s(t+%d)' % (x, i)), axis='columns', inplace=True)
        # Συνενώνουμε το dataframe df με το sdf_shifted dataframe που προκύπτει κάθε φορά
        df = pd.concat([df, sdf_shifted], axis=1)

    # Αφαιρούμε τις γραμμές που εντοπίζουμε missing values και αποθηκεύονται οι αλλαγές στο dataframe
    df.dropna(inplace=True)
    return df
