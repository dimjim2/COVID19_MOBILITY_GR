from fitter import Fitter
from distfit import distfit
import matplotlib.pyplot as plt

# Μέθοδοι εύρεσης της κατανομής των δεδομένων

'''
Χρησιμοποιεί το αντικείμενο Fitter για να προσδιορίσει τις κατάλληλες πιθανοτικές κατανομές
που ταιριάζουν με τα δεδομένα που παρέχονται. Οι κατανομές τις οποίες ψάχνει να ταιριάξει ο
Fitter με τα δεδομένα προέρχονται από το Scipy, και κατά τον έλεγχο θα αγνοήσει αυτές όπου
εκπέμπουν σφάλματα ή τρέχουν επ αόριστον.
'''
def find_best_distribution_fitter(X, description):
    f = Fitter(X)
    f.fit()
    print(f.summary())
    plt.title(description)
    plt.show()
    print(f.get_best(method='sumsquare_error'))

'''
Προσδιορίζει την κατανομή που ταιριάζει καλύτερα στα δεδομένα μέσα από ένα σύνολο 89 κατανομών. 
Με μια τυχαία μεταβλητή ως είσοδο, η distfit μπορεί να βρει την καλύτερη προσαρμογή για παραμετρικές,
μη παραμετρικές και διακριτές κατανομές.
'''
def find_best_distribution_distfit(X, description):
    dist = distfit()
    dist.fit_transform(X)
    print(dist.summary)
    dist.plot(title=description)
    plt.show()