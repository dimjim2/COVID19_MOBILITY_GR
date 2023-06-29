import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer

from Plots.plots import make_boxplot, make_hist, make_scatter_plot, make_two_subplots, make_plot_apple_mobility_indexes
from utils.dataOperations import load_data_from_table


df_apple_greece = load_data_from_table("apple_mobility_trends")
print(df_apple_greece)
print(df_apple_greece.dtypes)
# Εκτυπώνουμε τις πρώτες 5 γραμμές
print(df_apple_greece.head())
# Εκτυπώνουμε τις στήλες
print(df_apple_greece.columns)
cols = ["driving", "walking"]
make_plot_apple_mobility_indexes(df_apple_greece, "Προβολή δεικτών κινητικότητας ανά γεωγραφική περιοχή")
make_boxplot(df_apple_greece, cols, "Boxplot δεικτών κινητικότητας Apple μετά τον καθαρισμό")
make_hist(df_apple_greece, cols, 50, "Ιστόγραμμα Apple mobility dataset έπειτα από καθαρισμό")
make_scatter_plot(df_apple_greece, "driving", "Scatter plot για Driving έπειτα από τον καθαρισμό", "ημερομηνία", "οδήγηση")
make_scatter_plot(df_apple_greece, "walking", "Scatter plot για Walking έπειτα από τον καθαρισμό", "ημερομηνία", "περπάτημα")
make_two_subplots(df_apple_greece, "driving", "walking", "Χρονοσειρές κινητικότητας Apple κάτα την διάρκεια της πανδημίας",
                      'Οδήγηση', "Περπάτημα")
print(df_apple_greece.dtypes)

# lower quantile
Q1 = df_apple_greece[cols].quantile(0.25)
# upper quantile
Q3 = df_apple_greece[cols].quantile(0.75)
# 50 % quantile,median
Q2 = df_apple_greece[cols].quantile(0.50)
# υπολογισμός του ΙQR
IQR = Q3 - Q1
print(" IQR ")
print(IQR)
print("Ακραίες τιμές")
print(((df_apple_greece[cols] < (Q1 - 1.8 * IQR)) | (df_apple_greece[cols] > (Q3 + 1.8 * IQR))).sum())

#print("Ακραίες τιμές μετα την αντικατάσταση με το Q2")
#print(((df_greece[cols] < (Q1 - 1.8 * IQR)) | (df_greece[cols] > (Q3 + 1.8 * IQR))).sum())

'''
Μετασχηματίζει τα χαρακτηριστικά κανονικοποιώντας κάθε χαρακτηριστικό στο διάστημα [0,1]
'''
scaler = MinMaxScaler()
X = df_apple_greece[cols].values
# Κάνει fit τα δεδομένα υπολογίζοντας το min και το max που θα χρησιμοποιηθεί στο scaling και έπειτα γίνεται ο μετασχηματισμός.
X_scaled = pd.DataFrame(scaler.fit_transform(X))
X_scaled = X_scaled.values
print("Κανονικοποιήση με MinMaxScaler")
print(X_scaled)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Κανονικοποιήση με Standard Scaler Z-Score Normalization")
print(X_scaled)

'''
Διακριτοποιήση των δεδομένων.
Παράμετροι
n_bins-> ο αριθμός των κάδων - intervals που θα δημιουργηθούν.
encode -> Χρήση  κωδικοποίησης ordinal που αριθμεί και διατάσσει τις ομάδες.Τα bins κωδικοποιούνται με ακέραιους αριθμούς
strategy -> 
uniform -Τεχνική ίσου πλάτους όπου όλα τα bins έχουν το ίδιο πλάτος
quantile -Τεχνική ίσης συχνότητας -βάθους όπου όλα τα bins προσπαθούμε να έχουν τον ίδιο αριθμό δεδομένων
kmeans -> Κάθε στοιχείο μέσα στο bin βρίσκεται πιο κόντα στο κέντρο του κάθε cluster από Kmeans
'''

print("Διακριτοποιήση με τεχνική ίσου πλάτους")
disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
data = disc.fit_transform(X_scaled)
print(data)

print("Διακριτοποιήση με τεχνική ίσης συχνότητας")
disc = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
data = disc.fit_transform(X_scaled)
print(data)

print("Διακριτοποιήση με τεχνική Kmeans")
disc = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')
data = disc.fit_transform(X_scaled)
print(data)

'''
Principal Component Analysis
Γραμμική μείωση διαστάσεων χρησιμοποιώντας Singular Value Decomposition για να προβάλλουμε τα δεδομένα μας σε έναν
χώρο με μικρότερο χώρο διαστάσεων.Είναι ιδιαιτέρα σημαντικό να αναφερθεί ότι αυτή η μέθοδος δεν απαλείφει τα χαρακτηριστικά
αλλά τα μετασχηματίζει.Τα μετασχηματισμένα χαρακτηριστικά είναι ανεξάρτητα μεταξύ τους.
Παράμετροι
n_components -> αριθμός των component  που θα διατηρήσουμε
'''
pca = PCA(1)
# Κάνουμε fit και εφαρμόζουμε μετασχηματισμό στα δεδομένα
df1 = pca.fit_transform(X_scaled)
print(df1.shape)
# Τα components είναι ταξινομημένα με βάση τα explained_variance_
print("PCA components")
print(pca.n_components)
# To ποσό της διακύμανσης για κάθε component.
# Είναι ίσο το n_components μεγαλύτερες ιδιοτιμές της μήτρας συνδιακύμανσης του Χ.
print("PCA explained variance")
print(pca.explained_variance_)

'''
Locally linear embedding (LLE) αναζητεί μια προβολή των δεδομένων σε έναν μικρότερο χώρο διατηρώντας
τις αποστάσεις ανάμεσα στις τοπικές γειτονιές.Είναι ένα μέρος Principal Component Analyses που έχουν
στόχο την εύρεση της καλύτερης μη γραμμικής embedding.
n_neighbors -> αριθμός των γειτόνων για κάθε σημείο
n_components -> αριθμός features μετά την μείωση διαστάσεων
'''
embedding = LocallyLinearEmbedding(n_components=1, n_neighbors=5, eigen_solver='dense')
X_transformed = embedding.fit_transform(X_scaled)
print(X_transformed.shape)
print(X_transformed)
print("Number of features seen during fit")

# αριθμός δεδομένων εισόδου
print(embedding.n_features_in_)
# προβάλλει το σφάλμα ανακατασκευής που είναι συσχετισμένα με τα embedding διανύσματα
print("Reconstruction error associated with embedding_")
print(embedding.reconstruction_error_)
# αποθηκεύει τα embedding διανύσματα
print("the embedding vectors")
print(embedding.embedding_)

'''
Επιστρέφει ένα τυχαίο δείγμα γραμμών από το dataset.
Δέχεται σαν παράμετρο τον αριθμό των δειγμάτων που θα προκύψουν από δειγματοληψία χωρίς αντικατάσταση
'''
print("Δειγματοληψία χωρίς αντικατάσταση")
print(df_apple_greece.sample(650))
df_Attica = df_apple_greece[df_apple_greece["region"] == "Athens"]
print(df_Attica.shape)
print(df_Attica)
