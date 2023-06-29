import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, KBinsDiscretizer, \
    RobustScaler, QuantileTransformer

from Plots.plots import make_boxplot, make_hist, make_google_plot, \
    make_google_plot_by_activity_regions, make_google_scatter_plot

from utils.dataOperations import load_data_from_table


df_google = load_data_from_table("google_region_mobility")
print(df_google)

# Εκτυπώνουμε τις πρώτες 5 γραμμές
print(df_google.head())
# Εκτυπώνουμε τις στήλες
print(df_google.columns)
# Παράγει στατιστικά
print(df_google.describe())
print(df_google.dtypes)

cols = ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']

make_hist(df_google, cols, 10, "Ιστόγραμμα Apple mobility dataset μετά από καθαρισμό")
df_google_Attica = df_google[df_google["sub_region"] == "Decentralized Administration of Attica"]
df_google_Greece = df_google[df_google["sub_region"] == "Greece"]
make_google_plot(df_google_Attica, "Ημερήσιοι δείκτες κινητικότητας στις κατηγορίες Google στην Αττική")
make_google_plot_by_activity_regions(df_google_Greece, "Google ανθρώπινες μετακινήσεις σε διαφορετικές κατηγορίες μερών")
# Δημιουργία scatter plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία
make_google_scatter_plot(df_google_Greece, "Scatter plot των πανελλήνιων δεικτών κινητικότητας")
make_boxplot(df_google, cols, "Boxplot στο dataset google mobility μετά το preprocessing")

'''
Δημιουργία boxplot για προβολής της κατανομής των δεδομένων χρησιμοποιώντας την five-number summary (minimum
,first quartile 25 % (Q1), median 50% , third quartile 75% (Q3), and maximum).Στο boxplot τα δεδομένα αναπαρίστανται με
το κουτί όπου η αρχή του κουτιού είναι το Q1 και το τέλος του το Q3,το median με μια γραμμή μέσα στο κουτί ,
τα minimum και τα maximum με τα whiskers και τα outliers με τα κυκλάκια.
'''


print("Mέγεθος dataset")
print(df_google.shape)
# Εμφανίζουμε τον αριθμό των στοιχείων στις οποίες έχουμε missing values
print(df_google.isna().sum())
print("--------------------------------------------")
print(df_google.isnull().sum())


# lower quantile
Q1 = df_google[cols].quantile(0.25)
# upper quantile
Q3 = df_google[cols].quantile(0.75)
# 50 % quantile,median
Q2 = df_google[cols].quantile(0.50)
# υπολογισμός του ΙQR
IQR = Q3 - Q1
print(" IQR ")
print(IQR)
print("Ακραίες τιμές")
print(((df_google[cols] < (Q1 - 1.5 * IQR)) | (df_google[cols] > (Q3 + 1.5 * IQR))).sum())


X = df_google[cols].values

normalizer = Normalizer()
X_scaled = normalizer.fit_transform(X)
print("Κανονικοποιήση -normalization με βάση την νόρμα l2")
print(X_scaled)

scaler = MaxAbsScaler()
# Κάνει fit τα δεδομένα υπολογίζοντας το min και το max που θα χρησιμοποιηθεί στο scaling και έπειτα γίνεται ο μετασχηματισμός.
X_scaled = pd.DataFrame(scaler.fit_transform(X))
X_scaled = X_scaled.values
print("Κανονικοποιήση με MaxAbsScaler")
print(X_scaled)
'''
Normalization με την χρήση του RobustScaler που είναι ανθεκτικός στους outliers.
Η κανονικοποιήση αφαιρεί τον median και μετασχηματίζει τα δεδομένα στο διάστημα IQR.
'''
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print("Κανονικοποιήση με RobustScaler")
print(X_scaled)
# Η μεσαία τιμή για το κάθε χαρακτηριστικό
print(scaler.center_)
# το μετασχηματισμένο IQR για κάθε χαρακτηριστικό
print(scaler.scale_)
# ο αριθμός των χαρακτηριστικών που δόθηκαν σαν είσοδος
print(scaler.n_features_in_)

'''
Η κανονικοποιήση QuantileTransformer μετασχηματίζει με μη γραμμικό τρόπο τα δεδομένα έτσι
ώστε να ακολουθούν την ομοιόμορφη ή την κανονική κατανομή.Ο μετασχηματίσμος απλώνεται/βασίζεται
στις πιο συχνές τιμές ενώ μειώνονται τα outliers.
Αρχικά ξεχωριστά για κάθε χαρακτηριστικό υπολογίζεται η cumulative distribution function του
για να γίνει προβολή των δεδομένων σε ομοιόμορφη κατανομή.Τα δεδομένα προβάλλονται έπειτα στην επιθυμητή
κατανομή με την χρήση των outliers.
n_quantiles ->αριθμός των quantiles που θα υπολογιστούν
output_distribution -> προβολή στην κανονική κατανομή
random_state -> αριθμός για υποδειγματολήψια και smoothing noise
'''
qt = QuantileTransformer(n_quantiles=10,output_distribution="normal", random_state=0)
X_scaled = qt.fit_transform(X)
print(X_scaled)
# Ο πραγματικός αριθμός των quantile που χρησιμοποιήθηκαν για την διακριτοποιήση της cumulative distribution function
print("the actual number of quantiles used to discretize the cumulative distribution function")
print(qt.n_quantiles)
# Οι τιμές που αντιστοιχούν στα quantiles
print("The values corresponding the quantiles of reference")
print(qt.quantiles_)


print("Διακριτοποιήση με τεχνική ίσου πλάτους")
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
data = disc.fit_transform(X_scaled)
print(data)
print("Διακριτοποιήση με τεχνική ίσης συχνότητας")
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
data = disc.fit_transform(X_scaled)
print(data)
print("Διακριτοποιήση με τεχνική Kmeans")
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
data = disc.fit_transform(X_scaled)
print(data)


'''
Principal Component Analysis
Γραμμική μείωση διαστάσεων χρησιμοποιώντας Singular Value Decomposition για να προβάλλουμε τα δεδομένα μας σε έναν
χώρο με μικρότερο χώρο διαστάσεων.Είναι ιδαιτέρα σημαντικό ότι αυτή η μέθοδος δεν απαλείφει τα χαρακτηριστικά
αλλά τα μετασχηματίζει.Τα μετασχηματισμένα χαρακητριστικά είναι ανεξάρτητα μεταξύ τους.
Παράμετροι
n_components -> αριθμός των component  που θα διατηρήσουμε
'''

#pca = PCA(3)
pca = PCA(n_components="mle")
# Κάνουμε fit και εφαρμόζουμε μετασχηματισμό στα δεδομένα
df1 = pca.fit_transform(X_scaled)
print(df1.shape)
# Τα components είναι ταξινομημένα με βάση τα explained_variance_
print("PCA components")
print(pca.n_components)
# To πόσο της διακύμανσης για κάθε component. Είναι ίσο το n_components μεγαλύτερες ιδιοτιμές της μήτρας συνδιακύμανσης του Χ.
print("PCA explained variance")
print(pca.explained_variance_)
# Singular values που αντιστοιχούν σε κάθε component.Για τον υπολογισμό του χρησιμοποιείται το γεγόνος ότι οι τιμές των
# singular values είναι ίσες με τις 2-νόρμες των n_component μεταβλητών στον χαμηλότερο χώρο διαστάσεων.
print("Singular Values")
print(pca.singular_values_)
print("df1", df1)

'''
Isomap Embedding
Μη γραμμική μείωση διαστάσεων με τη χρήση της Isometric
n_neighbors -> αριθμός των γειτόνων για κάθε σημείο
n_components -> αριθμός features μετά την μείωση διαστάσεων

'''

embedding = Isomap(n_components=2, n_neighbors=7)
X_transformed = embedding.fit_transform(X_scaled)
print(X_transformed)
print(X_transformed.shape)
print("Number of features seen during fit")
# αριθμός δεδομένων εισόδου
print(embedding.n_features_in_)
print(" geodesic distance matrix of training data")
print(embedding.dist_matrix_)
# αποθηκεύει τα embedding διανύσματα
print("the embedding vectors")
print(embedding.embedding_)

'''
Locally linear embedding (LLE) αναζητεί μια προβολή των δεδομένων σε έναν μικρότερο χώρο διατηρώντας
τις αποστάσεις ανάμεσα στις τοπικές γειτονιές.Είναι ένα μέρος Principal Component Analyses που έχουν
στόχο την εύρεση της καλύτερης μη γραμμικής embedding.
n_neighbors -> αριθμός των γειτόνων για κάθε σημείο
n_components -> αριθμός features μετά την μείωση διαστάσεων
'''

embedding = LocallyLinearEmbedding(n_components=3, n_neighbors=7)
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
Γραμμική μείωση των διαστάσεων με την εφαρμογή του SVD.
Σε αντίθεση με τον PCA δεν πραγματοποιείται center στα δεδομένα πριν
τον υπολογισμό του SVD
n_components-> αριθμός επιθυμητών διαστάσεων 
'''
svd = TruncatedSVD(n_components=2)
df1 = svd.fit_transform(X_scaled)
# Τα singular διανύσματα των δεδομέων μας
print(svd.components_)
#Η διακύμανση των singular vectors
print(svd.explained_variance_)

'''
Επιστρέφει ένα τυχαίο δείγμα γραμμών από το dataset.
Δέχεται σαν παράμετρο τον αριθμό των δειγμάτων που θα προκύψουν από δειγματοληψία χωρίς αντικατάσταση
'''
print("Δειγματοληψία χωρίς αντικατάσταση")
print(df_google.sample(800))
print("Συστηματική δειγματοληψία")
print(df_google.loc[0:2000:10])

#Stratified sampling
df_Crete = df_google[df_google["sub_region"] == "Decentralized Administration of Crete"]
print(df_Crete.shape)
print(df_Crete.sample(50))
print(df_google.groupby('sub_region', group_keys=False).apply(lambda x: x.sample(4)))
print(df_google.groupby('sub_region', group_keys=False).apply(lambda x: x.sample(frac=0.1)))
