import pandas as pd

from Plots.plots import make_plot, make_hist, make_scatter_plot
from utils.dataOperations import load_data_from_table

df_intensive_cares = load_data_from_table("intensive_care_cases")
df_intensive_cares["date"] = pd.to_datetime(df_intensive_cares["date"])
print(df_intensive_cares)
print(df_intensive_cares.dtypes)

# Εκτυπώνουμε τις πρώτες 5 γραμμές
print(df_intensive_cares.head())
# Εκτυπώνουμε τις στήλες
print(df_intensive_cares.columns)

make_plot(df_intensive_cares, "intensive_care_patients", "Μονάδες εντατικής θεραπείας", "ημερομηνία", "αριθμός ασθενών")
make_scatter_plot(df_intensive_cares, "intensive_care_patients", "Ασθενείς σε ΜΕΘ", "ημερομηνία", "αριθμός ασθενών")