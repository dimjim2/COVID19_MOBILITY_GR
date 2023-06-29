from datetime import datetime

from Plots.plots import make_bar_plot, make_plot, make_hist, make_boxplot
from utils.dataOperations import load_data_from_table

df_covid_cases_greece = load_data_from_table("covid_cases_greece")
print(df_covid_cases_greece)
print(df_covid_cases_greece.dtypes)

make_plot(df_covid_cases_greece, "total_cases", "Συνολικά κρούσματα Ελλάδα", "ημερομηνία", "αριθμός κρουσμάτων")
df_covid_daily = df_covid_cases_greece.loc[df_covid_cases_greece["date"].dt.date <= datetime.strptime('2022-07-10', "%Y-%m-%d").date()]
make_plot(df_covid_daily, "cases", "Πορεία ημερήσιων κρουσμάτων στην Ελλάδα", "ημερομηνία", "αριθμός μολύνσεων")

df_covid_cases_greece.set_index("date", inplace=True)
print(df_covid_cases_greece.dtypes)
# Μετασχηματίζει τα κρούσματα σε εβδομαδιαία μορφή πραγματοποιώντας το άθροισμα τους
df_covid_cases_greece_weekly = df_covid_cases_greece.resample('W').apply('sum')
df_covid_cases_greece_weekly.reset_index(inplace=True)

make_bar_plot(df_covid_cases_greece_weekly, "cases", "Εβδομαδιαία κρούσματα από COVID-19", "ημερομηνία", "αριθμός μολύνσεων")