import pandas as pd
from Plots.plots import make_bar_plot_regions_by_date
from utils.dataOperations import load_data_from_table

df_history_cases = load_data_from_table("covid_by_area_history_cases")
df_history_cases['cases'].iloc[52:] = df_history_cases.groupby('area_gr')['cases'].diff().iloc[52:]
print(df_history_cases)
make_bar_plot_regions_by_date(df_history_cases, "cases", "Εβδομαδιαία κρούσματα COVID-19 ανά νομό")