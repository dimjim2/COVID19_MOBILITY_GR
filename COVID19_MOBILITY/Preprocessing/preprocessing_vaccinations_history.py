from Plots.plots import make_bar_plot_regions_by_date, make_plot_vaccinations_by_region, make_boxplot
from utils.dataOperations import load_data_from_table

df_vaccinations = load_data_from_table("vaccinations_data_history_per_area")
print(df_vaccinations)
print(df_vaccinations.groupby('date')["totalvaccinations"].sum())

make_plot_vaccinations_by_region(df_vaccinations, "Πορεία εμβολιασμών που πραγματοποιήθηκαν ανά νομό")
make_bar_plot_regions_by_date(df_vaccinations, 'dailyvaccinations', "Ημερήσιος αριθμός εμβολιασμών "
                                                                    "που πραγματοποιήθηκαν ανά νομό")