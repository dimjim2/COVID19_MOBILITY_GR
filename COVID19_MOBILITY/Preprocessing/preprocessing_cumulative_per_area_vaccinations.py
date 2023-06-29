from Plots.plots import barplot_vaccinations_cumulative, make_boxplot
from utils.dataOperations import load_data_from_table

df_vaccinations = load_data_from_table("cumulative_per_area_vaccinations")
print(df_vaccinations)
barplot_vaccinations_cumulative(df_vaccinations, "Συνολικοί εμβολιασμοί ανά περιοχή")

print(df_vaccinations["totaldose1"])
print(df_vaccinations["totaldose2"])
print(df_vaccinations["totaldose3"])
print(df_vaccinations["totalvaccinations"])

make_boxplot(df_vaccinations, ['totaldose1', 'totaldose2', 'totaldose3', 'totalvaccinations'], "Αθροιστικός αριθμός εμβολιασμών")