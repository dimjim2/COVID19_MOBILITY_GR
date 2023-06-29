from Plots.plots import make_bar_plot_age, make_age_plot
from utils.dataOperations import load_data_from_table

df_age_distribution = load_data_from_table("age_distribution")
# Λαμβάνει δεδομένα μέχρι και την ημερομηνία 2022-07-10
df_age_distribution = df_age_distribution.query("date <= '2022-07-10'")
print(df_age_distribution)
print(df_age_distribution.columns)

cases_men = ['men_cases_0-17', 'men_cases_18-39', 'men_cases_40-64', 'men_cases_65+']
cases_women = ['women_cases_0-17', 'women_cases_18-39', 'women_cases_40-64', 'women_cases_65+']
deaths_men = ['men_deaths_0-17', 'men_deaths_18-39', 'men_deaths_40-64', 'men_deaths_65+']
deaths_women = ['women_deaths_0-17', 'women_deaths_18-39', 'women_deaths_40-64', 'women_deaths_65+']

df = df_age_distribution.tail(1)
make_bar_plot_age(df, cases_men, cases_women, "Συνολικά κρούσματα COVID-19")
make_bar_plot_age(df, deaths_men, deaths_women, "Συνολικοί θάνατοι COVID-19")
make_age_plot(df_age_distribution, "intensive", "Αριθμός ασθενών από COVID-19 που βρίσκονται διασωληνωμένοι σε ΜΕΘ", "που βρίσκονται διασωληνωμένοι σε ΜΕΘ")

