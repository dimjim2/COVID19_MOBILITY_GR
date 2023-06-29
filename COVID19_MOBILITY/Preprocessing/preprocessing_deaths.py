from Plots.plots import make_two_subplots, make_bar_plot
from utils.dataOperations import load_data_from_table


df_deaths = load_data_from_table("deaths")
print(df_deaths)
make_two_subplots(df_deaths, "deaths_cum", "deaths", 'Χρονοσειρές θανάτων από Covid-19', 'Πορεία αθροιστικών θανάτων',
                  'Πορεία ημερίσιων θανάτων')


df_deaths.set_index("date", inplace=True)
print(df_deaths.dtypes)

df_deaths_weekly = df_deaths.resample('W').apply('sum')
df_deaths_weekly.reset_index(inplace=True)

make_bar_plot(df_deaths_weekly, "deaths", "Εβδομαδιαίοι θάνατοι από COVID-19", "ημερομηνία", "θάνατοι")