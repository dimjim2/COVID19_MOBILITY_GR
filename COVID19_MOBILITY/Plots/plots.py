import numpy as np
from matplotlib import pyplot as plt, pyplot
from matplotlib.dates import DateFormatter

# Δημιουργία γραφήματος για το χαρακτηριστικό feature
def make_plot(df, feature, message, label_xaxis, label_yaxis):
    plt.plot(df['date'], df[feature])
    plt.xlabel(label_xaxis)
    plt.ylabel(label_yaxis)
    plt.gcf().autofmt_xdate()
    plt.title(message)
    plt.show()

# Δημιουργία bar plot για το χαρακτηριστικό feature του df
def make_bar_plot(df, feature, message, label_xaxis, label_yaxis):

    fig, axs = plt.subplots()
    plt.suptitle(message)
    plt.bar(df["date"], df[feature])
    date_form = DateFormatter("%y-%m")
    plt.xlabel(label_xaxis)
    plt.ylabel(label_yaxis)
    axs.xaxis.set_major_formatter(date_form)
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)
    plt.show()

# Δημιουργία ενός γραφήματος με δύο subplots για τα χαρακτηριστικά feature1 και feature2
def make_two_subplots(df, feature1, feature2, message, title1, title2):
    # Φτιάχνουμε ένα figure με δύο subplots
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(message)
    ax1.plot(df["date"], df[feature1])
    ax1.set_title(title1)

    ax2.plot(df["date"], df[feature2], 'tab:orange')
    ax2.set_title(title2)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

# Δημιουργία barplot άνα περιοχή για το χαρακτηριστικό feature που εξελίσσεται στην ροή του χρόνου
def make_bar_plot_regions_by_date(df, feature, message):
    grouped = df.groupby('area_gr')
    fig, ax = plt.subplots(13, 4, figsize=(20, 20))
    plt.suptitle(message)
    fig.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.3,
                        hspace=1.5)

    for (key, ax) in zip(grouped.groups.keys(), ax.flatten()):
        print(key)
        ax.bar(grouped.get_group(key)['date'], grouped.get_group(key)[feature])
        ax.set_title(key)
    plt.show()

# Δημιουργία plot για τους εμβολιασμούς ανά περιοχή
def make_plot_vaccinations_by_region(df, message):
    grouped = df.groupby('area_gr')
    fig, ax = plt.subplots(13, 4,figsize=(30,30))
    plt.suptitle(message)
    fig.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)

    for (key, ax) in zip(grouped.groups.keys(), ax.flatten()):
        print(key)
        ax.plot(grouped.get_group(key)['date'], grouped.get_group(key)['totalvaccinations'])
        ax.plot(grouped.get_group(key)['date'], grouped.get_group(key)['totaldose1'])
        ax.plot(grouped.get_group(key)['date'], grouped.get_group(key)['totaldose2'])
        ax.plot(grouped.get_group(key)['date'], grouped.get_group(key)['totaldose3'])
        ax.set_title(key)
    # Προσθήκη legend
    ax.legend(["Συνολικοί εμβολιασμοί", "Συνολικοί εμβολιασμοί με τουλάχιστον 1 δόση",
               "Συνολικοί ολοκληρωμένοι εμβολιασμοί", "Συνολικοί εμβολιασμοί με την τρίτη  δόση"],
              loc='upper left', bbox_to_anchor=(0.8, 0))
    plt.show()
# Προβολή των δεικτών οδήγησης και περπατήματος σε δύο ξεχωριστά subplots για την Ελλάδα και την Αττική
def make_plot_apple_mobility_indexes(df_greece, message):
    # Φτιάχνουμε ένα figure με δύο subplots
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(message)

    dates = df_greece["date"].unique()
    ax1.plot(dates, df_greece.loc[df_greece["region"] == "Greece"]["driving"], c='r', label="οδήγηση")
    ax1.plot(dates, df_greece.loc[df_greece["region"] == "Greece"]["walking"], c='b', label="περπάτημα")
    ax1.set_title('Ελλάδα')

    ax2.plot(dates, df_greece.loc[df_greece["region"] == "Athens"]["driving"], c='r', label="οδήγηση")
    ax2.plot(dates, df_greece.loc[df_greece["region"] == "Athens"]["walking"], c='b', label="πεπάτημα")
    ax1.axhline(y=100, color='k', linestyle='-')
    ax2.axhline(y=100, color='k', linestyle='-')

    ax1.set_title('Ελλάδα')
    ax2.set_title("Αττική")
    plt.xlabel("Ημερομηνία")
    plt.ylabel("Μεταβολή κινητικότητας")

    plt.legend(loc='best')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()
# Δημιουργία boxplot για τα χαρακτηριστικά στη λίστα cols
def make_boxplot(df, cols, message):
    plt.figure()
    plt.boxplot(df[cols])
    cols = get_google_short_labels(cols)
    plt.xticks(list(range(1, len(cols)+1)), cols)
    plt.title(message)
    plt.show()

# Μετονομασία στηλών στην περίπτωση που το boxplot δημιουργείται για τους δείκτες Google
def get_google_short_labels(cols):
    if cols == ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
              'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
              'workplaces_percent_change_from_baseline','residential_percent_change_from_baseline']:
        cols = ['retail', 'grocery', 'park', 'transit', 'workplace', 'residential']
    print("----------------------------")
    print(cols)
    return cols

# Δημιουργία ιστογράμματος
def make_hist(df, cols, bins, message):
    # Κάνει ένα ιστόγραμμα για κάθε column που ανήκει στη λίστα cols
    df.hist(column=cols, alpha=0.5, bins=bins, figsize=(15, 10))
    # Βάζουμε έναν υπότιτλο κεντραρισμένο στο figure
    plt.suptitle(message)
    # Γίνεται προβολή του figure
    plt.show()

# Δημιουργούμε ένα scatter plot
def make_scatter_plot(df, col, message, label_xaxis, label_yaxis):
    plt.scatter(df["date"], df[col])
    plt.gcf().autofmt_xdate()
    plt.xlabel(label_xaxis)
    plt.ylabel(label_yaxis)
    # Θέτει για τον x άξονα κάθετη περιστροφή
    plt.xticks(rotation='vertical')

    plt.title(message)
    # Κάνουμε πριστροφή της ημερομηνίας και την κάνουμε στόιχιση προς τα δεξιά
    # To gcf δέχεται το τρέχον παράθυρο
    plt.gcf().autofmt_xdate()
    # Δίνει αυτόματα στα subplots συγκεκριμένο padding
    plt.tight_layout()
    plt.show()

# Δημιουργία barplot για τα αθροιστικά δεδομένα των κρουσμάτων
def barplot_regions_cumulative_covid_cases(df_covid_cases_by_area):
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))
    # Οριζόντιο Bar Plot
    ax.barh(df_covid_cases_by_area["area_gr"], df_covid_cases_by_area["total_cases"])
    # Αφαίρεση axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Αφαίρεση x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.ticklabel_format(axis="x", style='plain')
    # Προσθήκη padding ανάμεσα σε axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)
    ax.set_xlabel("Αριθμός κρουσμάτων")
    ax.invert_yaxis()
    # Προσθήκη αριθμού κρουσμάτων πάνω στις μπάρες
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')
    # Προσθήκη τίτλου
    ax.set_title('Συνολικά κρούσματα COVID-19 ανά περιοχή',
                 loc='center')
    plt.show()

# Δημιουργία Pie plot για συνολικά κρούσματα COVID-19 ανά αποκεντρωμένη διοίκηση της Ελλάδας
def pieplot_covid_regional(df_cumulative_covid_cases_by_area):
    df_cumulative_covid_cases_by_area['area_gr'].replace(['ΔΩΔΕΚΑΝΗΣΩΝ', 'ΚΥΚΛΑΔΩΝ', 'ΣΑΜΟΥ', 'ΛΕΣΒΟΥ', 'ΧΙΟΥ'],
                                             "Αποκεντρωμένη Διοίκηση Αιγαίου", inplace=True)

    df_cumulative_covid_cases_by_area['area_gr'].replace(
        ['ΙΩΑΝΝΙΝΩΝ', 'ΑΡΤΑΣ', 'ΘΕΣΠΡΩΤΙΑΣ', 'ΠΡΕΒΕΖΑΣ', 'ΓΡΕΒΕΝΩΝ', 'ΚΑΣΤΟΡΙΑΣ', 'ΚΟΖΑΝΗΣ', 'ΦΛΩΡΙΝΑΣ'],
        "Αποκεντρωμένη Διοίκηση Ηπείρου - Δυτικής Μακεδονίας", inplace=True)

    df_cumulative_covid_cases_by_area['area_gr'].replace(
        ['ΒΟΙΩΤΙΑΣ', 'ΕΥΒΟΙΑΣ', 'ΕΥΡΥΤΑΝΙΑΣ', 'ΤΡΙΚΑΛΩΝ', 'ΦΩΚΙΔΑΣ', 'ΦΘΙΩΤΙΔΑΣ', 'ΜΑΓΝΗΣΙΑΣ', 'ΛΑΡΙΣΑΣ', 'ΚΑΡΔΙΤΣΑΣ'],
        "Αποκεντρωμένη Διοίκηση Θεσσαλίας - Στερεάς Ελλάδας", inplace=True)

    df_cumulative_covid_cases_by_area['area_gr'].replace(
        ['ΕΒΡΟΥ', 'ΗΜΑΘΙΑΣ', 'ΘΕΣΣΑΛΟΝΙΚΗΣ', 'ΑΓΙΟ ΟΡΟΣ', 'ΧΑΛΚΙΔΙΚΗΣ', 'ΣΕΡΡΩΝ', 'ΡΟΔΟΠΗΣ', 'ΠΙΕΡΙΑΣ', 'ΠΕΛΛΑΣ',
         'ΞΑΝΘΗΣ', 'ΚΙΛΚΙΣ', 'ΚΑΒΑΛΑΣ', 'ΔΡΑΜΑΣ'], "Αποκεντρωμένη Διοίκηση Μακεδονίας - Θράκης", inplace=True)

    df_cumulative_covid_cases_by_area['area_gr'].replace(
        ['ΑΙΤΩΛΟΑΚΑΡΝΑΝΙΑΣ', 'ΑΡΓΟΛΙΔΑΣ', 'ΑΡΚΑΔΙΑΣ', 'ΑΧΑΪΑΣ', 'ΗΛΕΙΑΣ', 'ΖΑΚΥΝΘΟΥ', 'ΚΕΡΚΥΡΑΣ', 'ΚΕΦΑΛΛΟΝΙΑΣ',
         'ΚΟΡΙΝΘΟΥ', 'ΛΑΚΩΝΙΑΣ', 'ΜΕΣΣΗΝΙΑΣ', 'ΛΕΥΚΑΔΑΣ'],
        "Αποκεντρωμένη Διοίκηση Πελοποννήσου, Δυτικής Ελλάδας και Ιονίου", inplace=True)

    df_cumulative_covid_cases_by_area['area_gr'].replace(['ΗΡΑΚΛΕΙΟΥ', 'ΛΑΣΙΘΙΟΥ', 'ΡΕΘΥΜΝΟΥ', 'ΧΑΝΙΩΝ'],
                                             "Αποκεντρωμένη Διοίκηση Κρήτης", inplace=True)

    df_cumulative_covid_cases_by_area['area_gr'].replace('ΑΤΤΙΚΗΣ', "Αποκεντρωμένη Διοίκηση Αττικής", inplace=True)

    df_cumulative_covid_cases_by_area = df_cumulative_covid_cases_by_area.groupby(by=['area_gr']).sum()
    df_cumulative_covid_cases_by_area.sort_values(by=['area_gr'], inplace=True)
    df_cumulative_covid_cases_by_area.reset_index(inplace=True)
    print(df_cumulative_covid_cases_by_area)

    cumulative_cases = df_cumulative_covid_cases_by_area["total_cases"].sum()
    # Δημιουργία χρωμάτων για κάθε πίτα
    colors = ("orange", "cyan", "brown",
              "grey", "indigo", "beige", "blue")
    fig, ax = plt.subplots()

    # Ορίζει το κλάσμα της ακτίνας που μετατοπίζεται κάθε κoμμάτι της πίτας
    explode = (0.1, 0.0, 0.2, 0.5, 0.0, 0.0, 0.2)
    fig.set_figwidth(12)
    # Το startangle μετατοπίζει την πίτα κατά 90 μοίρες δεξιόστροφα από τον άξονα x
    wedges, texts, autotexts = ax.pie(df_cumulative_covid_cases_by_area["total_cases"],
                                      autopct=lambda pct: "{:.1f}%\n({:d})".format(pct,
                                                                                   int(pct / 100. * np.sum(cumulative_cases))),
                                      labels=df_cumulative_covid_cases_by_area["area_gr"],
                                      shadow=True,
                                      explode=explode,
                                      colors=colors,
                                      startangle=90)
    # Τοποθετεί στα κομμάτια ως label το ποσοστό των κρουσμάτων τους
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Συνολικά κρούσματα Covid-19 ανά αποκεντρωμένη διοίκηση")
    plt.show()

# Δημιουργία barplot για τα αθροιστικά δεδομένα των εμβολιασμών
def barplot_vaccinations_cumulative(df,message):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    plt.title(message)
    fig.tight_layout(pad=5.0)
    axs[0, 0].barh(df["area_gr"], df["totaldose1"])
    axs[0, 0].set_title('Συνολικές πρώτες δόσεις')
    axs[0, 1].barh(df["area_gr"], df["totaldose2"])
    axs[0, 1].set_title('Συνολικές δεύτερες δόσεις')
    axs[1, 0].barh(df["area_gr"], df["totaldose3"])
    axs[1, 0].set_title('Συνολικές τρίτες δόσεις')
    axs[1, 1].barh(df["area_gr"], df["totalvaccinations"])
    axs[1, 1].set_title('Συνολικές δόσεις')
    plt.gcf().autofmt_xdate()

    axs[0, 0].tick_params(axis='both', which='both', labelbottom=True)
    axs[0, 1].tick_params(axis='both', which='both', labelbottom=True)

    for s in ['top', 'bottom', 'left', 'right']:
        axs[0, 0].spines[s].set_visible(False)
        axs[1, 0].spines[s].set_visible(False)
        axs[0, 1].spines[s].set_visible(False)
        axs[1, 1].spines[s].set_visible(False)

    # Remove x, y Ticks
    axs[0, 1].xaxis.set_ticks_position('none')
    axs[0, 1].yaxis.set_ticks_position('none')
    axs[0, 0].xaxis.set_ticks_position('none')
    axs[0, 0].yaxis.set_ticks_position('none')
    axs[1, 0].xaxis.set_ticks_position('none')
    axs[1, 0].yaxis.set_ticks_position('none')
    axs[1, 1].xaxis.set_ticks_position('none')
    axs[1, 1].yaxis.set_ticks_position('none')

    axs[0, 0].ticklabel_format(axis="x", style='plain')
    axs[0, 1].ticklabel_format(axis="x", style='plain')
    axs[1, 0].ticklabel_format(axis="x", style='plain')
    axs[1, 1].ticklabel_format(axis="x", style='plain')
    axs[0, 0].set_yticklabels(df["area_gr"], fontsize=5)
    axs[0, 1].set_yticklabels(df["area_gr"], fontsize=5)
    axs[1, 0].set_yticklabels(df["area_gr"], fontsize=5)
    axs[1, 1].set_yticklabels(df["area_gr"], fontsize=5)

    # Add padding between axes and labels
    axs[0, 0].xaxis.set_tick_params(pad=5)
    axs[0, 0].yaxis.set_tick_params(pad=10)
    axs[0, 1].xaxis.set_tick_params(pad=5)
    axs[0, 1].yaxis.set_tick_params(pad=10)
    axs[1, 0].xaxis.set_tick_params(pad=5)
    axs[1, 0].yaxis.set_tick_params(pad=10)
    axs[1, 1].xaxis.set_tick_params(pad=5)
    axs[1, 1].yaxis.set_tick_params(pad=10)

    plt.show()

def make_bar_plot_age(df, df_men, df_women , metric_message):
    fig = plt.figure(figsize = (10, 5))
    total = df_men + df_women
    total_metric = df[total].values.sum()
    percentage = []
    for i in (total):
        pct = (df[i].values[0] / total_metric) * 100
        percentage.append(round(pct, 2))

    # display percentage
    print(percentage)
    # creating the bar plot
    X = ["0-17", "18-39", "40-64", "65+"]
    X_axis = np.arange(len(X))

    graph_men = plt.bar(X_axis - 0.2, df[df_men].values.flatten().tolist(), color='b',
                  width=0.4, label="men")
    graph_women = plt.bar(X_axis + 0.2, df[df_women].values.flatten().tolist(), color='tab:pink',
                     width=0.4, label="women")
    i = 0
    graphs = [graph_men, graph_women]

    for gr in graphs:
        for p in gr:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()

            plt.text(x+width / 2,
                     y+height * 1.01,
                     str(percentage[i]) + '%',
                     ha='center',
                     weight='bold')
            i += 1

    plt.xticks(X_axis, X)
    plt.xlabel("Ηλικίες")
    plt.ylabel(metric_message + "ανά ηλικιακή κατηγορία")
    plt.title(metric_message + "και ποσοστά αυτών με βάση το φύλο και την ηλικιακή κατανομή μέχρι "+df["date"].to_string(index=False))
    plt.legend(["Άνδρες", "Γυναίκες"])
    plt.show()

# Δημιουργεί για το χαρακτηριστικό feature γραφήματα ανάλογα το φύλο και το ηλικιακό γκρουπ
def make_age_plot(df, feature, title, message):
    fig, axs = plt.subplots(4, 2, figsize=(20,20))
    plt.suptitle(title)
    fig.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.6)

    axs[0, 0].plot(df["date"], df["men_"+feature+"_0-17"])
    axs[0, 0].set_title('Άνδρες σε ηλικίες 0-17 ' + message)
    axs[0, 1].plot(df["date"], df["men_"+feature+"_18-39"], c="r")
    axs[0, 1].set_title('Άνδρες σε ηλικίες 18-39 ' + message)
    axs[1, 0].plot(df["date"], df["men_"+feature+"_40-64"], c="g")
    axs[1, 0].set_title('Άνδρες σε ηλικίες 40-64 ' + message)
    axs[1, 1].plot(df["date"], df["men_"+feature+"_65+"], c="m")
    axs[1, 1].set_title('Άνδρες σε ηλικίες 65+ ' + message)
    axs[2, 0].plot(df["date"], df["women_"+feature+"_0-17"], c="k")
    axs[2, 0].set_title('Γυναίκες σε ηλικίες 0-17 ' + message)
    axs[2, 1].plot(df["date"], df["women_"+feature+"_18-39"], c="y")
    axs[2, 1].set_title('Γυναίκες σε ηλικίες 18-39 ' + message)
    axs[3, 0].plot(df["date"], df["women_"+feature+"_40-64"], c="k")
    axs[3, 0].set_title('Γυναίκες σε ηλικίες 40-64 ' + message)
    axs[3, 1].plot(df["date"], df["women_"+feature+"_65+"], c="y")
    axs[3, 1].set_title('Γυναίκες σε ηλικίες 65+ ' + message)
    plt.show()

# Δημιουργεί scatter plot για την κάθε κατηγορία κινητικότητας Google
def make_google_scatter_plot(df, message):
    # Δημιουργία scatter plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία

    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()

    axs[0, 0].scatter(df["date"], df["retail_and_recreation_percent_change_from_baseline"])
    axs[0, 0].set_title('Retail and recreation')
    axs[0, 1].scatter(df["date"], df["grocery_and_pharmacy_percent_change_from_baseline"], c="r")
    axs[0, 1].set_title('Grocery and pharmacy')
    axs[1, 0].scatter(df["date"], df["parks_percent_change_from_baseline"], c="g")
    axs[1, 0].set_title('Parks')
    axs[1, 1].scatter(df["date"], df["transit_stations_percent_change_from_baseline"], c="m")
    axs[1, 1].set_title('Transit stations')
    axs[2, 0].scatter(df["date"], df["workplaces_percent_change_from_baseline"], c="k")
    axs[2, 0].set_title('Workplaces')
    axs[2, 1].scatter(df["date"], df["residential_percent_change_from_baseline"], c="y")
    axs[2, 1].set_title('Residential')

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

# Δημιουργεί scatter plot για την κάθε κατηγορία κινητικότητας Google
def make_google_plot(df,message):
    # Δημιουργία plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()

    axs[0, 0].plot(df["date"], df["retail_and_recreation_percent_change_from_baseline"])
    axs[0, 0].set_title('Retail and recreation')
    axs[0, 1].plot(df["date"], df["grocery_and_pharmacy_percent_change_from_baseline"], 'tab:orange')
    axs[0, 1].set_title('Grocery and pharmacy')
    axs[1, 0].plot(df["date"], df["parks_percent_change_from_baseline"], 'tab:green')
    axs[1, 0].set_title('Parks')
    axs[1, 1].plot(df["date"], df["transit_stations_percent_change_from_baseline"], 'tab:red')
    axs[1, 1].set_title('Transit stations')
    axs[2, 0].plot(df["date"], df["workplaces_percent_change_from_baseline"], 'tab:grey')
    axs[2, 0].set_title('Workplaces')
    axs[2, 1].plot(df["date"], df["residential_percent_change_from_baseline"], 'tab:purple')
    axs[2, 1].set_title('Residential')

    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()

# Χαράζει την κάθε κατηγορία κινητικότητας Google με ξεχωριστό χρώμα στο ίδιο γράφημα
def make_google_plot_in_the_same_axes(df, message):
    # Δημιουργία plot για κάθε χαρακτηριστικό όπου στο x άξονα έχουμε την ημερομηνία
    plt.plot(df["date"], df["retail_and_recreation_percent_change_from_baseline"], label='Λιανική & ψυχαγωγία')
    plt.plot(df["date"], df["grocery_and_pharmacy_percent_change_from_baseline"], 'tab:orange', label='Αγορές τροφ. & φαρμακεία')
    plt.plot(df["date"], df["parks_percent_change_from_baseline"], 'tab:green', label='Πάρκα')
    plt.plot(df["date"], df["transit_stations_percent_change_from_baseline"], 'tab:red', label='Δημόσιες συγκοινωνίες')
    plt.plot(df["date"], df["workplaces_percent_change_from_baseline"], 'tab:grey', label='Χώροι εργασίας')
    plt.plot(df["date"], df["residential_percent_change_from_baseline"], 'tab:purple', label='Κατοικίες')


    plt.xlabel("Ημερομηνία")
    plt.ylabel("Μεταβολή κινητικότητας ανά κατηγορία")
    plt.axhline(y=0.0, color='k', linestyle='-')
    plt.title(message)
    plt.legend(loc='best')
    plt.gcf().autofmt_xdate()
    pyplot.show()

# Ανά κατηγορία Google χαράσσεται η κινητικότητα που παρατηρήθηκε ανά περιοχή στο ίδιο subplot με ξεχωριστό χρώμα
def make_google_plot_by_activity_regions(df, message):
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    pyplot.suptitle(message)
    fig.tight_layout()
    # Αποκεντρωμένες διοικήσεις
    regions = ["Decentralized Administration of Attica", "Decentralized Administration of the Aegean",
               "Decentralized Administration of Macedonia and Thrace",
               "Decentralized Administration of Peloponnese, Western Greece and the Ionian",
               "Decentralized Administration of Thessaly and Central Greece", "Decentralized Administration of Crete"]
    regions_colors_cols = ['k', 'b', 'r', 'm', 'y', 'g', 'c']
    dates = df["date"].unique()

    for (region_name, region_color) in zip(regions, regions_colors_cols):

        axs[0, 0].plot(dates, df.loc[df["sub_region"] == region_name]["retail_and_recreation_percent_change_from_baseline"],
                       c=region_color, label=region_name)
        axs[0, 0].plot(dates, df.loc[df["sub_region"] == region_name]["retail_and_recreation_percent_change_from_baseline"],
                       c=region_color, label=region_name)
        axs[0, 1].plot(dates, df.loc[df["sub_region"] == region_name]["grocery_and_pharmacy_percent_change_from_baseline"],
                       c=region_color, label=region_name)
        axs[1, 0].plot(dates, df.loc[df["sub_region"] == region_name]["parks_percent_change_from_baseline"],
                       c=region_color, label=region_name)


        axs[1, 1].plot(dates, df.loc[df["sub_region"] == region_name]["transit_stations_percent_change_from_baseline"],
                       c=region_color, label=region_name)
        axs[2, 0].plot(dates, df.loc[df["sub_region"] == region_name]["workplaces_percent_change_from_baseline"],
                       c=region_color, label=region_name)
        axs[2, 1].plot(dates, df.loc[df["sub_region"] == region_name]["residential_percent_change_from_baseline"],
                       c=region_color, label=region_name)

    plt.gcf().autofmt_xdate()
    axs[0, 0].set_title('Λιανική & ψυχαγωγία')
    axs[0, 0].set_ylabel("Μεταβολή κινητικότητας")
    axs[0, 1].set_title('Αγορές τροφ. & φαρμακεία')
    axs[0, 1].set_ylabel("Μεταβολή κινητικότητας")
    axs[1, 0].set_title('Πάρκα')
    axs[1, 0].set_ylabel("Μεταβολή κινητικότητας")
    axs[1, 1].set_title('Δημόσιες συγκοινωνίες')
    axs[1, 1].set_ylabel("Μεταβολή κινητικότητας")

    axs[2, 0].set_title('Χώροι εργασίας')
    axs[2, 0].set_xlabel("Ημερομηνία")
    axs[2, 0].set_ylabel("Μεταβολή κινητικότητας")
    axs[2, 1].set_title('Κατοικίες')
    axs[2, 1].set_xlabel("Ημερομηνία")
    axs[2, 1].set_ylabel("Μεταβολή κινητικότητας")

    # Προσθήκη legend που περιγράφει το χρώμα που ανατέθηκε σε κάθε αποκεντρωμένη διοίκηση
    plt.legend(loc='upper center', bbox_to_anchor=(0, -0.25),
               shadow=True, ncol=3)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    pyplot.show()