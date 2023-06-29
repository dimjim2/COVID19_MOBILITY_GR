import pandas as pds
from sqlalchemy import create_engine

path = "postgresql://postgres:computer2000dj2@localhost:5432/diploma_thesis"

# Λαμβάνει τα δεδομένα από τον πίνακα της βάσης table_name
def load_data_from_table(table_name):
    # Δημιουργεί ένα αντικείμενο Engine και δημιουργεί σύνδεση με τη βάση μας
    alchemyEngine = create_engine(path)
    dbConnection = alchemyEngine.connect()
    # Επιστρέφει ένα dataframe το οποίο περιλαμβάνει τα δεδομένα του πίνακα
    data_frame = pds.read_sql_table(table_name, dbConnection)
    dbConnection.close()
    return data_frame

# Εισάγει τα δεδομένα ενός dataframe στον πίνακα της βάσης table_name
def insert_data_to_table(dataframe, table_name, primary_key):
    try:
        db = create_engine(path)
        conn = db.connect()
        # Γράφει τις εγγραφές του dataframe στον πίνακα της βάσης
        dataframe.to_sql(table_name, con=conn, if_exists='replace', index=False)
        # Προσθέτουμε το primary key
        conn.execute("ALTER TABLE {0} ADD PRIMARY KEY {1};".format(table_name, primary_key))
    except Exception as e:
        print(e)
