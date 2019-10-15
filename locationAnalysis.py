from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
from xml.dom import minidom
import re

regioni = {"Piemonte",
           "Valle d'Aosta/Vallée d'Aoste",
           "Lombardia",
           "Trentino-Alto Adige/Südtirol",
           "Veneto",
           "Friuli-Venezia Giulia",
           "Liguria",
           "Emilia-Romagna",
           "Toscana",
           "Umbria",
           "Marche",
           "Lazio",
           "Abruzzo",
           "Molise",
           "Campania",
           "Puglia",
           "Basilicata",
           "Calabria",
           "Sicilia",
           "Sardegna"}
nomiSconosciuti = set()

df = pd.read_excel('data/Elenco-comuni-italiani.xls')
# print(df['Denominazione in italiano'])
with open('data/final_users.txt', encoding="utf8") as f:
    with open("data/final_users_location.txt", "w", encoding="utf8") as outWithLocation:
        with open("data/final_users_no_location.txt", "w", encoding="utf8") as outNoLocation:
            line = f.readline()
            numGuessed = 0
            numNotGuessed = 0
            haveLocation = False
            addToXML = ""
            while line:
                if line.startswith("<doc"):
                    addToXML = ""
                    # [15:-2] to delete task_location from the start and "> at the end
                    location = re.findall('task_location=".+">', line)[0][15:-2].title()
                    location = re.sub("&#[0-9]+", "", location)
                    location = re.sub("Italy", "", location)
                    location = re.sub("Italia", "", location)
                    for reg in regioni:
                        if re.search(reg, location):
                            location = reg
                    if location.endswith(" ") or location.endswith(".") or location.endswith(";") or location.endswith(","):
                        location = location[:-1]
                    if re.search("Provincia Di [A-Za-z]+", location):
                        location = re.findall("Provincia Di [A-Za-z]+", location)[0][13:]
                    if re.search("Provincia Di  [A-Za-z]+", location):
                        location = re.findall("Provincia Di  [A-Za-z]+", location)[0][14:]
                    if re.search("Prov. Di [A-Za-z]+", location):
                        location = re.findall("Prov. Di [A-Za-z]+", location)[0][9:]
                    if re.search("[A-Za-z]+ Prov", location):
                        location = re.findall("[A-Za-z]+ Prov", location)[0][:-5]
                    location = re.split(" ?/ ?", location)[0]
                    location = re.split(" ?- ?", location)[0]
                    if re.search("\([A-Za-z][A-Za-z]\)", location):
                        location = re.findall("\([A-Za-z][A-Za-z]\)", location)[0][1:-1].upper()
                    if len(location) == 2:
                        location = location.upper()
                    if location == "Alto Adige" or location == "Trentino" or location == "Trentino-Alto Adige":
                        location = "Trentino-Alto Adige/Südtirol"
                    if location == "Valle D'Aosta":
                        location = "Valle d'Aosta/Vallée d'Aoste"
                    if location == "Friul":
                        location = "Friuli-Venezia Giulia"
                    if re.search("[^a-zA-Z][A-Za-z][A-Za-z]$", location):
                        location = re.findall("[^a-zA-Z][A-Za-z][A-Za-z]$", location)[0][1:].upper()
                    if (df.loc[df['Denominazione in italiano'] == location]).empty:
                        if (df.loc[df['Denominazione regione'] == location]).empty:
                            if (df.loc[df['Sigla automobilistica'] == location]).empty:
                                if re.search("Rome", location) or re.search("Roma", location) or re.search("Osti", location)\
                                        or re.search("Tevere", location) or re.search("Urbs Aeterna", location) or re.search("Caput Mundi", location):
                                    location = "Roma"
                                if re.search("Milano", location) or re.search("Milan", location) or re.search("Lumbardia", location)\
                                        or re.search("San Siro", location) or re.search("Brianza", location) or re.search("Martesana", location)\
                                        or re.search("S.Stefano Ticino", location):
                                    location = "Milano"
                                if re.search("Firenze", location) or re.search("Florence", location) or re.search("Toscan", location)\
                                        or re.search("Florentia", location) or re.search("Mugello", location) or re.search("Curva Fiesole", location):
                                    location = "Firenze"
                                if re.search("Napoli", location) or re.search("Naples", location) or re.search("Napoletan",location)\
                                        or re.search("Torre Del Greco", location) or re.search("Stadio San Paolo", location):
                                    location = "Napoli"
                                if re.search("Bologn", location) or re.search("Reggio Emilia", location):
                                    location = "Bologna"
                                if re.search("Monferrato Casalese", location):
                                    location = "Casale Monferrato"
                                if re.search("Inverigo", location):
                                    location = "Inverigo"
                                if re.search("San Sperate", location):
                                    location = "San Sperate"
                                if re.search("Falicetto", location):
                                    location = "Verzuolo"
                                if re.search("Lurago Marinone", location):
                                    location = "Lurago Marinone"
                                if re.search("Valsesia", location):
                                    location = "Vercelli"
                                if re.search("Lodi", location):
                                    location = "Lodi"
                                if re.search("Berghem", location) or re.search("Cortenuova", location) or re.search("Seriate", location)\
                                        or re.search("Bassa Bergamasca", location):
                                    location = "Bergamo"
                                if re.search("Siena", location):
                                    location = "Siena"
                                if re.search("Marchigiane", location):
                                    location = "Ancona"
                                if re.search("Monza", location):
                                    location = "Monza"
                                if re.search("Isernia", location):
                                    location = "Isernia"
                                if re.search("Mondragone", location):
                                    location = "Mondragone"
                                if re.search("San Giorgio A Cremano", location):
                                    location = "San Giorgio a Cremano"
                                if re.search("Pisa", location) or re.search("Versilia", location):
                                    location = "Pisa"
                                if re.search("Isola Di Montecristo", location):
                                    location = "Portoferraio"
                                if re.search("Pistoia", location):
                                    location = "Pistoia"
                                if re.search("Monferrato", location):
                                    location = "Asti"
                                if re.search("Lucania", location) or re.search("Potenza", location):
                                    location = "Potenza"
                                if re.search("Bassano Del Grappa", location):
                                    location = "Bassano del Grappa"
                                if re.search("Cremona", location):
                                    location = "Cremona"
                                if re.search("Longone Al Segrino", location):
                                    location = "Como"
                                if re.search("Messina", location):
                                    location = "Messina"
                                if re.search("Cervia", location):
                                    location = "Cervia"
                                if re.search("Auronzo Di Cadore", location):
                                    location = "Belluno"
                                if re.search("Busto Arsizio", location):
                                    location = "Busto Arsizio"
                                if re.search("Terni", location):
                                    location = "Terni"
                                if re.search("San Benedetto Del Tronto", location):
                                    location = "Ascoli Piceno"
                                if re.search("Livorno", location):
                                    location = "Livorno"
                                if re.search("Bergamo", location):
                                    location = "Bergamo"
                                if re.search("Novara", location):
                                    location = "Novara"
                                if re.search("Kroton", location):
                                    location = "Crotone"
                                if re.search("Besenum", location):
                                    location = "Besenello"
                                if re.search("Vicentino", location):
                                    location = "Vicenza"
                                if re.search("Ziznatik", location):
                                    location = "Cesenatico"
                                if re.search("Valle D'Itria", location):
                                    location = "Bari"
                                if re.search("Torino", location) or re.search("Torinese", location) or re.search("Turin", location)\
                                        or re.search("Juventus", location) or re.search("Augusta Taurinorum", location)\
                                        or re.search("Collegno", location):
                                    location = "Torino"
                                if re.search("Ferrara", location):
                                    location = "Ferrara"
                                if re.search("Montese", location):
                                    location = "Modena"
                                if re.search("Catanisi", location):
                                    location = "Catania"
                                if re.search("Veron", location):
                                    location = "Verona"
                                if re.search("Vares", location):
                                    location = "Verona"
                                if re.search("Pordenone", location):
                                    location = "Pordenone"
                                if re.search("Arezzo", location) or re.search("Arretium", location):
                                    location = "Arezzo"
                                if re.search("Canosa", location):
                                    location = "Canosa di Puglia"
                                if re.search("Castelfranco Di Sotto", location):
                                    location = "Castelfranco di Sotto"
                                if re.search("Sant'Ilario D'Enza", location):
                                    location = "Sant'Ilario d'Enza"
                                if re.search("Gragnano Trebbiense", location):
                                    location = "Gragnano Trebbiense"
                                if re.search("Cento", location):
                                    location = "Cento"
                                if re.search("Porto Viro", location):
                                    location = "Porto Viro"
                                if re.search("Cala Gonone", location):
                                    location = "Dorgali"
                                if re.search("Genova", location) or re.search("Palmaro", location) or re.search("Ciavai", location)\
                                        or re.search("Zena", location) or re.search("Alta Valle D'Orba", location):
                                    location = "Genova"
                                if re.search("Sicily", location) or re.search("Panhormus", location) or re.search("Sicula", location)\
                                        or re.search("Trinacria", location):
                                    location = "Palermo"
                                if re.search("Isola Di La Maddalena", location):
                                    location = "Sassari"
                                if re.search("Verbania", location):
                                    location = "Verbania"
                                if re.search("Ficcardum", location):
                                    location = "San Ginesio"
                                if re.search("Quiliano", location):
                                    location = "Quiliano"
                                if re.search("Follonica", location):
                                    location = "Follonica"
                                if re.search("Martesana", location):
                                    location = "Martesana"
                                if re.search("Cuneo", location):
                                    location = "Cuneo"
                                if re.search("Brescia", location):
                                    location = "Brescia"
                                if re.search("Vinovo", location):
                                    location = "Vinovo"
                                if re.search("Forl", location):
                                    location = "Forlì"
                                if re.search("Bari", location):
                                    location = "Bari"
                                if re.search("Gioia Tauro", location):
                                    location = "Gioia Tauro"
                                if re.search("Udine", location):
                                    location = "Udine"
                                if re.search("Modena", location):
                                    location = "Modena"
                                if re.search("Salerno", location):
                                    location = "Salerno"
                                if re.search("Carrara", location):
                                    location = "Carrara"
                                if re.search("Treviso", location) or re.search("Tarvisium", location):
                                    location = "Treviso"
                                if re.search("Valstrona", location):
                                    location = "Valstrona"
                                if re.search("Prascorsano", location):
                                    location = "Prascorsano"
                                if re.search("Cosenza", location):
                                    location = "Cosenza"
                                if re.search("Cagliari", location) or re.search("Sardinia", location):
                                    location = "Cagliari"
                                if re.search("Padova", location) or re.search("padov", location):
                                    location = "Padova"
                                if re.search("Garda", location):
                                    location = "Verona"
                                if re.search("Caserta", location) or re.search("Capua", location):
                                    location = "Caserta"
                                if re.search("Salent", location):
                                    location = "Salento"
                                if re.search("Venice", location) or re.search("Venezia", location) or re.search("Piave", location)\
                                        or re.search("Mestre", location):
                                    location = "Venezia"
                                if re.search("Rimini", location):
                                    location = "Rimini"
                                if re.search("Piaseinsa", location):
                                    location = "Piacenza"
                                if re.search("Benevento", location):
                                    location = "Benevento"
                                if re.search("Druent", location):
                                    location = "Druento"
                                if re.search("Castellammare Di Stabia", location):
                                    location = "Castellammare di Stabia"
                                if re.search("Massa", location):
                                    location = "Massa"
                                if re.search("Somaglia", location):
                                    location = "Somaglia"
                                if re.search("San Cesareo", location):
                                    location = "San Cesareo"
                                if re.search("Invorio", location):
                                    location = "Invorio"
                                if re.search("Calcata", location):
                                    location = "Calcata"
                                if re.search("Pesaro", location):
                                    location = "Pesaro"
                                if re.search("Lipari", location):
                                    location = "Lipari"
                                if re.search("Castromediano", location):
                                    location = "Lecce"
                                if re.search("Lecco", location):
                                    location = "Lecco"
                                if re.search("Montecatini Terme", location):
                                    location = "Montecatini-Terme"
                                if re.search("Tergeste", location) or re.search("Friul", location) or re.search("Trieste", location):
                                    location = "Trieste"
                                if re.search("Pav", location):
                                    location = "Pavia"
                                if re.search("Courmayeur", location):
                                    location = "Courmayeur"
                                if re.search("Riviera Del Conero", location):
                                    location = "Ancora"
                                if re.search("Bibbiano", location):
                                    location = "Bibbiano"
                                if re.search("Fara In Sabina", location):
                                    location = "Fara in Sabina"
                                if re.search("Pomigliano D'Arco", location):
                                    location = "Pomigliano d'Arco"
                                if re.search("Pieve A Nievole", location):
                                    location = "Pieve a Nievole"
                                if re.search("Airola", location):
                                    location = "Airola"
                                if re.search("Welschtirol", location) or re.search("Trentino", location):
                                    location = "Trento"
                                if re.search("Pescara", location):
                                    location = "Pescara"
                                if re.search("Abruzz", location) or re.search("L' Aquila", location) or re.search("Colline Teatine", location):
                                    location = "L'Aquila"
                                if re.search("San Martino Di Castrozza", location):
                                    location = "Primiero San Martino di Castrozza"
                                if re.search("Sandigliano", location):
                                    location = "Sandigliano"
                                if re.search("Sale Marasino", location):
                                    location = "Sale Marasino"
                                if re.search("Vaprio D'Adda", location):
                                    location = "Vaprio d'Adda"
                                if (df.loc[df['Denominazione in italiano'] == location]).empty:
                                    numNotGuessed += 1
                                    nomiSconosciuti.add(location)
                                else:
                                    numGuessed += 1
                                    row = df.loc[df['Denominazione in italiano'] == location]
                                    addToXML = ' task_regione="' + row['Denominazione regione'].values[
                                        0] + '" task_ripartizione_geografica="' + row['Ripartizione geografica'].values[0] + '"'
                            else:
                                numGuessed += 1
                                row = df.loc[df['Sigla automobilistica'] == location]
                                addToXML = ' task_regione="' + row['Denominazione regione'].values[
                                    0] + '" task_ripartizione_geografica="' + row['Ripartizione geografica'].values[0] + '"'
                        else:
                            numGuessed += 1
                            row = df.loc[df['Denominazione regione'] == location]
                            addToXML = ' task_regione="' + row['Denominazione regione'].values[
                                0] + '" task_ripartizione_geografica="' + row['Ripartizione geografica'].values[0] + '"'
                    else:
                        numGuessed += 1
                        row = df.loc[df['Denominazione in italiano'] == location]
                        addToXML = ' task_regione="' + row['Denominazione regione'].values[
                            0] + '" task_ripartizione_geografica="' + row['Ripartizione geografica'].values[0] + '"'
                    if addToXML == "":
                        haveLocation = False
                    else:
                        haveLocation = True
                if haveLocation:
                    if line.startswith("<doc"):
                        line = line[:-1] + addToXML + ">"
                    outWithLocation.write(line)
                else:
                    outNoLocation.write(line)
                line = f.readline()

    with open("data/paesiSconosciuti.txt", "w", encoding="utf8") as out:
        for n in nomiSconosciuti:
            out.write(n + "\n")

    print("Paese trovata: " + str(numGuessed))
    print("Paese non trovata: " + str(numNotGuessed))
    print("Nomi sconosciuti: " + str(len(nomiSconosciuti)))
