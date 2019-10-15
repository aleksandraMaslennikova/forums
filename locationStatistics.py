import re

dictRipartizioneGeografica = {}
dictRegioni = {}
with open("data/final_users_location_user_id.txt", encoding="utf8") as f:
    line = f.readline()
    while line:
        if line.startswith("<doc"):
            ripartizioneGeografica = re.findall('task_ripartizione_geografica=".+?"', line)[0][30:-1]
            if ripartizioneGeografica not in dictRipartizioneGeografica.keys():
                dictRipartizioneGeografica[ripartizioneGeografica] = 1
            else:
                dictRipartizioneGeografica[ripartizioneGeografica] += 1
            regione = re.findall('task_regione=".+?"', line)[0][14:-1]
            if regione not in dictRegioni.keys():
                dictRegioni[regione] = 1
            else:
                dictRegioni[regione] += 1
        line = f.readline()

print("Ripartizione geografica (numero documenti):")
for ripartizioneGeografica in dictRipartizioneGeografica:
    print(ripartizioneGeografica + " -> " + str(dictRipartizioneGeografica[ripartizioneGeografica]))
print()
print("Regioni (numero documenti):")
for regione in dictRegioni:
    print(regione + " -> " + str(dictRegioni[regione]))