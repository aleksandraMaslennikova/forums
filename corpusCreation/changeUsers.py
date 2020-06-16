import re

dictUsernames = {}
num = 1
with open("data/final_users_location.txt", encoding="utf8") as f:
    line = f.readline()
    while line:
        if line.startswith("<doc"):
            username = re.findall('user=".+?"', line)[0][6:-1]
            if username not in dictUsernames.keys():
                dictUsernames[username] = num
                num += 1
        line = f.readline()

with open("data/final_users_location.txt", encoding="utf8") as f:
    with open("../data/final_users_location_user_id.txt", "w", encoding="utf8") as outWithLocationNoUsername:
        line = f.readline()
        while line:
            if line.startswith("<doc"):
                username = re.findall('user=".+?"', line)[0][6:-1]
                line = re.sub('user=".+?"', 'user_id="' + str(dictUsernames[username]) + '"', line)
            outWithLocationNoUsername.write(line)
            line = f.readline()

with open("data/user_id_dictionary.txt", "w", encoding="utf8") as out:
    for username in dictUsernames:
        out.write(str(dictUsernames[username]) + "\t" + username + "\n")