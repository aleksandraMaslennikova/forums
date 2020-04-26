dict_forum_thematic = {}
with open("data/ForumsWithThematics.csv", encoding="utf8") as f:
    line = f.readline()
    while line:
        line_arr = line.split(";")
        forum = line_arr[0]
        thematic = line_arr[1][:-1]
        dict_forum_thematic[forum] = thematic
        line = f.readline()

with open("data/final_users_location_user_id.txt", encoding="utf8") as fin:
    with open("data/final_corpus.txt", "w", encoding="utf8") as fout:
        line = fin.readline()
        while line:
            if line.startswith("<doc"):
                forum = line.split(" ")[2].split('"')[1]
                thematic = dict_forum_thematic[forum]
                line = line[:-2] + ' forums_thematic="' + thematic + '">\n'
            fout.write(line)
            line = fin.readline()