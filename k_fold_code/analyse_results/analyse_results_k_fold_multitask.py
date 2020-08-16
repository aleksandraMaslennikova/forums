import re
import numpy as np

max_length = 1000
num_neurons = 100
task = "multi-task"
word_embedding = "itwac"
fileName = "results_" + task + "_k-fold_" + word_embedding + "_max_length_" + str(max_length) + "_num_neurons_" + str(num_neurons) + ".txt"
path = "../../results/"

arr_results = []
n = -1
with open(path + fileName, 'r') as f:
    line = f.readline()
    while line:
        if re.match("^[A-Z]", line) and not re.match("Watches", line) and not re.match("Rest of the corpus", line):
            n += 1
            arr_results.append({})
            arr_results[n]["name"] = line[:-1]
            arr_results[n]["training_arr_age"] = []
            arr_results[n]["validation_arr_age"] = []
            arr_results[n]["training_arr_country_part"] = []
            arr_results[n]["validation_arr_country_part"] = []
        elif re.match("^\t\tRESULTS", line):
            line = f.readline()
            arr_results[n]["training_arr_age"].append(float(line.split("Train:")[1].split("%,")[0]))
            arr_results[n]["validation_arr_age"].append(float(line.split("Validation:")[1].split("%")[0]))
            line = f.readline()
            arr_results[n]["training_arr_country_part"].append(float(line.split("Train:")[1].split("%,")[0]))
            arr_results[n]["validation_arr_country_part"].append(float(line.split("Validation:")[1].split("%")[0]))
        line = f.readline()

for result in arr_results:
    print(result["name"])
    print("age")
    print("training")
    print(result["training_arr_age"])
    print(str(round(np.mean(result["training_arr_age"]), 2)) + "%")
    print("validation")
    print(result["validation_arr_age"])
    print(str(round(np.mean(result["validation_arr_age"]), 2)) + "%")
    print("country_part")
    print("training")
    print(result["training_arr_country_part"])
    print(str(round(np.mean(result["training_arr_country_part"]), 2)) + "%")
    print("validation")
    print(result["validation_arr_country_part"])
    print(str(round(np.mean(result["validation_arr_country_part"]), 2)) + "%")
    print()
