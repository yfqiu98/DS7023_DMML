import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# pd.readcsv will consider first row as the header automatically
# Hence we use header=None to remain the first row in the values field
store_data = pd.read_csv("store_data/store_data.csv", header=None)

# Pre-processing step: Integrate all transaction record in to a 2 dimensions list

# records = []
# for i in range(0, 7501):
#     records.append([str(store_data.values[i,j]) for j in range(0, 20)])

records = []
for i in range(0, 7501):
    row_tmp = []
    for j in range(0, 20):
        row_tmp.append(str(store_data.values[i][j]))
    records.append(row_tmp)
print(records)

# Define the model
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    # second index of the inner list
    print("Support: " + str(item[1]))

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")