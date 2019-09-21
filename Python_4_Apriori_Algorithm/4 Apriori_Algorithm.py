import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# pd.readcsv will consider first row as the header automatically
# Hence we use header=None to remain the first row in the values field
store_data = pd.read_csv("store_data/store_data.csv", header=None).values

# Pre-processing step: Integrate all transaction record in to a 2 dimensions list
records = []
for record in store_data:
    record_row_tmp = []
    for item in record:
        record_row_tmp.append(str(item))
    records.append(record_row_tmp)

# Define the model
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print(association_results[0])