import pandas as pd
import os

# soft ensemble
output_path = 'your output path'
output_name_list = ['your out files in output path']
outputs = []
    
for output_name in output_name_list:
    outputs.append(pd.read_csv(os.join(output_path, output_name)))

outputs_pd = outputs[-1].copy()
outputs_pd['prediction'] = 0.0

for i in range(len(outputs_pd)):
    outputs_pd['prediction'] += outputs_pd[i]['prediction']
outputs_pd['prediction'] = outputs_pd['prediction'] / len(outputs_pd)
outputs_pd.to_csv("soft_ensemble.csv", index=False)