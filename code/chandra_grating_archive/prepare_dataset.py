import os
import pandas as pd

df = pd.read_csv("chaser_all_ACIS_grating_obsids.txt", sep="\t")

for index, row in df.iterrows():
    print(row.ObsID)
    print(f"Processing Obs ID {row.ObsID}")
    os.system(f"./process.sh {row.ObsID}")v