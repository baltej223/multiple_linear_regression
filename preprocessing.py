import pandas as pd 
  
# making dataframe 
df = pd.read_csv("housing.csv") 

new_df = df.copy()
new_df = new_df.drop(columns=["ocean_proximity"])

# column for near bay 
list_near_bay = []
for e in df["ocean_proximity"]:
    if e == "NEAR BAY":
        list_near_bay.append(1)
    else:
        list_near_bay.append(0)

# setting it as a cloumn
new_df["is_near_bay"] = list_near_bay

# column for <1H OCEAN 
list_1hr_ocean = []
for e in df["ocean_proximity"]:
    if e == "<1H OCEAN":
        list_1hr_ocean.append(1)
    else:
        list_1hr_ocean.append(0)

# setting it as a cloumn
new_df["is_1hour_ocean"] = list_1hr_ocean

# column for INLAND 
list_inland = []
for e in df["ocean_proximity"]:
    if e == "INLAND":
        list_inland.append(1)
    else:
        list_inland.append(0)

# setting it as a cloumn
new_df["is_inland"] = list_inland

# column for near_ocean 
list_near_ocean = []
for e in df["ocean_proximity"]:
    if e == "NEAR OCEAN":
        list_near_ocean.append(1)
    else:
        list_near_ocean.append(0)

# setting it as a cloumn
# new_df["list_near_ocean"] = list_near_ocean

print(new_df)
new_df = new_df.dropna();

new_df.to_csv("processed.csv", index=False)
