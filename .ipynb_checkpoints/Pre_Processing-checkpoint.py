# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:15:59 2023

@author: coope
"""

import json
import pandas as pd
import numpy as np
from io import StringIO

## Import Data
path_bus = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\yelp_dataset\\yelp_academic_dataset_business.json'
path_review = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\yelp_dataset\\yelp_academic_dataset_review.json'
output_path = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\yelp_dataset\\business_id.csv'
output_path_data = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\yelp_dataset\\total_data.csv'


df = pd.read_json(path_bus, lines = True)

df = df.fillna("")


resturants = df[df["categories"].str.contains("Restaurants")]

#resturants.to_csv(output_path)

res_id = resturants["business_id"]

del df
del resturants


chunks = pd.read_json(path_review, lines = True, chunksize = 10000)
#i = 0
total_data = pd.DataFrame()

for chunk in chunks:
    #if i >= 500:
        #break
    #i += 1
    resturant = chunk[chunk["business_id"].isin(res_id)][["stars", "text"]]
    total_data = pd.concat([total_data, resturant])
    
del resturant
del chunk
del chunks
del i
del output_path
del path_bus
del path_review
del res_id

total_data["text"][0]
total_data["stars"][0]

total_data.to_csv(output_path_data)



        
        
        
        
