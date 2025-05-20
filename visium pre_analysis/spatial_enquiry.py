#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 20:11:36 2025

@author: raluca
"""

import h5py
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial import distance_matrix
import pickle


adata = sc.read_10x_h5('GSM8238473_P20063_108_filtered_feature_bc_matrix.h5')

spatial_data = pd.read_csv('tissue_positions_list.csv', header=None)

# Assign column names
spatial_data.columns = ['barcode', 'in_tissue', 'row', 'column', 'x', 'y']

# Inspect the first few rows
print(spatial_data.head())

# Filter spots that are in tissue
spatial_data_in_tissue = spatial_data[spatial_data['in_tissue'] == 1]

# Verify filtered data
print(f"Number of spots in tissue: {spatial_data_in_tissue.shape[0]}")
print(spatial_data_in_tissue.head())





matching_barcodes = spatial_data_in_tissue['barcode'].isin(adata.obs_names)

# Filter spatial data to include only matching barcodes
spatial_data_matched = spatial_data_in_tissue[matching_barcodes]

# Verify matched data
print(f"Number of matching barcodes: {spatial_data_matched.shape[0]}")
print(spatial_data_matched.head())


spatial_data_matched = spatial_data_matched.set_index('barcode')
spatial_data_matched = spatial_data_matched.loc[adata.obs_names]

# Verify reordered data
print(spatial_data_matched.head())

# Add spatial coordinates (x, y) to adata.obsm
adata.obsm['spatial'] = np.array(spatial_data_matched[['x', 'y']])

# Verify added spatial coordinates
print("Spatial coordinates added to adata.obsm['spatial']:")
print(adata.obsm['spatial'][:5])  # First 5 rows of spatial coordinates

adata.obs['row'] = spatial_data_matched['row'].values
adata.obs['column'] = spatial_data_matched['column'].values

# Verify added row/column metadata
print(adata.obs[['row', 'column']].head())

plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], s=10)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Filtered Spatial Coordinates')
plt.show()


# Load the JSON file
with open('scalefactors_json.json', 'r') as f:
    json_data = json.load(f)

# Access the 'resolutions' key (if it exists)
spot_diameter_res = json_data['spot_diameter_fullres']


nucleotide_ID = adata.obs.index
spatial_data_pixel = spatial_data_matched.iloc[:,[3,4]].to_numpy()
dis_matrix_pixel = distance_matrix(spatial_data_pixel,spatial_data_pixel)
dis_matrix_um = dis_matrix_pixel*55/spot_diameter_res


neighbour_list = [];
for i in range(dis_matrix_um.shape[0]):
    local_list = []
    for j in range(dis_matrix_um.shape[0]):       
        if dis_matrix_um[i,j]>0 and dis_matrix_um[i,j] <100:
            local_list.append(j)
    neighbour_list.append(np.array(local_list))
    
neighbours_info = pd.DataFrame({
    'ID': nucleotide_ID,
    'Arrays': neighbour_list
})   
        



# Save the DataFrame to a file using pickle
with open('dataframe.pkl', 'wb') as file:
    pickle.dump(neighbours_info, file)