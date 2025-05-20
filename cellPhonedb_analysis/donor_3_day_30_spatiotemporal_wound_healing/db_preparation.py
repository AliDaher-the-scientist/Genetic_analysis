#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 21:07:04 2025

@author: raluca
"""

import scanpy as sc
import pandas as pd
import numpy as np
import cellphonedb
from cellphonedb.src.core.methods import cpdb_analysis_method
import glob
import os
from IPython.display import HTML, display
from cellphonedb.utils import db_releases_utils
from cellphonedb.utils import db_utils
display(HTML(db_releases_utils.get_remote_database_versions_html()['db_releases_html_table']))


# -- Version of the databse
cpdb_version = 'v5.0.0'

# -- Path where the input files to generate the database are located
cpdb_target_dir = os.path.join('/home/raluca/spatiotemporal skin wound healing/donor 4/cellphonedb analysis', cpdb_version)
db_utils.download_database(cpdb_target_dir, cpdb_version)








#used for mapping between gene symbols and gene ids.
features = pd.read_csv("features.tsv", sep="\t", header=None)
features.columns = ["gene_id", "gene_symbol", "gene_type"]
features_unique = features.drop_duplicates(subset=["gene_symbol"])

# 1. Load Raw scRNA-seq Data and prepare count file
sc_adata = sc.read_h5ad('global_adata_with_celltypes.h5ad')
df_expr_matrix = sc_adata.X
df_expr_matrix = df_expr_matrix.T
df_expr_matrix = pd.DataFrame(df_expr_matrix)
# Set cell IDs as columns.
df_expr_matrix.columns = sc_adata.obs.index
# Genes should be either Ensembl IDs or gene names.
gene_symbols = pd.DataFrame(sc_adata.var.index, columns=["gene_symbol"])
merged = gene_symbols.merge(features_unique[["gene_id", "gene_symbol"]], on="gene_symbol", how="left")
gene_iDs = merged["gene_id"]
df_expr_matrix.set_index(gene_iDs, inplace=True)
savepath_counts = 'counts_matrix.tsv'  # or any path you want
df_expr_matrix.to_csv(savepath_counts,sep='\t')

#Also save as h5ad
sc_adata.var["gene_id"] = gene_iDs.astype(str).values
sc_adata.var.set_index("gene_id", inplace=True)

# Save the updated AnnData object
savepath_h5ad = "counts_matrix.h5ad"
sc_adata.write(savepath_h5ad)


#prepapre metadata file
df_meta = pd.DataFrame(data={'Cell':list(sc_adata.obs.index),
                             'cell_type':[ i for i in sc_adata.obs['cell type']]
                            })
df_meta.set_index('Cell', inplace=True)
df_meta.to_csv('doc_meta.tsv', sep = '\t')



#optional prepare DEG file
cell_types = sc_adata.obs['cell type'].unique()
file_path = 'The top 50 marker genes of the 27 cell clusters identified by scRNA-seq of human skin and acute wounds.ods'
df = pd.read_excel(file_path, engine='odf')

# Select the first two columns: 'Cluster' and 'gene'
df_filtered = df[['Cluster', 'gene']]

# Define the cell types of interest

# Filter rows where the 'Cluster' is in the list of cell types of interest
df_filtered = df_filtered[df_filtered['Cluster'].isin(cell_types)]
gene_mapping = dict(zip(features_unique['gene_symbol'], features_unique['gene_id']))

# Map gene symbols to gene IDs in df_filtered
df_filtered['gene_id'] = df_filtered['gene'].map(gene_mapping)

# Now you can drop the original 'gene' column and keep 'gene_id' as the second column
df_filtered = df_filtered.drop(columns=['gene'])
deg_output_file = 'degs.txt'
df_filtered[['Cluster', 'gene_id']].to_csv(deg_output_file, sep='\t', index=False, header=False)


from cellphonedb.src.core.methods import cpdb_degs_analysis_method

cpdb_results = cpdb_degs_analysis_method.call(
         cpdb_file_path = '/home/raluca/spatiotemporal skin wound healing/donor 3/cellphonedb analysis/v5.0.0/cellphonedb.zip',
         meta_file_path = 'doc_meta.tsv',
         counts_file_path = 'counts_matrix.h5ad',
         degs_file_path = 'degs.txt',
         threshold = 0.1,
         counts_data = 'ensembl',
         output_path = 'out_path')


df_mean = pd.read_csv('./out_path/degs_analysis_means_04_24_2025_201647.txt', sep='\t')
df_mean.to_csv('./out_path/donor_3_day_30_mean_deg_analysis.csv',delim_whitespace=True, index=False)
