pip install scanpy
pip install leidenalg


import numpy as np
import pandas as pd
import scanpy as sc
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


adata = sc.read_csv('DataRaw/AML5_expression_counts.csv') #first_column_names='gene_symbols'

adata=adata.transpose()

adata2=sc.read_csv('DataRaw/H7_expression_counts.csv')

adata2=adata2.transpose()

adata3 = sc.read_csv('DataRaw/AML4_expression_counts.csv') #first_column_names='gene_symbols'

adata3=adata3.transpose()


sc.pp.filter_cells(adata, min_genes=200) # filter out cells that have less than 200 genes
sc.pp.filter_genes(adata, min_cells=3) # filter out genes that are detected in less than 3 cells
adata.shape


# Generate quality control metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata
# High proportions of mito genes are indicative of poor-quality cells (Islam et al. 2014; Ilicic et al. 2016), possibly because of loss of cytoplasmic RNA from perforated cells
adata.var[adata.var['mt']] # look at the mitochrondrial genes
adata.obs['pct_counts_mt']
adata.obs['pct_counts_mt'].describe()

# Violin plots of qc metrics
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save= '.pdf') # can also save as png


# Scatter plots of qc metrics
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', save ='_total_counts_pct_mt.pdf')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='_total_counts_genes.pdf')

# Visualize optimal cutoff values prior to filtering
with PdfPages('figures/scatter_total_counts_pct_mt_with_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=7.5, color='red', linestyle='--')
  ax1.axvline(x=500, color = 'red', linestyle='--')
  ax1.axvline(x=15000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Remove cells that have too many mitochondrial genes expressed or too many total counts
keep = (adata.obs['pct_counts_mt']> 0.001) & (adata.obs['pct_counts_mt'] < 7.5) & (adata.obs['total_counts'] > 500) & (adata.obs['total_counts'] < 15000)
print("Removed cells: %d"%(adata.n_obs - sum(keep)))

# Actually do the filtering
adata = adata[keep, :]
adata.shape


# Check to make sure filters worked
with PdfPages('figures/scatter_total_counts_pct_mt_after_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=7.5, color='red', linestyle='--')
  ax1.axvline(x=500, color = 'red', linestyle='--')
  ax1.axvline(x=15000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Normalize so that counts become comparable among cells
sc.pp.normalize_total(adata, target_sum=1e4)

# Logarithmize the data
sc.pp.log1p(adata)

# Identify and plot highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=3000) # set your own number of highly variable genes
sc.pl.highly_variable_genes(adata, save='.pdf')

# Save raw data prior to subsetting data to highly variable genes
adata.raw = adata

# Subset data for highly variable genes
adata = adata[:, adata.var.highly_variable]
adata

# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# Scale each gene to unit variance
sc.pp.scale(adata, max_value=10)


#----------------------------------------------
# PCA / clustering /  marker gene analysis
#----------------------------------------------

# PCA
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CD3D', save = '.pdf')
sc.pl.pca_variance_ratio(adata, log=True, save = '.pdf')
adata

# Compute the neighborhood graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)

# Visualize cells in 2D  set(s)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['CD34','RGS5','PDGFRB','CD3D','CD3E','CD79A','HLA-DPB1','APOC1','CCL20','FABP7','GFAP','GAPDH','CCNA2','CCNB1','PCNA','TOP2A','VEGFA','NDRG1'], save ='.pdf')
sc.pl.umap(adata, color=['CD34','RGS5','PDGFRB','CD3D','CD3E','CD79A','HLA-DPB1','APOC1','CCL20','FABP7','GFAP','GAPDH','CCNA2','CCNB1','PCNA','TOP2A','VEGFA','NDRG1'], use_raw=False, save='_V2.pdf') # scaled and corrected gene expression values

# Clustering
sc.tl.leiden(adata, resolution=0.3) # scanpy recommends the Leiden graph-clustering method (community detection based on optimizing modularity)
sc.pl.umap(adata, color=['leiden', 'CD3D', 'NKG7'], save='_leiden.pdf')

# Finding marker genes
#sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
#sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='.pdf')

sc.tl.rank_genes_groups(adata, 'leiden',corr_method='benjamini-hochberg', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_V2.pdf')
sc.pl.rank_genes_groups_heatmap(adata, save='_marker_genes.pdf')

# Define a list of marker genes (literature markers)
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

sc.pl.umap(adata, color=marker_genes, save='_with_marker_genes.pdf')

# Show the 10 top ranked genes per cluster
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

# Get a table with scores and results
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)


# Compare to a single cluster
sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)

sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8, save='marker_genes_violin.pdf')

sc.pl.violin(adata, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_three_genes.pdf')

# Identify cell types
new_cluster_names = [
    'CTL', 'Cancer Cells',
    'naïve T cells', 'Lymphoma',
    'Myleoid cells', 
    'T ', 'Leukemia cells', 'B ', 'HSC', 'MDSC', 'NK', 'Platelets','MLSC']
adata.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

# other marker gene visualization
sc.pl.violin(adata, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_V2')
sc.pl.violin(adata, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', rotation=90, save='_V3')
sc.pl.violin(adata, marker_genes, groupby='leiden', rotation=90, save='_V4')

for gene in marker_genes[0:5]:
    sc.pl.violin(adata, gene, groupby='leiden', rotation=90, save = '_'+gene+'.pdf')

sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', swap_axes = True, save='.pdf')
sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', save='V2.pdf')

sc.pl.dotplot(adata, marker_genes, groupby='leiden', save='pdf')
sc.pl.heatmap(adata, marker_genes, groupby='leiden', save='.pdf')

marker_genes_dict = {'B-cell': ['CD79A', 'MS4A1'],
                     'T-cell': 'CD3D',
                     'T-cell CD8+': ['CD8A', 'CD8B'],
                     'NK': ['GNLY', 'NKG7'],
                     'Myeloid': ['CST3', 'LYZ'],
                     'Monocytes': ['FCGR3A'],
                     'Dendritic': ['FCER1A']}

sc.pl.heatmap(adata, marker_genes_dict, groupby='leiden', save='_V2.pdf')





########################################################################################
########################################################################################
# Now for the Healthy Patients 
########################################################################################
########################################################################################


sc.pp.filter_cells(adata2, min_genes=200) # filter out cells that have less than 200 genes
sc.pp.filter_genes(adata2, min_cells=3) # filter out genes that are detected in less than 3 cells
adata2.shape


# Generate quality control metrics
adata2.var['mt'] = adata2.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata2, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata2
# High proportions of mito genes are indicative of poor-quality cells (Islam et al. 2014; Ilicic et al. 2016), possibly because of loss of cytoplasmic RNA from perforated cells
adata2.var[adata2.var['mt']] # look at the mitochrondrial genes
adata2.obs['pct_counts_mt']
adata2.obs['pct_counts_mt'].describe()

# Violin plots of qc metrics
sc.pl.violin(adata2, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save= '.pdf') # can also save as png


# Scatter plots of qc metrics
sc.pl.scatter(adata2, x='total_counts', y='pct_counts_mt', save ='_total_counts_pct_mt.pdf')
sc.pl.scatter(adata2, x='total_counts', y='n_genes_by_counts', save='_total_counts_genes.pdf')

# Visualize optimal cutoff values prior to filtering
with PdfPages('figures/scatter_total_counts_pct_mt_with_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata2, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=8, color='red', linestyle='--')
  ax1.axvline(x=500, color = 'red', linestyle='--')
  ax1.axvline(x=13000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Remove cells that have too many mitochondrial genes expressed or too many total counts
keep = (adata2.obs['pct_counts_mt']> 0.001) & (adata2.obs['pct_counts_mt'] < 7.5) & (adata2.obs['total_counts'] > 500) & (adata2.obs['total_counts'] < 15000)
print("Removed cells: %d"%(adata2.n_obs - sum(keep)))

# Actually do the filtering
adata2 = adata2[keep, :]
adata2.shape


# Check to make sure filters worked
with PdfPages('figures/scatter_total_counts_pct_mt_after_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata2, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=7.5, color='red', linestyle='--')
  ax1.axvline(x=500, color = 'red', linestyle='--')
  ax1.axvline(x=15000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Normalize so that counts become comparable among cells
sc.pp.normalize_total(adata2, target_sum=1e4)

# Logarithmize the data
sc.pp.log1p(adata2)

# Identify and plot highly variable genes
sc.pp.highly_variable_genes(adata2, n_top_genes=3000) # set your own number of highly variable genes
sc.pl.highly_variable_genes(adata2, save='.pdf')

# Save raw data prior to subsetting data to highly variable genes
adata2.raw = adata2

# Subset data for highly variable genes
adata2 = adata2[:, adata2.var.highly_variable]
adata2

# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
sc.pp.regress_out(adata2, ['total_counts', 'pct_counts_mt'])

# Scale each gene to unit variance
sc.pp.scale(adata2, max_value=10)


#----------------------------------------------
# PCA / clustering /  marker gene analysis
#----------------------------------------------

# PCA
sc.tl.pca(adata2, svd_solver='arpack')
sc.pl.pca(adata2, color='CD3D', save = '.pdf')
sc.pl.pca_variance_ratio(adata2, log=True, save = '.pdf')
adata2

# Compute the neighborhood graph
sc.pp.neighbors(adata2, n_neighbors=15, n_pcs=40)

# Visualize cells in 2D  set(s)
sc.tl.umap(adata2)
sc.pl.umap(adata2, color=['CD34','RGS5','PDGFRB','CD3D','CD3E','CD79A','HLA-DPB1','APOC1','CCL20','GAPDH','CCNA2','CCNB1','PCNA','TOP2A','VEGFA','NDRG1'], save ='.pdf')
sc.pl.umap(adata2, color=['CD34','RGS5','PDGFRB','CD3D','CD3E','CD79A','HLA-DPB1','APOC1','CCL20','GAPDH','CCNA2','CCNB1','PCNA','TOP2A','VEGFA','NDRG1'], use_raw=False, save='_V2.pdf') # scaled and corrected gene expression values

# Clustering
sc.tl.leiden(adata2, resolution=0.3) # scanpy recommends the Leiden graph-clustering method (community detection based on optimizing modularity)
sc.pl.umap(adata2, color=['leiden', 'CD3D', 'NKG7'], save='_leiden.pdf')

# Finding marker genes
#sc.tl.rank_genes_groups(adata2, 'leiden', method='t-test')
#sc.pl.rank_genes_groups(adata2, n_genes=25, sharey=False, save='.pdf')

sc.tl.rank_genes_groups(adata2, 'leiden',corr_method='benjamini-hochberg', method='wilcoxon')
sc.pl.rank_genes_groups(adata2, n_genes=25, sharey=False, save='_V2.pdf')
sc.pl.rank_genes_groups_heatmap(adata2, save='_marker_genes.pdf')

# Define a list of marker genes (literature markers)
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

sc.pl.umap(adata2, color=marker_genes, save='_with_marker_genes.pdf')

# Show the 10 top ranked genes per cluster
pd.DataFrame(adata2.uns['rank_genes_groups']['names']).head(5)

# Get a table with scores and results
result = adata2.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)


# Compare to a single cluster
sc.tl.rank_genes_groups(adata2, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata2, groups=['0'], n_genes=20)

sc.pl.rank_genes_groups_violin(adata2, groups='0', n_genes=8, save='marker_genes_violin.pdf')

sc.pl.violin(adata2, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_three_genes.pdf')

# Identify cell types
new_cluster_names = [
    'naïve T cells', 'T',
    'NK', 'Dentritic',
    'Monocytes', 
    'B ', 'Platelets', 'HSC']
adata2.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata2, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

sc.pl.umap(adata2, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

# other marker gene visualization
sc.pl.violin(adata2, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_V2')
sc.pl.violin(adata2, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', rotation=90, save='_V3')
sc.pl.violin(adata2, marker_genes, groupby='leiden', rotation=90, save='_V4')

for gene in marker_genes[0:5]:
    sc.pl.violin(adata2, gene, groupby='leiden', rotation=90, save = '_'+gene+'.pdf')

sc.pl.stacked_violin(adata2, marker_genes, groupby='leiden', swap_axes = True, save='.pdf')
sc.pl.stacked_violin(adata2, marker_genes, groupby='leiden', save='V2.pdf')

sc.pl.dotplot(adata2, marker_genes, groupby='leiden', save='pdf')
sc.pl.heatmap(adata2, marker_genes, groupby='leiden', save='.pdf')

marker_genes_dict = {'B-cell': ['CD79A', 'MS4A1'],
                     'T-cell': 'CD3D',
                     'T-cell CD8+': ['CD8A', 'CD8B'],
                     'NK': ['GNLY', 'NKG7'],
                     'Myeloid': ['CST3', 'LYZ'],
                     'Monocytes': ['FCGR3A'],
                     'Dendritic': ['FCER1A']}

sc.pl.heatmap(adata2, marker_genes_dict, groupby='leiden', save='_V2.pdf')








########################################################################################
########################################################################################
# Now for the the second AML Patient 
########################################################################################
########################################################################################




sc.pp.filter_cells(adata3, min_genes=200) # filter out cells that have less than 200 genes
sc.pp.filter_genes(adata3, min_cells=3) # filter out genes that are detected in less than 3 cells
adata3.shape


# Generate quality control metrics
adata3.var['mt'] = adata3.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata3, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata3
# High proportions of mito genes are indicative of poor-quality cells (Islam et al. 2014; Ilicic et al. 2016), possibly because of loss of cytoplasmic RNA from perforated cells
adata3.var[adata3.var['mt']] # look at the mitochrondrial genes
adata3.obs['pct_counts_mt']
adata3.obs['pct_counts_mt'].describe()

# Violin plots of qc metrics
sc.pl.violin(adata3, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True, save= '.pdf') # can also save as png


# Scatter plots of qc metrics
sc.pl.scatter(adata3, x='total_counts', y='pct_counts_mt', save ='_total_counts_pct_mt.pdf')
sc.pl.scatter(adata3, x='total_counts', y='n_genes_by_counts', save='_total_counts_genes.pdf')

# Visualize optimal cutoff values prior to filtering
with PdfPages('figures/scatter_total_counts_pct_mt_with_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata3, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=8, color='red', linestyle='--')
  ax1.axvline(x=1000, color = 'red', linestyle='--')
  ax1.axvline(x=11000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Remove cells that have too many mitochondrial genes expressed or too many total counts
keep = (adata3.obs['pct_counts_mt']> 0.001) & (adata3.obs['pct_counts_mt'] < 8) & (adata3.obs['total_counts'] > 1000) & (adata3.obs['total_counts'] < 11000)
print("Removed cells: %d"%(adata3.n_obs - sum(keep)))

# Actually do the filtering
adata3 = adata3[keep, :]
adata3.shape


# Check to make sure filters worked
with PdfPages('figures/scatter_total_counts_pct_mt_after_cutoffs.pdf') as pp:
  ax1 = sc.pl.scatter(adata3, x='total_counts', y='pct_counts_mt', show=False)
  ax1.axhline(y=0.001, color='red', linestyle='--')
  ax1.axhline(y=8, color='red', linestyle='--')
  ax1.axvline(x=1000, color = 'red', linestyle='--')
  ax1.axvline(x=11000, color = 'red', linestyle='--')
  pp.savefig()
  plt.close()

# Normalize so that counts become comparable among cells
sc.pp.normalize_total(adata3, target_sum=1e4)

# Logarithmize the data
sc.pp.log1p(adata3)

# Identify and plot highly variable genes
sc.pp.highly_variable_genes(adata3, n_top_genes=3000) # set your own number of highly variable genes
sc.pl.highly_variable_genes(adata3, save='.pdf')

# Save raw data prior to subsetting data to highly variable genes
adata3.raw = adata3

# Subset data for highly variable genes
adata3 = adata3[:, adata3.var.highly_variable]
adata3

# Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed
sc.pp.regress_out(adata3, ['total_counts', 'pct_counts_mt'])

# Scale each gene to unit variance
sc.pp.scale(adata3, max_value=10)


#----------------------------------------------
# PCA / clustering /  marker gene analysis
#----------------------------------------------

# PCA
sc.tl.pca(adata3, svd_solver='arpack')
sc.pl.pca(adata3, color='CD3D', save = '.pdf')
sc.pl.pca_variance_ratio(adata3, log=True, save = '.pdf')
adata3

# Compute the neighborhood graph
sc.pp.neighbors(adata3, n_neighbors=15, n_pcs=40)

# Visualize cells in 2D  set(s)
sc.tl.umap(adata3)
sc.pl.umap(adata3, color=['CD34','RGS5','PDGFRB','CD3D','CD3E','CD79A','HLA-DPB1','APOC1','CCL20','GAPDH','CCNA2','CCNB1','PCNA','TOP2A','VEGFA','NDRG1'], save ='.pdf')
sc.pl.umap(adata3, color=['CD34','RGS5','PDGFRB','CD3D','CD3E','CD79A','HLA-DPB1','APOC1','CCL20','GAPDH','CCNA2','CCNB1','PCNA','TOP2A','VEGFA','NDRG1'], use_raw=False, save='_V2.pdf') # scaled and corrected gene expression values

# Clustering
sc.tl.leiden(adata3, resolution=0.3) # scanpy recommends the Leiden graph-clustering method (community detection based on optimizing modularity)
sc.pl.umap(adata3, color=['leiden', 'CD3D', 'NKG7'], save='_leiden.pdf')

# Finding marker genes
#sc.tl.rank_genes_groups(adata3, 'leiden', method='t-test')
#sc.pl.rank_genes_groups(adata3, n_genes=25, sharey=False, save='.pdf')

sc.tl.rank_genes_groups(adata3, 'leiden',corr_method='benjamini-hochberg', method='wilcoxon')
sc.pl.rank_genes_groups(adata3, n_genes=25, sharey=False, save='_V2.pdf')
sc.pl.rank_genes_groups_heatmap(adata3, save='_marker_genes.pdf')

# Define a list of marker genes (literature markers)
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

sc.pl.umap(adata3, color=marker_genes, save='_with_marker_genes.pdf')

# Show the 10 top ranked genes per cluster
pd.DataFrame(adata3.uns['rank_genes_groups']['names']).head(5)

# Get a table with scores and results
result = adata3.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)


# Compare to a single cluster
sc.tl.rank_genes_groups(adata3, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata3, groups=['0'], n_genes=20)

sc.pl.rank_genes_groups_violin(adata3, groups='0', n_genes=8, save='marker_genes_violin.pdf')

sc.pl.violin(adata3, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_three_genes.pdf')

# Identify cell types
new_cluster_names = ['Cytoxic T Cells', 'B','HSC', 'Macrophages','T', 'Lymphocytes', 'NK Cells', 'Myeloid Leukocytes','Platelets','Dendritic cells','Monocytes']
adata3.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata3, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

sc.pl.umap(adata3, color='leiden', legend_loc='on data', title='', frameon=False, save='_new_idents.pdf')

# other marker gene visualization
sc.pl.violin(adata3, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', save='_V2')
sc.pl.violin(adata3, ['CD3D', 'NKG7', 'PPBP'], groupby='leiden', rotation=90, save='_V3')
sc.pl.violin(adata3, marker_genes, groupby='leiden', rotation=90, save='_V4')

for gene in marker_genes[0:5]:
    sc.pl.violin(adata3, gene, groupby='leiden', rotation=90, save = '_'+gene+'.pdf')

sc.pl.stacked_violin(adata3, marker_genes, groupby='leiden', swap_axes = True, save='.pdf')
sc.pl.stacked_violin(adata3, marker_genes, groupby='leiden', save='V2.pdf')

sc.pl.dotplot(adata3, marker_genes, groupby='leiden', save='pdf')
sc.pl.heatmap(adata3, marker_genes, groupby='leiden', save='.pdf')

marker_genes_dict = {'B-cell': ['CD79A', 'MS4A1'],
                     'T-cell': 'CD3D',
                     'T-cell CD8+': ['CD8A', 'CD8B'],
                     'NK': ['GNLY', 'NKG7'],
                     'Myeloid': ['CST3', 'LYZ'],
                     'Monocytes': ['FCGR3A'],
                     'Dendritic': ['FCER1A']}

sc.pl.heatmap(adata3, marker_genes_dict, groupby='leiden', save='_V2.pdf')

  