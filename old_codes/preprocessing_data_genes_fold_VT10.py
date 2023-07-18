#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import seaborn as sns
import glob
from numpy import genfromtxt
# from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np
import json
from collections import OrderedDict
import os
import re
import logging
import multiprocessing
from functools import partial
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
import gzip
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html


# ### Create map from gene ID to cluster ID

# In[30]:


# panta input directory
# pantain_dir = '/data/hoan/amromics/prediction/data/Ecoli1936/prokkatest/'
pantain_dir = '/data/hoan/amromics/prediction/data/Ecoli1936/prokka/'
# pantaout_dir = '/data/hoan/amromics/prediction/output/pantaEcoli1936aligntest/'
pantaout_dir = '/data/hoan/amromics/prediction/output/pantaEcoli1936align_v4/'


# In[31]:


with open(pantaout_dir + 'annotated_clusters.json', 'r') as JSON:
    json_dict = json.load(JSON)
# data = json.loads('/data/hoan/amromics/prediction/output/pantaEcoli1936aligntest/clusters.json')[0]


# In[32]:


# json_dict


# In[33]:


gene2clusterdict = {}
for key in json_dict:
    if len(json_dict[key])==0:
        gene2clusterdict[key] = key
    for gene in json_dict[key]['gene_id']:
        gene2clusterdict[gene] = key


# ### Find all AMR genes

# In[34]:


def parse_gff_AMRgene_finder(gff_fh, sample_id, min_protein_len=40):
    # gene_annotation = OrderedDict()
    # gene_position = OrderedDict()    
    # suffix = 1
    # bed_records = []
    # gene_index = 0
    seq_id = None
    min_cds_len = 3 * min_protein_len
    gene_list = []
    
    for line in gff_fh:            
        if line.startswith('##FASTA'):
            #Done reading gff, move on to reading fasta
            break

        if line[0] == '#':
            continue
        line = line.strip()
        #print(line)
        cells = line.split('\t')
        if cells[2] != 'CDS':
            continue
        if 'BARRGD' not in cells[8]:
            continue
        start = int(cells[3])
        end = int(cells[4])
        length = end - start + 1
        if length < min_cds_len:
            continue
        if length % 3 != 0:
            continue
        cells[0] = cells[0].replace('-','_') #make sure seq_id has no -
        
        if seq_id != cells[0]:
            seq_id = cells[0]
            gene_index = 0

        # strand = cells[6]
        tags = cells[8].split(';')
        gene_id = None
        gene_name = ''
        gene_product = ''
        for tag in tags:
            if tag.startswith('ID='):
                gene_id = tag[3:]
            elif tag.startswith('gene='):                    
                gene_name = tag[5:]
                gene_name = re.sub(r'\W', '_', gene_name)
            elif tag.startswith('product='):                    
                gene_product = tag[8:]
        if gene_id == None:
            continue

        # Ensure gene_id is in the format of sample_id-seq_id-gene_tag
        if not gene_id.startswith(sample_id + '-'):
            gene_id = sample_id + '-' + gene_id

        if not gene_id.startswith(sample_id + '-' + seq_id + '-'):
            gene_id = sample_id + '-' + seq_id + '-' + gene_id[len(sample_id)+1:]

        gene_list.append(gene_id)
    
    return gene_list


# In[35]:


# def parse_alignment(gff_fh):
#     sample_list = []
#     seq_list = []
#     index = 0
#     for line in gff_fh:            
#         if line[0] == '>':
#             if index >= 1:
#                 seq_list.append(seq)
#             index+=1
#             sample_list.append(line.split('-')[0][1:])
#             seq = ''
#         else:
#             seq += line[:-1]
#             # seq_list.append(line)
#     seq_list.append(seq)
#     return sample_list, seq_list


# In[36]:


amr_gene = []
for data_dir in glob.glob(pantain_dir + '*.gff'):
    # print(data_dir)
    in_fh = open(data_dir)
    sample_id = data_dir.split('/')[-1][:-4]
    amr_gene += parse_gff_AMRgene_finder(in_fh, sample_id)
    in_fh.close()


# In[37]:


amr_gene[:3], len(amr_gene)


# In[38]:


#### Map genes back to cluster IDs
amr_clusterID = [gene2clusterdict[gene] for gene in amr_gene]
amr_clusterID = list(set(amr_clusterID))


# In[39]:


len(amr_clusterID)


# ### Compute the core genes

# In[40]:


pa_matrix = pd.read_csv(pantaout_dir+'gene_presence_absence.Rtab', sep='\t', index_col=0).T


# In[41]:


n_samples = pa_matrix.shape[0]
n_genes = pa_matrix.shape[1]


# In[42]:


colsum = pa_matrix.sum()
common_gene_cluster = [colsum.index[idx] for idx in range(n_genes) if colsum[idx] > 0.1*n_samples]


# In[43]:


common_gene_cluster[:4], len(common_gene_cluster)


# ## Compute label encoder for gene clusters

# In[48]:


## TODO: Very important: Choose which gene clusters to encode
for fold_idx in range(15):
    print(fold_idx)
    # computed_gene_cluster = amr_clusterID;
    # computed_gene_cluster = common_gene_cluster[1000*fold_idx: min(1000*(fold_idx + 1), len(common_gene_cluster))];
    computed_gene_cluster = common_gene_cluster[500*fold_idx: min(500*(fold_idx + 1), len(common_gene_cluster))];


    with open(pantaout_dir + 'samples.json', 'r') as JSON:
        sample_dict = json.load(JSON)
    sample2integerindex = {}
    for idx in range(len(sample_dict)):
        sample2integerindex[sample_dict[idx]['id']] = idx
    n_samples = len(sample_dict)

    amr_mat = None;
    start_idx = [0];
    pass_gene_cluster = [];
    for idx in range(len(computed_gene_cluster)):
        alignment_dir = pantaout_dir + 'clusters/' + computed_gene_cluster[idx] +'/'+computed_gene_cluster[idx]+'.faa.aln.gz'
        codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
        le = preprocessing.LabelEncoder()
        le.fit(codes)
        mat = None; index = 0; index_set = []
        with gzip.open(alignment_dir, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                name, sequence = record.id, str(record.seq)
                sample_id = name.split('-')[0]
                if index == 0:
                    mat = np.zeros((n_samples, len(sequence)))
                index += 1
                mat[sample2integerindex[sample_id],:] = 1 + le.transform([*sequence])
                index_set.append(sample2integerindex[sample_id])
                # print(record.id)
        if idx==0:
            pass_gene_cluster.append(computed_gene_cluster[idx])
            start_idx += [start_idx[-1] + mat.shape[1]]
            amr_mat = mat
        else:
            # ## Run feature selection
            variant_thres = 0.1
            vs = True
            if len(index_set) >= int(n_samples*0.05): # khong co y nghia vi minh da chon nhung gene phai xuat hien it nhat 10% roi
                try:
                    sel = VarianceThreshold(variant_thres)
                    sel.fit(mat[index_set,:])
                except ValueError:
                    vs = False
                if vs:
                    mat = mat[:, sel.variances_>variant_thres]
                    if mat.shape[0] > 0:
                        pass_gene_cluster.append(computed_gene_cluster[idx])
                        start_idx += [start_idx[-1] + mat.shape[1]]
                        amr_mat = np.append(amr_mat, mat, axis=1)
    end_idx = [start_idx[idx]-1 for idx in range(1, len(start_idx))]
    start_idx = start_idx[:-1]

    # np.save(pantaout_dir + 'amrlabelencodermat_VarianceThreshold.npy', amr_mat) # save numpy array
    outdata_name = 'genes_fold_VT10_' + str(fold_idx)
    print(outdata_name, amr_mat.shape)
    np.save(pantaout_dir + outdata_name + '.npy', amr_mat) # save numpy array
    len(start_idx), len(end_idx), len(pass_gene_cluster)
    amrgene_annotation = pd.DataFrame({'gene': pass_gene_cluster, 'start_index': start_idx, 'end_index': end_idx})
    amrgene_annotation.to_csv(pantaout_dir + outdata_name + '_geneindex.csv', index=None)

