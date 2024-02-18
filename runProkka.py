import os
import traceback
import logging
import os.path
import networkx as nx
import pandas as pd
import numpy as np
import pylab as pl
from matplotlib import collections  as mc
import glob
import re
import csv
import shutil

# 1 True, 0 False
run_prokka = 1
art_simulator = 0
run_art = 0
run_spades = 0
run_shovill = 0
run_panta = 1
split_paralogs = 1
run_multicsar = 0
run_ragout = 0
run_pangraph = 1
# pangenome_data = '/data/hoan/amromics/prediction/data/Ecoli1936/'
pangenome_data = '/data/hoan/amromics/kover2_paper/data/KpPatric/'


# In[10]:


if 'Kp' in pangenome_data:
    data_base = ' --genus Klebsiella --species pneumoniae --cpus 30'
elif 'Ecoli' in pangenome_data:
    data_base = ' --genus Escherichia --species coli --cpus 30'
else:
    data_base = ' --cpus 30'
print("Use the correct database for the species: ", data_base)
print("Use the correct database for the species: ", data_base)


# In[ ]:


output_dir = pangenome_data + 'temp'
# prokka_dir = pangenome_data + 'prokka'
prokka_dir = pangenome_data + 'prokkaMore'
os.system('mkdir '+ prokka_dir)
conda_dir = 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate amromics && '
if run_prokka:
    fffile = pangenome_data + '*.fna'
    # fffile = pangenome_data + '*.fa'
    list_database = glob.glob(fffile)
    for i in range(len(list_database)):
        input_data = list_database[i]
        print(input_data)
        file_name = input_data.split("/")[-1][:-4] 
        prokka_bin = conda_dir + 'prokka --outdir ' + output_dir +data_base +' --prefix ' + file_name +' '+input_data
        # print(prokka_bin)
        os.system(prokka_bin)
        os.system('cp ' + output_dir + '/'+file_name+'.gff ' + prokka_dir)
        os.system('cp ' + output_dir + '/'+file_name+'.fna ' + prokka_dir)
        os.system('cp ' + output_dir + '/'+file_name+'.faa ' + prokka_dir)
        # os.system('cp ' + output_dir + '/'+file_name+'.* ' + prokka_dir)
        os.system('rm -r '+ output_dir)
        # if i>=0:
        #     break;


# In[ ]:





# ### Then, run panta

# In[2]:


# Instruction to run panta here: https://github.com/amromics/panta


# In[2]:


# Run Panta in: /data/hoan/amromics/pantaPred/ file runme.sh (env amromics)


# In[1]:


# panta main -o /data/hoan/amromics/prediction/output/pantaEcoli1936align_v4 -g /data/hoan/amromics/prediction/data/Ecoli1936/prokka/*.gff -a protein
# panta main -o /data/hoan/amromics/prediction/output/pantaEcoli1936align_v5 -g /data/hoan/amromics/prediction/data/Ecoli1936/prokka/*.gff -a protein -i 0.95
# panta main -o /data/hoan/amromics/prediction/output/pantaEcoli1936align_v6 -g /data/hoan/amromics/prediction/data/Ecoli1936/prokka/*.gff -s -i 0.95


# In[ ]:




