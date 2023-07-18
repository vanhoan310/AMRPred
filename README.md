# AMRPred

First you need to clone PanPred to this directory.
https://github.com/DaneshMoradigaravand/PanPred.git

Then, run jupyternotebook in order 1, 2, 3, 4, 5.


pantaCombine: Use gene matrix + SNPs.
pantaCombineScale: Scale feature to have zero mean and 1 variance.
pantaHighGene: sequence features of top 800 high degree genes, use 3.b notebook, no VarianceThreshold(), label encoder, adj_matrix = adj_matrix.multiply(adj_matrix>=0.1*self.n_samples)
pantaVT10: sequence features of AMR genes, VarianceThreshold = 0.1, label encoder
pantaOneHotVT10: sequence features of AMR genes, VarianceThreshold = 0.1, one hot encoder.
pantaHighGeneNeighborVT5: sequence features of top 400 high degree genes + neibor, use 3.b notebook v2, VT = 0.05, label encoder
pantaDifferSite: Count number of different proteins between represenative and the sample, if not in: = 0
pantaSimSite: Count number of the same proteins between represenative and the sample, if not in: = 0