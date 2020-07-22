
# -------------------------------------
# BIOINFORMATICS Assignment 05b + 05c
# KMEANS 
# Christiane Hütter | me19m009
# -------------------------------------



# Short script Description:
# -------------------------
# PRECALCULATIONS:
# The script is using a randomly generated Dataset
# The embedding was done precalcualting a Distance matrix on the Dataset 
# and using a t-SNE embedding algorithm to generate 2D coordinates
# KMEANS CLUSTERING:
# The KMeans algorithm is performed on the embedded Dataset 
# To estimate the potential amount of clusters within the Dataset, 
# an "elbow method" - plot is generated (which will open in a browser when the script is run)
# Another plot for the Data visualization will open in a browser when the script is executed. 



import scipy
import numpy as np 
from sklearn.cluster import KMeans 
import random 
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE 
import seaborn as sns
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numpy.random import rand


# ---------------------------------------------------
# RANDOM DATASET 
# ---------------------------------------------------

X = np.random.normal(10,1,(200,2))

# from a random data sample calculate the distances (e.g. Euclidean metric) 
# resulting in a [n x n] distance matrix
calc_dist = pdist(X, metric='euclidean')
dist_matrix = squareform(calc_dist)

# Embedding of Distance matrix using t-SNE 
# resulting in x,y coordinates for each element of the input matrix 
X_embedded = TSNE(n_components=2).fit_transform(dist_matrix)



# ---------------------------------------------------
# CHOOSE CLUSTER COUNT - using the Elbow Method 
# ---------------------------------------------------

# Elbow Method to see how many clusters to choose 
# The decision on how many clusters to choose as input parameter
# for KMeans is based on the resulting plot (see line "elbow" shape)


wcss=[] # within cluster sum of squares 
for i in range(1,11): 
    kmeans = KMeans(n_clusters=i, max_iter=10,  n_init=100, random_state=0)
    kmeans.fit(X_embedded)
    wcss.append(kmeans.inertia_) #kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.

x = list(range(1,11))
y = wcss

# plot figure 
fig = go.Figure()

trace = go.Scattergl(x = x,
                y = y,
                mode = 'markers+lines',
                marker=dict(
                    size=2,
                    color='blue')
                )
fig.add_trace(trace)
fig.update_layout(title="Estimating Number of Clusters", template='none', width=500, height=500, showlegend=False)
fig.update_xaxes(title="Number of Clusters")
fig.update_yaxes(title="Cluster Sum of Squares")
fig.show()


# ---------------------------------------------------
# KMEANS / CLUSTERING
# ---------------------------------------------------

# choose this number based on the "elbow"-method
# see first generated diagram
n_clus = 3


# ---------------------------------------------------
# CHOOSING THE COUNT OF ITERATIONS 
# ---------------------------------------------------

iterations = range(1,11)

centers = []
clusterid = []
for i in iterations:
    kmeans = KMeans(init='random', n_clusters=n_clus, random_state=0, max_iter=i, n_init=1, verbose=0)
    ids = kmeans.fit_predict(X_embedded) # compute cluster centers and predict cluster index for each sample
    centers.append(kmeans.cluster_centers_)
    clusterid.append(ids)
    
    
# ---------------------------------------------------
# COLORS: assigned to each cluster 
# ---------------------------------------------------

X_clusterid = clusterid[0]
X_id = range(0,len(X_embedded))

colors = sns.color_palette('hls', n_clus)

# assign colors to cluster id
cols_to_clusters = {}
for ix,col in enumerate(colors):
    for j in X_clusterid:
        if j == ix:
            cols_to_clusters[j]=col
            
d_node_clusterid = {k:v for k,v in zip(X_id, X_clusterid)}

l_cols = []
for k,v in d_node_clusterid.items():
    for key,col in cols_to_clusters.items():
        if v==key:
            l_cols.append(col)
         
        
# ---------------------------------------------------
# COORDINATES of embedded Dataset
# ---------------------------------------------------

l_x = []
l_y = []
cc = 0
for i in X_embedded:
    x = X_embedded[cc][0]
    y = X_embedded[cc][1]
    l_x.append(x)
    l_y.append(y)
    cc+=1
    

# ---------------------------------------------------
# COORDINATES of computed Center points 
# ---------------------------------------------------

lx=[]
ly=[]
for clu in range(n_clus):
    l_x_cen = []
    l_y_cen = []
    for i in centers:
        l_x_cen.append(i[clu][0])
        l_y_cen.append(i[clu][1])
    lx.append(l_x_cen)
    ly.append(l_y_cen)
    
    
# ---------------------------------------------------
# VISUALIZE 
# ---------------------------------------------------

trace_centers_all= []
for idx in range(n_clus):
        trace = go.Scattergl(x = lx[idx],
                                 y = ly[idx],
                                 mode = 'markers',
                                 marker=dict(
                                    size=10,
                                    color='black',
                                    opacity=0.15)
                                 )
        trace_centers_all.append(trace)
        

# ---------------------------------------------------
# PLOT 
# ---------------------------------------------------

fig = go.Figure()
        
trace = go.Scattergl(x = l_x,
                    y = l_y,
                    mode = 'markers',
                    marker=dict(
                        size=4,
                        color=l_cols)
                        )
for j in trace_centers_all:
    fig.add_trace(j)

fig.add_trace(trace)
       
fig.update_layout(title="KMeans | Data visualization", template='none', 
        width=800, height=800,
        showlegend=False)
fig.update_xaxes(title="t-SNE component 1")
fig.update_yaxes(title="t-SNE component 2")
fig.write_image("KMeans_nclus_" + str(n_clus) + "_niter_" + str(len(iterations)) + ".png")

print('')
print('Number of Total Iterations:', len(iterations))
print('Number of Clusters:', n_clus)
print('')
fig.show()