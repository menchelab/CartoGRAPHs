
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains F U N C T I O N S  F O R  L A Y O U T S 
# 
########################################################################################


import os
import networkx as nx
import numpy as np
import pandas as pd
#from node2vec import Node2Vec
#from ge import Struc2Vec
import umap.umap_ as umap


from func_calculations import *
from func_embed_plot import *

########################################################################################

#--------------------
#
# L O C A L 
#
#--------------------

def layout_local_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    A_array = A.toarray()
    DM = pd.DataFrame(A_array, columns = list(G.nodes()), index=list(G.nodes()))
    DM.index = list(G.nodes())
    DM.columns = list(G.nodes()) 
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)
        
        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


def layout_local_umap(G,dim,n_neighbors, spread, min_dist, metric):
    
    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    A_array = A.toarray()
    DM = pd.DataFrame(A_array, columns = list(G.nodes()), index=list(G.nodes()))
    DM.index = list(G.nodes())
    DM.columns = list(G.nodes()) 
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3:
        r_scale = 1.2
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


#--------------------
#
# G L O B A L  
#
#--------------------

def layout_global_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    r=0.9
    alpha=1.0
    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM = pd.DataFrame(FM_m_array).T
    DM.index = list(G.nodes())
    DM.columns = list(G.nodes()) 
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
def layout_global_umap(G,dim,n_neighbors, spread, min_dist, metric):
    
    r=0.9
    alpha=1.0
    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM = pd.DataFrame(FM_m_array).T
    DM.index = list(G.nodes())
    DM.columns = list(G.nodes()) 
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')
        
        
#--------------------
#
# N O D E 2 V E C 
#
#--------------------

def layout_nodevec_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    walk_lngth = 50
    num_wlks = 10
    wrks = 1
    dmns = 50 
    
    node2vec = Node2Vec(G, dimensions=dmns, walk_length=walk_lngth, num_walks=num_wlks, workers=wrks, quiet=True)
    model = node2vec.fit(window=10, min_count=1)
    arr = np.array([model.wv[str(x)] for x in G.nodes()])
    DM = pd.DataFrame(arr)
    DM.index = list(G.nodes())

    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
def layout_nodevec_umap(G,dim,n_neighbors, spread, min_dist, metric):
    
    walk_lngth = 50
    num_wlks = 10
    wrks = 1
    dmns = 50 
    
    node2vec = Node2Vec(G, dimensions=dmns, walk_length=walk_lngth, num_walks=num_wlks, workers=wrks, quiet=True)
    model = node2vec.fit(window=10, min_count=1)
    arr = np.array([model.wv[str(x)] for x in G.nodes()])
    DM = pd.DataFrame(arr)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')
        

#--------------------
#
# I M P O R T A N C E
#
#--------------------

def layout_importance_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    degs = dict(G.degree())
    d_deghubs = {}
    for node, de in sorted(degs.items(),key = lambda x: x[1], reverse = 1):
        d_deghubs[node] = round(float(de/max(degs.values())),4)

    closeness = nx.closeness_centrality(G)
    d_clos = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1):
        d_clos[node] = round(cl,4)

    betweens = nx.betweenness_centrality(G)
    d_betw = {}
    for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
         d_betw[node] = round(be,4)

    eigen = nx.eigenvector_centrality(G)
    d_eigen = {}
    for node, eig in sorted(eigen.items(), key = lambda x: x[1], reverse = 1):
         d_eigen[node] = round(eig,4)

    d_deghubs_sorted = {key:d_deghubs[key] for key in sorted(d_deghubs.keys())}
    d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
    d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
    d_eigen_sorted = {key:d_eigen[key] for key in sorted(d_eigen.keys())}

    feature_dict = dict(zip(d_deghubs_sorted.keys(), zip(d_deghubs_sorted.values(),d_clos_sorted.values(),d_betw_sorted.values(),d_eigen_sorted.values())))

    feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
    DM = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs','clos','betw','eigen'])
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
def layout_importance_umap(G,dim,n_neighbors, spread, min_dist, metric):
    
    degs = dict(G.degree())
    d_deghubs = {}
    for node, de in sorted(degs.items(),key = lambda x: x[1], reverse = 1):
        d_deghubs[node] = round(float(de/max(degs.values())),4)

    closeness = nx.closeness_centrality(G)
    d_clos = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1):
        d_clos[node] = round(cl,4)

    betweens = nx.betweenness_centrality(G)
    d_betw = {}
    for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
         d_betw[node] = round(be,4)

    eigen = nx.eigenvector_centrality(G)
    d_eigen = {}
    for node, eig in sorted(eigen.items(), key = lambda x: x[1], reverse = 1):
         d_eigen[node] = round(eig,4)

    d_deghubs_sorted = {key:d_deghubs[key] for key in sorted(d_deghubs.keys())}
    d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
    d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
    d_eigen_sorted = {key:d_eigen[key] for key in sorted(d_eigen.keys())}

    feature_dict = dict(zip(d_deghubs_sorted.keys(), zip(d_deghubs_sorted.values(),d_clos_sorted.values(),d_betw_sorted.values(),d_eigen_sorted.values())))

    feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
    DM = pd.DataFrame.from_dict(feature_dict_sorted, orient = 'index', columns = ['degs','clos','betw','eigen'])
    DM.index = list(G.nodes())

    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

    
#--------------------
#
# S T R U C 2 V E C 
#
#--------------------

def layout_strucvec_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    nx.write_edgelist(G, 'temp.txt')
    G_ = nx.read_edgelist('temp.txt')
    os.remove('temp.txt')
    
    walk_lngth = 50
    num_wlks = 10
    wrks = 1
    dmns = 50 # len(G.nodes())

    model = model = Struc2Vec(G_, walk_length=walk_lngth , num_walks=num_wlks, workers=wrks , verbose=False) #init model
    model.train(window_size = 5, iter = 3)# train model
    embeddings = model.get_embeddings()
    DM = pd.DataFrame(embeddings).T
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
def layout_strucvec_umap(G,dim,n_neighbors, spread, min_dist, metric):
   
    nx.write_edgelist(G, 'temp.txt')
    G_ = nx.read_edgelist('temp.txt')
    os.remove('temp.txt')
    
    walk_lngth = 50
    num_wlks = 10
    wrks = 1
    dmns = 50 # len(G.nodes())

    model = model = Struc2Vec(G_, walk_length=walk_lngth , num_walks=num_wlks, workers=wrks , verbose=False) #init model
    model.train(window_size = 5, iter = 3)# train model
    embeddings = model.get_embeddings()
    DM = pd.DataFrame(embeddings).T
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


        
#--------------------
#
# F U N C T I O N A L
#
#--------------------

def layout_functional_tsne(G, Matrix,dim,prplxty, density, l_rate, steps, metric):
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(Matrix, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, Matrix, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(Matrix, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, Matrix, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
def layout_functional_umap(G, Matrix,dim,n_neighbors, spread, min_dist, metric):
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(Matrix, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, Matrix, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, Matrix, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')



#--------------------
#
# T O P O G R A P H I C  M A P 
#
#--------------------
        
def layout_topographic(posG2D, d_z):
    
    z_list_norm = preprocessing.minmax_scale((list(d_z.values())), feature_range=(0, 1.0), axis=0, copy=True)

    posG_topographic = {}
    cc = 0
    for k,v in posG2D.items():
        posG_topographic[k] = (v[0],v[1],z_list_norm[cc])
        cc+=1
    
    return posG_topographic


#--------------------
#
# G E O D E S I C  M A P 
#
#--------------------

def layout_geodesic(G, d_radius, n_neighbors, spread, min_dist, DM=None):
    
    #radius_list_norm = preprocessing.minmax_scale((list(d_radius.values())), feature_range=(0, 1.0), axis=0, copy=True)
    #d_radius_norm = dict(zip(list(G.nodes()), radius_list_norm))
    
    if DM.empty is True:
        r=0.9
        alpha=1.0
        A = nx.adjacency_matrix(G)
        FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
        DM = pd.DataFrame(FM_m_array).T
    
    elif DM.all != None:
        pass 

    umap_geodesic = embed_umap_sphere(DM, n_neighbors, spread, min_dist)
    posG_geodesic = get_posG_sphere_norm(G, DM, umap_geodesic, d_radius, #d_radius_norm,
                                         radius_rest_genes = 20)

    return posG_geodesic


#--------------------
#
# N E T W O R K X Spring Layouts
#
#--------------------

def springlayout_2D(G, itr):
    
    posG_spring2D = nx.spring_layout(G, iterations = itr, dim = 2)

    df_posG = pd.DataFrame(posG_spring2D).T
    x = df_posG.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)
    
    posG_spring2D_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))
    
    del posG_spring2D
    del df_posG
    
    return posG_spring2D_norm


def springlayout_3D(G, itr):
    
    posG_spring3D = nx.spring_layout(G, iterations = itr, dim = 3)

    df_posG = pd.DataFrame(posG_spring3D).T
    x = df_posG.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)
    
    posG_spring3D_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values,df_posG_norm[2].values)))
    
    del posG_spring3D
    del df_posG
    
    return posG_spring3D_norm



#--------------------
#
# Specific for Precalculated Matrix (i.e. DM)
#
#--------------------

def layout_portrait_tsne(G, DM, dim, prplxty, density, l_rate, steps, metric):
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)
        
        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')



def layout_portrait_umap(G, DM, dim, n_neighbors, spread, min_dist, metric):
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3:
        r_scale = 1.2
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')