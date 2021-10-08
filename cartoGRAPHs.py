
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

# replaced by stellargraph library 
#from node2vec import Node2Vec
#from ge import Struc2Vec

import stellargraph as sg
from stellargraph import StellarGraph 
from tensorflow import keras
import gensim
from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification
from stellargraph.mapper import GraphWaveGenerator

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
    
    # old version of Node2Vec library (by shenweichen "Graph Embeddings" on github)
    #node2vec = Node2Vec(G, dimensions=dmns, walk_length=walk_lngth, num_walks=num_wlks, workers=wrks, quiet=True)
    #model = node2vec.fit(window=10, min_count=1)
    
    # using Stellargraph Library instead 
    walk_length = 100  # maximum length of a random walk to use throughout this notebook
    stellarG = StellarGraph.from_networkx(G)

    rw = BiasedRandomWalk(stellarG)

    weighted_walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    weighted_model = gensim.models.Word2Vec(weighted_walks, 
                                            vector_size=128, 
                                            window=5, 
                                            min_count=0, 
                                            sg=1, 
                                            workers=1, 
                                            epochs=1)
    
    arr = np.array([weighted_model.wv[x] for x in G.nodes()])
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
    
    # old version of Node2Vec library (by shenweichen "Graph Embeddings" on github)
    #node2vec = Node2Vec(G, dimensions=dmns, walk_length=walk_lngth, num_walks=num_wlks, workers=wrks, quiet=True)
    #model = node2vec.fit(window=10, min_count=1)
    
    # using Stellargraph Library instead 
    walk_length = 100  # maximum length of a random walk to use throughout this notebook
    stellarG = StellarGraph.from_networkx(G)

    rw = BiasedRandomWalk(stellarG)

    weighted_walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    weighted_model = gensim.models.Word2Vec(weighted_walks, 
                                            vector_size=128, 
                                            window=5, 
                                            min_count=0, 
                                            sg=1, 
                                            workers=1, 
                                            epochs=1)
    
    arr = np.array([weighted_model.wv[x] for x in G.nodes()])
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
    
    feature_dict_sorted = compute_centralities(G) 
    
    DM = pd.DataFrame.from_dict(feature_dict_sorted, 
                                orient = 'index', 
                                columns = ['degs','clos','betw','eigen'])
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
    
        feature_dict_sorted = compute_centralities(G) 

    DM = pd.DataFrame.from_dict(feature_dict_sorted, 
                                orient = 'index', 
                                columns = ['degs','clos','betw','eigen'])
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
# A T T R I 2 V E C 
#
#--------------------

def layout_attrivec_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    features = compute_centralityfeatures(G)
    d_features = pd.DataFrame(features).T
    d_features.index = list(G.nodes())

    stellarG = StellarGraph.from_networkx(G, node_features=d_features)
    nodes = list(stellarG.nodes())
    number_of_walks = 4
    length = 5

    unsupervised_samples = UnsupervisedSampler(stellarG, 
                                               nodes=nodes, 
                                               length=length, 
                                               number_of_walks=number_of_walks)

    batch_size = 50
    epochs = 4
    
    generator = Attri2VecLinkGenerator(stellarG, batch_size)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [128]
    attri2vec = Attri2Vec(layer_sizes=layer_sizes, 
                          generator=generator, 
                          bias=False, 
                          normalize=None)

    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.in_out_tensors()

    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = Attri2VecNodeGenerator(stellarG, batch_size).flow(stellarG.nodes())
    embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)
    DM = pd.DataFrame(embeddings)
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

        
def layout_attrivec_umap(G,dim,n_neighbors, spread, min_dist, metric):
    
    features = compute_centralityfeatures(G)
    d_features = pd.DataFrame(features).T
    d_features.index = list(G.nodes())

    stellarG = StellarGraph.from_networkx(G, node_features=d_features)
    nodes = list(stellarG.nodes())
    number_of_walks = 4
    length = 5

    unsupervised_samples = UnsupervisedSampler(stellarG, 
                                               nodes=nodes, 
                                               length=length, 
                                               number_of_walks=number_of_walks)

    batch_size = 50
    epochs = 4
    
    generator = Attri2VecLinkGenerator(stellarG, batch_size)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [128]
    attri2vec = Attri2Vec(layer_sizes=layer_sizes, 
                          generator=generator, 
                          bias=False, 
                          normalize=None)

    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.in_out_tensors()

    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = Attri2VecNodeGenerator(stellarG, batch_size).flow(stellarG.nodes())
    embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)
    DM = pd.DataFrame(embeddings)
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
# G R A P H W A V E 
#
#--------------------


def layout_graphwave_tsne(G,dim,prplxty, density, l_rate, steps, metric):
    
    features = compute_centralityfeatures(G)
    d_features = pd.DataFrame(features).T
    d_features.index = list(G.nodes())

    stellarG = StellarGraph.from_networkx(G, node_features=d_features)
    
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    degree = 20
    scales = [5, 10]

    generator = GraphWaveGenerator(stellarG, scales=scales, degree=degree)

    embeddings_dataset = generator.flow(
        node_ids=G.nodes(), 
        sample_points=sample_points, 
        batch_size=1, repeat=False)

    embeddings_notstacked = [x.numpy() for x in embeddings_dataset]
    embeddings = np.vstack(embeddings_notstacked)
    DM = pd.DataFrame(embeddings)
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


        
def layout_graphwave_umap(G,dim,n_neighbors, spread, min_dist, metric):
    
    stellarG = StellarGraph.from_networkx(G, node_features=d_features)
    
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    degree = 20
    scales = [5, 10]

    generator = GraphWaveGenerator(stellarG, scales=scales, degree=degree)

    embeddings_dataset = generator.flow(
        node_ids=G.nodes(), 
        sample_points=sample_points, 
        batch_size=1, repeat=False)

    embeddings_notstacked = [x.numpy() for x in embeddings_dataset]
    embeddings = np.vstack(embeddings_notstacked)
    DM = pd.DataFrame(embeddings)
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