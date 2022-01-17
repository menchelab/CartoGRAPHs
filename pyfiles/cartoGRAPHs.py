 
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

import umap.umap_ as umap

#print('DEBUG:in cartographs')

from cartoGRAPHs.func_load_data import *
from cartoGRAPHs.func_visual_properties import *
from cartoGRAPHs.func_calculations import * 

from cartoGRAPHs.func_embed_plot import * 
from cartoGRAPHs.func_embed_plot import *

from cartoGRAPHs.func_exportVR import * 

#print('DEBUG:in cartographs - import done')

########################################################################################


def generate_layout(G, dim, layoutmethod, dimred_method='umap', Matrix = None):
    '''
    Generates a layout of choice.
    
    Input: 
    G - A networkx Graph
    dim - int; 2 or 3 dimensions
    layouttype - string; for layout type > 'local','global','importance','functional'
    dimred_method - string; optional > choose between e.g. tsne or umap 
    
    Result: 
    A generated layout of choice to be input to a plot function e.g. plot_2Dfigure, plot_3Dfigure
    '''
    if layoutmethod == 'local':
        if dimred_method == 'tsne':
            return layout_local_tsne(G, dim, prplxty=50, density=12, l_rate=200, steps=250, metric='cosine')
        elif dimred_method == 'umap':
            return layout_local_umap(G, dim, n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine')
            
    elif layoutmethod == 'global':
        if dimred_method == 'tsne':
            return layout_global_tsne(G, dim, prplxty=50, density=12, l_rate=200, steps=250, metric='cosine')
        elif dimred_method == 'umap':
            return layout_global_umap(G, dim, n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine')
        
    elif layoutmethod == 'importance':
        if dimred_method == 'tsne':
            return layout_importance_tsne(G, dim, prplxty=50, density=12, l_rate=200, steps=250, metric='cosine')
        elif dimred_method == 'umap':
            return layout_importance_umap(G, dim, n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine')
        
    elif layoutmethod == 'functional':
        if Matrix is None: 
            print('Please specify a functional matrix of choice with N x rows with G.nodes and M x feature columns.')
        elif dimred_method == 'tsne' and Matrix is not None:
            return layout_functional_tsne(G, Matrix, dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine')
        elif dimred_method == 'umap' and Matrix is not None:
            return layout_functional_umap(G, Matrix,dim,n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine')  
        else: 
            print('Something went wrong. Please enter a valid layout type.')
    
    elif layoutmethod == 'precalculated':
        if Matrix is None: 
            print('Please specify a precalculated matrix of choice with N x rows of G.nodes and M x columns of features.')
        elif dimred_method == 'tsne':
            return layout_portrait_tsne(G,Matrix,dim,prplxty=50, density=1, l_rate=200, steps=250, metric='cosine') 
        elif dimred_method == 'umap':
            return layout_portrait_umap(G,Matrix,dim,n_neighbors=8, spread=1, min_dist=0.0, metric='cosine')
    else: 
        print('Something went wrong. Please enter a valid layout type.')
        
        

#--------------------
#
# L O C A L 
#
#--------------------

def layout_local_tsne(G,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
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


def layout_local_umap(G,dim,n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine'):
    
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

def layout_global_tsne(G,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
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

        
def layout_global_umap(G,dim,n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine'):
    
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
# I M P O R T A N C E
#
#--------------------

def layout_importance_tsne(G,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
    feature_dict_sorted = compute_centralityfeatures(G) 
    
    DM = pd.DataFrame.from_dict(feature_dict_sorted,orient = 'index',columns = ['degs','clos','betw','eigen'])
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


def layout_importance_umap(G,dim,n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine'):
    
    feature_dict_sorted = compute_centralityfeatures(G) 

    DM = pd.DataFrame.from_dict(feature_dict_sorted,orient = 'index',columns = ['degs','clos','betw','eigen'])
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

def layout_functional_tsne(G, Matrix,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine',r_scale = 1.2):
    
    if dim == 2:
        tsne2D = embed_tsne_2D(Matrix, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, Matrix, tsne2D, r_scale)
        
        return posG
    
    elif dim == 3: 
        tsne3D = embed_tsne_3D(Matrix, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, Matrix, tsne3D, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


def layout_functional_umap(G, Matrix,dim,n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine',r_scale = 1.2):
    
    if dim == 2:
        umap2D = embed_umap_2D(Matrix, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, Matrix, umap2D,r_scale)
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, Matrix, umap_3D,r_scale)

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

def layout_geodesic(G, d_radius, n_neighbors=8, spread=1.0, min_dist=0.0, DM=None):
    
    #radius_list_norm = preprocessing.minmax_scale((list(d_radius.values())), feature_range=(0, 1.0), axis=0, copy=True)
    #d_radius_norm = dict(zip(list(G.nodes()), radius_list_norm))
    
    if DM is None or DM.empty is True:
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


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#
# N E T W O R K X Spring Layouts
#
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

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



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#
# Specific for Precalculated Matrix (i.e. DM)
#
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def layout_portrait_tsne(G, DM, dim, prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
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



def layout_portrait_umap(G, DM, dim, n_neighbors=8, spread=1.0, min_dist=0.0, metric='cosine',r_scale = 1.2):
    
    if dim == 2:
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D,r_scale)
        
        return posG
    
    elif dim == 3:
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D,r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')