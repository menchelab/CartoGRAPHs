
########################################################################################
#
#  L I B R A R I E S
#
########################################################################################

import ast 

from Bio import Entrez

import collections
from collections import defaultdict as dd
from collections import Counter as ct
from collections import OrderedDict
import colorsys
#from colormap import rgb2hex, rgb2hls, hls2rgb
from colormath.color_objects import sRGBColor, LabColor
#from colormath.color_conversions import convert_color

#from fisher import pvalue
#from fa2 import ForceAtlas2
#from fastdist import fastdist

#from html2image import Html2Image

import itertools as it
import igraph as ig

import math
import matplotlib.pyplot as plt
#%matplotlib inline
#import multiprocessing
#import mygene

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.generators.degree_seq import expected_degree_graph
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
from numpy import pi, cos, sin, arccos, arange
import numpy.linalg as la
#import numba
#@numba.njit(fastmath=True)

import os
import os.path

import pandas as pd
import pickle
import plotly
import plotly.graph_objs as pgo
import plotly.offline as py
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
import pylab
#py.init_notebook_mode(connected = True)
import pymysql as mysql

import random as rd

from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
import scipy.stats as st
from scipy import stats

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection,cluster)
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
import statistics
import sys 

import time

import umap.umap_ as umap
from node2vec import Node2Vec

import warnings
#warnings.filterwarnings("ignore", category=UserWarning)





########################################################################################
#
# E M B E D D I N G & P L O T T I N G  2D + 3D 
#
########################################################################################


# -------------------------------------------------------------------------------------
#
#      ######     #######
#    ##     ##    ##    ##
#           ##    ##     ## 
#          ##     ##     ##
#        ##       ##     ##
#      ##         ##     ##
#    ##           ##    ##
#    ##########   #######
#
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# E M B E D D I N G 
# -------------------------------------------------------------------------------------


def embed_tsne_2D(Matrix, prplxty, density, l_rate, steps, metric = 'precomputed'):
    '''
    Dimensionality reduction from Matrix using t-SNE.
    Return dict (keys: node IDs, values: x,y).
    ''' 
    
    tsne = TSNE(n_components = 2, random_state = 0, perplexity = prplxty, metric = metric, init='pca',
                     early_exaggeration = density,  learning_rate = l_rate ,n_iter = steps)
    
    embed = tsne.fit_transform(Matrix)
    
    return embed


def embed_umap_2D(Matrix, n_neigh, spre, m_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix using UMAP.
    Return dict (keys: node IDs, values: x,y).
    ''' 
    n_comp = 2 

    U = umap.UMAP(
        n_neighbors = n_neigh,
        spread = spre,
        min_dist = m_dist,
        n_components = n_comp,
        metric = metric, 
        random_state=42,
        learning_rate = learn_rate, 
        n_epochs = n_ep,
        )
    
    embed = U.fit_transform(Matrix)
    
    return embed


def get_posG_2D(l_nodes, embed):
    '''
    Get 2D coordinates for each node.
    Return dict with node: x,y coordinates.
    '''
    
    posG = {}
    cc = 0
    for entz in l_nodes:
        # posG[str(entz)] = (embed[cc,0],embed[cc,1])
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    return posG



def get_posG_2D_norm(G, DM, embed, r_scalingfactor = 5):
    '''
    Generate coordinates from embedding. 
    Input:
    - G = Graph
    - DM = matrix; index and columns must be same as G.nodes
    - embed = embedding from e.g. tSNE , UMAP ,... 
    
    Return dictionary with nodes as keys and coordinates as values in 3D normed. 
    '''
    
    genes = []
    for i in DM.index:
        if str(i) in G.nodes() or int(i) in G.nodes():
            genes.append(i)

    genes_rest = [] 
    for i in G.nodes():
        if i not in genes:
            genes_rest.append(i)

    #print(len(genes))
    #print(len(genes_rest))
        
    posG = {}
    cc = 0
    for entz in genes:
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    #--------------------------------------------------------------
    # REST (if genes = G.nodes then rest will be ignored / empty)
    
    # generate circle coordinates for rest genes (without e.g. GO term or Disease Annotation)
    t = np.random.uniform(0,2*np.pi,len(genes_rest))
    
    xx=[]
    yy=[]
    for i in posG.values():
        xx.append(i[0])
        yy.append(i[1])
    
    cx = np.mean(xx)
    cy = np.mean(yy)

    xm, ym = max(posG.values())
    r = (math.sqrt((xm-cx)**2 + (ym-cy)**2))*r_scalingfactor #*1.05 # multiplying with 1.05 makes cirle larger to avoid "outsider nodes/genes"
        
    x = r*np.cos(t)
    y = r*np.sin(t)
    rest = []
    for i,j in zip(x,y):
            rest.append((i,j))

    posG_rest = dict(zip(genes_rest, rest))

    posG_all = {**posG, **posG_rest}
    
    #G_nodes_str = [str(i) for i in list(G.nodes())]
    posG_complete = {key:posG_all[key] for key in list(G.nodes())}

    # normalize coordinates 
    x_list = []
    y_list = []
    for k,v in posG_complete.items():
        x_list.append(v[0])
        y_list.append(v[1])

    xx_norm = preprocessing.minmax_scale(x_list, feature_range=(0, 1), axis=0, copy=True)
    yy_norm = preprocessing.minmax_scale(y_list, feature_range=(0, 1), axis=0, copy=True)

    xx_norm_final=[]
    for i in xx_norm:
        xx_norm_final.append(round(i,10))

    yy_norm_final=[]
    for i in yy_norm:
        yy_norm_final.append(round(i,10))

    posG_complete_norm = dict(zip(list(G.nodes()),zip(xx_norm_final,yy_norm_final)))
    
    return posG_complete_norm




def labels2D(posG, feature_dict):
    '''
    Create Node Labels, based on a dict of coordinates (keys:node ID, values: x,y)
    Return new dict of node iDs and features for each node.
    '''

    labels = {} 
    c = 0
    for node, xy in sorted(posG.items(), key = lambda x: x[1][0]):
        labels[node] = ([node,feature_dict[node][0],feature_dict[node][1],feature_dict[node][2],feature_dict[node][3]])   
        c+=1
        
    return labels


def position_labels(posG, move_x, move_y):
    '''
    Create label position of coordinates dict.
    Return new dict with label positions. 
    '''    
    
    posG_labels = {}
    for key,val in posG.items():
        xx = val[0] + move_x
        yy = val[1] + move_y
        posG_labels[key] = (xx,yy)
        
    return posG_labels



# -------------------------------------------------------------------------------------
#
#      ######     #######
#    ##     ##    ##    ##
#           ##    ##     ## 
#      #####      ##     ##
#           ##    ##     ##
#    ##     ##    ##    ##
#     ######      #######
#    
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# E M B E D D I N G 
# -------------------------------------------------------------------------------------

def embed_tsne_3D(Matrix, prplxty, density, l_rate, n_iter, metric = 'cosine'):
    '''
    Dimensionality reduction from Matrix (t-SNE).
    Return dict (keys: node IDs, values: x,y,z).
    '''
    
    tsne3d = TSNE(n_components = 3, random_state = 0, perplexity = prplxty,
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric)
    embed = tsne3d.fit_transform(Matrix)

    return embed 


def embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric='cosine', learn_rate = 1, n_ep = None):
    '''
    Dimensionality reduction from Matrix (UMAP).
    Return dict (keys: node IDs, values: x,y,z).
    '''

    n_components = 3 # for 3D

    U_3d = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric,
        random_state=42,
        learning_rate = learn_rate, 
        n_epochs = n_ep)
    embed = U_3d.fit_transform(Matrix)
    
    return embed


def get_posG_3D(l_genes, embed):
    '''
    Generate coordinates from embedding. 
    Input:
    - l_genes = list of genes
    - embed = embedding from e.g. tSNE , UMAP ,... 
    
    Return dictionary with nodes as keys and coordinates as values in 3D. 
    '''
    
    posG = {}
    cc = 0
    for entz in l_genes:
        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1
    
    return posG



def get_posG_3D_norm(G, DM, embed, r_scalingfactor=1.05):
    '''
    Generate coordinates from embedding. 
    Input:
    - G = Graph
    - DM = matrix 
    - embed = embedding from e.g. tSNE , UMAP ,... 
    
    Return dictionary with nodes as keys and coordinates as values in 3D normed. 
    '''

    genes = []
    for i in DM.index:
        if str(i) in G.nodes() or int(i) in G.nodes():
            genes.append(i)

    genes_rest = [] 
    for i in G.nodes():
        if i not in genes:
            genes_rest.append(i)
            
    posG_3Dumap = {}
    cc = 0
    for entz in genes:
        posG_3Dumap[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1

    #--------------------------------------------------------------
    # REST (if genes = G.nodes then rest will be ignored / empty)
    
    # center for sphere to arrange rest gene-datapoints
    xx=[]
    yy=[]
    zz=[]
    for i in posG_3Dumap.values():
        xx.append(i[0])
        yy.append(i[1])
        zz.append(i[2]) 

    cx = sum(xx)/len(genes)
    cy = sum(yy)/len(genes)
    cz = sum(zz)/len(genes)

    # generate spherical coordinates for rest genes (without e.g. GO term or Disease Annotation)
    indices = arange(0, len(genes_rest))
    phi = arccos(1 - 2*indices/len(genes_rest)) # 2* --> for both halfs of sphere (upper+lower)
    theta = pi * (1 + 5**0.5) * indices

    xm, ym, zm = max(posG_3Dumap.values())
    r = (math.sqrt((cx - xm)**2 + (cy - ym)**2 + (cz - zm)**2))*r_scalingfactor # +10 > ensure colored nodes within sphere
    x, y, z = cx+r*cos(theta) * sin(phi),cy+r*sin(theta) * sin(phi), cz+r*cos(phi)

    rest_points = []
    for i,j,k in zip(x,y,z):
        rest_points.append((i,j,k))

    posG_rest = dict(zip(genes_rest, rest_points))

    posG_all = {**posG_3Dumap, **posG_rest}
    posG_3D_complete_umap = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates 
    x_list3D = []
    y_list3D = []
    z_list3D = []
    for k,v in posG_3D_complete_umap.items():
        x_list3D.append(v[0])
        y_list3D.append(v[1])
        z_list3D.append(v[2])

    xx_norm3D = preprocessing.minmax_scale(x_list3D, feature_range=(0, 1), axis=0, copy=True)
    yy_norm3D = preprocessing.minmax_scale(y_list3D, feature_range=(0, 1), axis=0, copy=True)
    zz_norm3D = preprocessing.minmax_scale(z_list3D, feature_range=(0, 1), axis=0, copy=True)

    xx_norm3D_final=[]
    for i in xx_norm3D:
        xx_norm3D_final.append(round(i,10))

    yy_norm3D_final=[]
    for i in yy_norm3D:
        yy_norm3D_final.append(round(i,10))

    zz_norm3D_final=[]
    for i in zz_norm3D:
        zz_norm3D_final.append(round(i,10)) 

    posG_3D_complete_umap_norm = dict(zip(list(G.nodes()), zip(xx_norm3D_final,yy_norm3D_final,zz_norm3D_final)))
    
    return posG_3D_complete_umap_norm



def embed_umap_sphere(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
    ''' 
    Generate spherical embedding of nodes in matrix input using UMAP.
    Input: 
    - Matrix = Feature Matrix with either all or specific  nodes (rows) and features (columns) or symmetric (nodes = rows and columns)
    - n_neighbors/spread/min_dist = floats; UMAP parameters.
    - metric = string; e.g. havervine, euclidean, cosine ,.. 
    
    Return sphere embedding. 
    '''
    
    model = umap.UMAP(
        n_neighbors = n_neighbors, 
        spread = spread,
        min_dist = min_dist,
        metric = metric)

    sphere_mapper = model.fit(Matrix)

    return sphere_mapper



def get_posG_sphere(l_genes, sphere_mapper):
    '''
    Generate coordinates from embedding. 
    Input:
    - l_genes = list of genes
    - sphere_mapper = embedding from UMAP spherical embedding 
    
    Return dictionary with nodes as keys and coordinates as values in 3D. 
    '''
    
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])
    
    posG = {}
    cc = 0
    for entz in l_genes:
        posG[entz] = (x[cc],y[cc], z[cc])
        cc += 1
    
    return posG


def get_posG_sphere_norm(G, l_genes, sphere_mapper, d_param, radius_rest_genes = 20):
    '''
    Generate coordinates from embedding. 
    Input:
    - G = Graph 
    - l_genes = list of node IDs, either specific or all nodes in the graph 
    - sphere_mapper = embedding from UMAP spherical embedding 
    - d_param = dictionary with nodes as keys and assigned radius as values 
    - radius_rest_genes = int; radius in case of genes e.g. not function associated if genes not all G.nodes()
    
    Return dictionary with nodes as keys and coordinates as values in 3D. 
    '''
    
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])
    
    genes = []
    for i in l_genes:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)
            
    posG_3Dsphere = {}
    cc = 0
    for entz in genes:
        posG_3Dsphere[entz] = (x[cc],y[cc], z[cc])
        cc += 1

    posG_3Dsphere_radius = {}
    for node,rad in d_param.items():
        for k,v in posG_3Dsphere.items():
            if k == node:
                posG_3Dsphere_radius[k] = (v[0]*rad, v[1]*rad, v[2]*rad)
 
    # generate spherical coordinates for rest genes (without e.g. GO term or Disease Annotation)
    indices = arange(0, len(genes_rest))
    phi = arccos(1 - 2*indices/len(genes_rest))
    theta = pi * (1 + 5**0.5) * indices

    r_rest = radius_rest_genes # radius for rest genes (e.g. if functional layout)
    x, y, z = r_rest*cos(theta) * sin(phi), r_rest*sin(theta) * sin(phi), r_rest*cos(phi)

    rest_points = []
    for i,j,k in zip(x,y,z):
        rest_points.append((i,j,k))

    posG_rest = dict(zip(genes_rest, rest_points))

    posG_all = {**posG_3Dsphere_radius, **posG_rest}
    posG_complete_sphere = {key:posG_all[key] for key in G.nodes()}

    # normalize coordinates 
    x_list = []
    y_list = []
    z_list = []
    for k,v in posG_complete_sphere.items():
        x_list.append(v[0])
        y_list.append(v[1])
        z_list.append(v[2])

    xx_norm = sklearn.preprocessing.minmax_scale(x_list, feature_range=(0, 1), axis=0, copy=True)
    yy_norm = sklearn.preprocessing.minmax_scale(y_list, feature_range=(0, 1), axis=0, copy=True)
    zz_norm = sklearn.preprocessing.minmax_scale(z_list, feature_range=(0, 1), axis=0, copy=True)

    posG_complete_sphere_norm = dict(zip(list(G.nodes()), zip(xx_norm,yy_norm,zz_norm)))
    
    return posG_complete_sphere_norm



########################################################################################
#
# F U N C T I O N S   F O R   B E N C H M A R K I N G   S C R I P T S
# 
########################################################################################


def rnd_walk_matrix2(A, r, a, num_nodes):
    '''
    Random Walk Operator with restart probability.
    Input: 
    - A = Adjanceny matrix (numpy array)
    - r = restart parameter e.g. 0.9
    - a = teleportation value e.g. 1.0 for max. teleportation
    - num_nodes = all nodes included in Adjacency matrix, e.g. amount of all nodes in the graph 

    Return Matrix with visiting probabilites (non-symmetric!!).
    ''' 
    n = num_nodes
    factor = float((1-a)/n)

    E = np.multiply(factor,np.ones([n,n]))              # prepare 2nd scaling term
    A_tele = np.multiply(a,A) + E  #     print(A_tele)
    M = normalize(A_tele, norm='l1', axis=0)                                 # column wise normalized MArkov matrix

    # mixture of Markov chains
    del A_tele
    del E

    U = np.identity(n,dtype=int) 
    H = (1-r)*M
    H1 = np.subtract(U,H)
    del U
    del M
    del H    

    W = r*np.linalg.inv(H1)   

    return W


def calc_dist_2D(posG):
    '''
    Validation of Layouts 2D. Calculates distances from layout.
    Return list with distances. 
    '''
    l_x= []
    l_y=[]
    for coords in posG.values():
            l_x.append(coords[0])
            l_y.append(coords[1])
            
    p_dist = []
    for idx,val in enumerate(l_x):
        d_list = []
        for c in range(len(l_x)):
            for yy in l_y:
                d = np.sqrt((l_x[idx]-l_x[c])**2+(l_y[idx]-l_y[c])**2)
            d_list.append(d)
        p_dist.append(d_list)
        
    return p_dist


def calc_dist_3D(posG):
    '''
    Validation of Layouts 3D. Calculates distances from layout.
    Return list with distances. 
    '''
    
    l_x = []
    l_y = []
    l_z = []
    for coords in posG.values():
            l_x.append(coords[0])
            l_y.append(coords[1])
            l_z.append(coords[2])
            
    p_dist = []
    for idx,val in enumerate(l_x):
        d_list = []
        for c in range(len(l_x)):
            d = np.sqrt((l_x[idx]-l_x[c])**2+(l_y[idx]-l_y[c])**2+(l_z[idx]-l_z[c])**2)
        d_list.append(d)
    p_dist.append(d_list)
        
    return p_dist


def get_trace_xy(x,y,trace_name,colour):
    '''
    Get trace 2D.
    Used for distance functions (2D; benchmarking) 
    '''    
    
    trace = pgo.Scatter(
        name = trace_name,
    x = x,
    y = y,
    mode='markers',
    marker=dict(
        size=2,
        color=colour
    ),)
    return trace


def get_trace_xyz(x,y,z,trace_name,colour):
    '''
    Generate 3D trace. 
    Used for distance functions (3D; benchmarking)
    '''    
    
    trace = pgo.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode='markers',
        text=trace_name,
        marker=dict(
            size=2,
            color=colour, 
            line_width=0.5,
            line_color = colour,
        ),)
    return trace


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 3)), int(round(m, 0)), int(round(h, 0))
    print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
   
    return m,s


def globallayout_2D(G,n_neighbors, spread, min_dist, metric):
    
    r=0.9
    alpha=1.0

    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM_rwr = pd.DataFrame(FM_m_array).T

    umap_rwr_2D = embed_umap_2D(DM_rwr, n_neighbors, spread, min_dist, metric)
    posG_umap_rwr = get_posG_2D(list(G.nodes()), umap_rwr_2D)
    posG_complete_umap_rwr = {key:posG_umap_rwr[key] for key in G.nodes()}

    df_posG = pd.DataFrame(posG_complete_umap_rwr).T
    x = df_posG.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)

    posG_complete_umap_rwr_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))
    
    del DM_rwr
    del df_posG
    
    return posG_complete_umap_rwr_norm


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


def globallayout_3D(G,n_neighbors, spread, min_dist, metric):
    
    r=0.9
    alpha=1.0
    
    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM_rwr = pd.DataFrame(FM_m_array).T

    umap_rwr_3D = embed_umap_3D(DM_rwr, n_neighbors, spread, min_dist, metric)
    posG_umap_rwr = get_posG_3D(list(G.nodes()), umap_rwr_3D)
    posG_complete_umap_rwr = {key:posG_umap_rwr[key] for key in G.nodes()}

    df_posG = pd.DataFrame(posG_complete_umap_rwr).T
    x = df_posG.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)

    posG_complete_umap_rwr_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values,df_posG_norm[2].values)))
    
    del DM_rwr
    del df_posG
    
    return posG_complete_umap_rwr_norm


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



########################################################################################
#
# P A I R W I S E   E U C L I D E A N   D I S T A N C E   C A L C U L A T I O N
#
########################################################################################


def pairwise_layout_distance_linalg(pairs,posG):  
    dist_layout = {}
    print(len(pairs))
    for p1,p2 in pairs:
        start = np.array(posG[p1])
        end = np.array(posG[p2])
        dist_layout[(p1,p2)]=np.linalg.norm(start-end)
        
        tenk = 1000
        if len(dist_layout) == (tenk): 
            print('1k done')
        elif len(dist_layout) == (tenk*10): 
            print('10k done')
        elif len(dist_layout) == (tenk*50):
            print('50k done')
        elif len(dist_layout) == (tenk*100):
            print('100k done')
        elif len(dist_layout) == (tenk*500):
            print('500k done')
        elif len(dist_layout) == (tenk*1000):
            print('1mio done')
        elif len(dist_layout) == (tenk*5000):
            print('5mio done')
        elif len(dist_layout) == (tenk*10000):
            print('10mio done')
        elif len(dist_layout) == (tenk*15000):
            print('15mio done')
        elif len(dist_layout) == (tenk*20000):
            print('20mio done')
        elif len(dist_layout) == (tenk*25000):
            print('25mio done')
        elif len(dist_layout) == (tenk*30000):
            print('30mio done')
        elif len(dist_layout) == (tenk*40000):
            print('40mio done')
        elif len(dist_layout) == len(pairs):
            print('complete')
        else:
            pass
        
    return dist_layout


def pairwise_layout_distance_linalg_parts(pairs,posG):  
    dist_layout = {}
    print(len(pairs))
    for p1,p2 in pairs:
        start = np.array(posG[p1])
        end = np.array(posG[p2])
        dist_layout[(p1,p2)]=np.linalg.norm(start-end)
        
        tenk = 1000
        if len(dist_layout) == (tenk): 
            print('1k done')
        elif len(dist_layout) == (tenk*10): 
            print('10k done')
        elif len(dist_layout) == (tenk*50):
            print('50k done')
        elif len(dist_layout) == (tenk*100):
            print('100k done')
        elif len(dist_layout) == (tenk*500):
            print('500k done')
        elif len(dist_layout) == (tenk*1000):
            print('1mio done')
        elif len(dist_layout) == (tenk*5000):
            print('5mio done')
        elif len(dist_layout) == (tenk*10000):
            print('10mio done')
        elif len(dist_layout) == (tenk*15000):
            print('15mio done')
        elif len(dist_layout) == (tenk*20000):
            print('20mio done')
        elif len(dist_layout) == (tenk*25000):
            print('25mio done')
        elif len(dist_layout) == (tenk*30000):
            print('30mio done')
        elif len(dist_layout) == (tenk*40000):
            print('40mio done')
        elif len(dist_layout) == len(pairs):
            print('complete')
        else:
            pass
        
    return dist_layout
            

def pairwise_network_distance(G):
    dist_network = {}
    print('total to calculate:',(len(list(it.combinations(G.nodes(),2)))))
                    
    for p1,p2 in it.combinations(G.nodes(),2):
        
        dist_network[(p1,p2)] = nx.shortest_path_length(G,p1,p2, method='dijkstra')  
        tenk = 1000
        if len(dist_network) == (tenk): 
            print('1k done')
        elif len(dist_network) == (tenk*10): 
            print('10k done')
        elif len(dist_network) == (tenk*50):
            print('50k done')
        elif len(dist_network) == (tenk*100):
            print('100k done')
        elif len(dist_network) == (tenk*500):
            print('500k done')
        elif len(dist_network) == (tenk*1000):
            print('1mio done')
        elif len(dist_network) == (tenk*5000):
            print('5mio done')
        elif len(dist_network) == (tenk*10000):
            print('10mio done')
        elif len(dist_network) == (tenk*15000):
            print('15mio done')
        elif len(dist_network) == (tenk*20000):
            print('20mio done')
        elif len(dist_network) == (tenk*25000):
            print('25mio done')
        elif len(dist_network) == (tenk*30000):
            print('30mio done')
        elif len(dist_network) == (tenk*40000):
            print('40mio done')
        elif len(dist_network) == len(list(it.combinations(G.nodes(),2))):
            print('complete')
        else:
            pass
    return dist_network   


def pairwise_network_distance_parts(G,pairs):
    dist_network = {}
    print('total to calculate:',(len(pairs)))
                    
    for p1,p2 in pairs:
        
        dist_network[(p1,p2)] = nx.shortest_path_length(G,p1,p2, method='dijkstra')  
        hund=100
        tenk = 1000
        if len(dist_network) == hund:
            print('100 done')  
        elif len(dist_network) == tenk: 
            print('1k done')
        elif len(dist_network) == (tenk*10): 
            print('10k done')
        elif len(dist_network) == (tenk*50):
            print('50k done')
        elif len(dist_network) == (tenk*100):
            print('100k done')
        elif len(dist_network) == (tenk*500):
            print('500k done')
        elif len(dist_network) == (tenk*1000):
            print('1mio done')
        elif len(dist_network) == (tenk*5000):
            print('5mio done')
        elif len(dist_network) == (tenk*10000):
            print('10mio done')
        elif len(dist_network) == (tenk*15000):
            print('15mio done')
        elif len(dist_network) == (tenk*20000):
            print('20mio done')
        elif len(dist_network) == (tenk*25000):
            print('25mio done')
        elif len(dist_network) == (tenk*30000):
            print('30mio done')
        elif len(dist_network) == (tenk*40000):
            print('40mio done')
        elif len(dist_network) == len(pairs):
            print('complete')
        else:
            pass
    return dist_network    


def pearson_corrcoef(dist_network, dist_layout):
    
    d_plot_layout = {}
    for spldist in range(1,int(max(dist_network.values()))+1):
        l_s = []
        for k, v in dist_network.items():
            if v == spldist:
                l_s.append(k)

        l_xy = []
        for nodes in l_s:
            try:
                dxy = dist_layout[nodes]
                l_xy.append(dxy)
            except:
                pass
        d_plot_layout[spldist] = l_xy
    
    print('done layout distances prep')
    l_medians_layout = []
    for k, v in d_plot_layout.items():
        l_medians_layout.append(statistics.median(v))
    
    print('calculate pearson correlation coefficient')
    x = np.array(range(1,int(max(dist_network.values()))+1))
    y = np.array(l_medians_layout)
    r_layout = np.corrcoef(x, y)
    
    return r_layout[0][1]



#################

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
        

        
def minmaxscaling_posG(G,posG):
    df_posG = pd.DataFrame(posG).T
    x = df_posG.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    
    df_posG_norm = pd.DataFrame(x_scaled)
    
    posG_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))

    return posG_norm 

