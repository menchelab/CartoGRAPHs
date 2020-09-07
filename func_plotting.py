# ============================================
# PLOTTING 
# ============================================

import ast 

from collections import (defaultdict,Counter)
from collections import defaultdict as dd
from collections import Counter as ct
from collections import OrderedDict
import colorsys
from colormap import rgb2hex, rgb2hls, hls2rgb
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from collections import Counter

#from dosnes import dosnes

from fisher import pvalue
from fa2 import ForceAtlas2

import itertools as it

from matplotlib import colors as mcolors
import math
from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
import mygene

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.generators.degree_seq import expected_degree_graph
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import numpy.linalg as la
import numba
#@numba.njit(fastmath=True)

import os
import os.path

import pandas as pd

import pickle
import plotly
import plotly.express as px
import plotly.graph_objs as pgo
import plotly.offline as py
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio

from prettytable import PrettyTable

import pylab

#py.init_notebook_mode(connected = True)

# import pymysql as mysql

import random as rd

from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
import scipy.stats as st
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
from scipy.spatial import distance_matrix
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection,cluster)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import sys 

from tqdm import tqdm_notebook as tqdm
import time

import umap 

import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------------------
# FUNCTIONS FOR 2D PORTRAITS
# -------------------------------------------------------------------------------------

'''
Dimensionality reduction from Matrix (t-SNE).
Return dict (keys: node IDs, values: x,y).
'''
def embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric = 'precomputed'):
    
    tsne = TSNE(n_components = 2, random_state = 0, perplexity = prplxty, metric = metric, init='pca',
                     early_exaggeration = density,  learning_rate = l_rate ,n_iter = steps)
    embed = tsne.fit_transform(DM)
    return embed



'''
Dimensionality reduction from Matrix (UMAP).
Return dict (keys: node IDs, values: x,y).
'''
def embed_umap_2D(DM, n_neighbors, spread, min_dist, metric):

    U = umap.UMAP(
            n_neighbors = n_neighbors,
            spread = spread,
            min_dist = min_dist,
            n_components = 2,
            metric = metric)
    embed = U.fit_transform(DM)
    return embed









'''
Get 2D coordinates for each node.
Return dict with node: x,y coordinates.
''' 
def get_posG(G, embed):
    posG = {}
    cc = 0
    for entz in G.nodes():
    #for entz in sorted(G.nodes()):
        posG[entz] = (embed[cc,0],embed[cc,1])
        cc += 1

    return posG


def get_posG_2D(l_nodes, embed):
    posG = {}
    cc = 0
    for entz in l_nodes:
        posG[str(entz)] = (embed[cc,0],embed[cc,1])
        cc += 1

    return posG

'''
Create Node Labels, based on a dict of coordinates (keys:node ID, values: x,y)
Return new dict of node iDs and features for each node.
'''
def labels2D(posG, feature_dict):
    labels = {} 
    c = 0
    for node, xy in sorted(posG.items(), key = lambda x: x[1][0]):
        labels[node] = ([node,feature_dict[node][0],feature_dict[node][1],feature_dict[node][2],feature_dict[node][3]])   
        c+=1
        
    return labels


'''
Create label position of coordinates dict.
Return new dict with label positions. 
'''
def position_labels(posG, move_x, move_y):
    posG_labels = {}
    for key,val in posG.items():
        xx = val[0] + move_x
        yy = val[1] + move_y
        posG_labels[key] = (xx,yy)
        
    return posG_labels


'''
Validation of Layouts 2D. Calculates distances from layout.
Return list with distances. 
'''
def calc_dist_from_layout(posG):
    
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


''' 
Generate trace 2D.
Return trace. 
'''
def get_trace_2D(posG, colour, size):
    
    l_x=[]
    l_y=[]
    for coords in posG.values():
            l_x.append(coords[0])
            l_y.append(coords[1])
    
    trace = pgo.Scatter(
        x = l_x,
        y = l_y,
        mode='markers',
        marker=dict(
            size=size,
            color=colour
        ),)
    return trace


def get_trace_edges_2D(G, posG, color_list):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = posG[edge[0]]
        x1, y1 = posG[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = pgo.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color=color_list),
        hoverinfo='none',
        mode='lines',
    opacity = 0.3)
    
    return edge_trace

# -------------------------------------------------------------------------------------
# FUNCTIONS FOR 3D GENERAL 
# -------------------------------------------------------------------------------------


'''
Function to make 3D html plot rotating.
Returns frames, to be used in "pgo.Figure(frames = frames)"
'''
def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z 

def export_xyz_to_csv(G, posG, title):
    coords = pd.DataFrame(posG, index = ['X','Y','Z'], columns = G.nodes()).T
    #coords['Gene ID'] = genes_subset
    #coords['Cluster ID'] = cluster_community
    
    return coords.to_csv(title +'.csv')

def get_trace_spec_edges_only(l_spec_edge, posG, color_list):
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in l_spec_edge:
            x0, y0, z0 = posG[edge[0]]
            x1, y1, z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)
    
        
    trace_edges = pgo.Scatter3d(
                        x = edge_x, 
                        y = edge_y, 
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.5, color = color_list),
                        opacity = 0.3
                )
    
    return trace_edges




# -------------------------------------------------------------------------------------
# FUNCTIONS FOR LANDSCAPES
# -------------------------------------------------------------------------------------

'''
From 2D embedded coordinates generate a 3D landscape by including the z-axis.
Return x,y,z coordinates, where z is set 0.
'''
def get_coords_landscape(G, posG):
    # IMPORTANT : sort pos by G.nodes()
    posG_sorted = dict([(key, posG[key]) for key in G.nodes()])

    df=pd.DataFrame(posG_sorted).T
    df['Z'] = 0*len(G.nodes())
    df.columns=['X','Y','Z']

    x=np.array(df['X'])
    y=np.array(df['Y'])
    z=np.array(df['Z'])

    return x,y,z


'''
Create trace of nodes for x,y,z0 and x,y,z=parameter (e.g.disease count).
Return traces of nodes.

'''   
def get_trace_nodes_landscape(x,y,z, colours, size3d):

    trace_z = pgo.Scatter3d(x=x,y=y,z=z,
                             mode = 'markers',
                           #text = list(d_gene_dc_sorted.items()),
                           #hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(                       
                color = colours,
                size = size3d,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = colours)
                           ),)
    
    return trace_z


'''
Create trace of vertical connecting edges in between node z0 and node z=parameter (e.g.disease count).
Return trace with edges.
'''

def get_trace_edges_landscape(x,y,z0,z):
    Xe = []
    for u in x:
        Xe += [u,u,None]

    Ye = []   
    for v in y:
        Ye += [v,v,None]  

    Ze = []  
    for w in z0:
        for t in z:
            Ze += [w,t,None]
            
    trace_edge = pgo.Scatter3d(
        x = Xe, 
        y = Ye, 
        z = Ze,
        mode = 'lines', hoverinfo='none',
        line = dict(width = 3.0, color = 'darkgrey'),
        opacity = 0.5
    )

    return trace_edge




# -------------------------------------------------------------------------------------
# FUNCTIONS FOR 3D PORTRAITS 
# -------------------------------------------------------------------------------------


'''
Dimensionality reduction from Matrix (t-SNE).
Return dict (keys: node IDs, values: x,y,z).
'''
def embed_tsne_3D(Matrix, prplxty, density, l_rate, n_iter, metric = 'precomputed'):
    tsne3d = TSNE(n_components = 3, random_state = 0, perplexity = prplxty,
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric)
    embed = tsne3d.fit_transform(Matrix)

    return embed

def get_posG_3D(G, embed):
    posG = {}
    cc = 0
    for entz in sorted(G.nodes()):
        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1
    posG_sorted = {key:posG[key] for key in G.nodes()}
    
    return posG_sorted

#def get_posG_3D_(l_genes, embed):
#    posG = {}
#    cc = 0
#    for entz in l_genes:
#        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
#        cc += 1
#    
#    return posG


'''
Dimensionality reduction from Matrix (UMAP).
Return dict (keys: node IDs, values: x,y,z).
'''
def embed_umap_3D(G, Matrix, n_neighbors, spread, min_dist, metric='cosine' ):
    n_components = 3 # for 3D

    U_3d = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric)
    embed = U_3d.fit_transform(Matrix)

    posG = {}
    cc = 0
    for entz in sorted(G.nodes()):
        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1
    posG_sorted = {key:posG[key] for key in G.nodes()}
    
    return posG_sorted



'''
Get node coordinates based on distribution on sphere layers (different radii) given by a dictionary with all nodes having specific values based on a function.
Return a dictionary  with node ids (keys) and node positions with assigned radius (values).
'''
def get_posG_with_sphere_radius(G, posG, d_param):
    l_dict = []
    for i in set(d_param.values()):
        dsub_nodes_score = {}
        for node, score in d_param.items():
            for n, c in posG.items():
                if score==i:
                    dsub_nodes_score[node] = score
        l_dict.append(dsub_nodes_score)
    
    radius_assigned = []
    for dct in l_dict:
        posG_withrad = {}
        for n,score in dct.items():
            for node,coords in posG.items():
                if n==node:
                    posG_withrad[n] = coords
        radius_assigned.append(posG_withrad)
    
    l_trace_prep = []
    for idx,dct in enumerate(radius_assigned):
        posG_new = {}
        for ir,rad in enumerate(range(1,len(set(d_param.values()))+1)):
            for n,c in dct.items():
                if ir == idx:
                    posG_new[n] = (c[0]*rad, c[1]*rad, c[2]*rad)
        l_trace_prep.append(posG_new) 
        
    posG_all = {}
    for i in l_trace_prep:
        posG_all.update(i)

    posG_all_sorted = {key:posG_all[key] for key in G.nodes()}
    
    return posG_all_sorted



'''
Plot node features within 3D plot. 
Return trace.
'''
def get_node_features(posG, features):
    
    key_list=list(posG.keys())
    node_labels = pgo.Scatter3d(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                               y=[posG[key_list[i]][1] for i in range(len(key_list))],
                               z=[posG[key_list[i]][2] for i in range(len(key_list))],
                                text = features,
                                mode='text', 
                                textfont=dict(
                                    size=10,
                                    color='silver'
                                ),
                                textposition='top center',
                                hoverinfo='none'
                               )
    return node_labels


'''
Generates 3D coordinates from nodes (dict).
Return trace.
'''
# former : "get_trace_nodes"
def get_trace_nodes_from_graph(G, posG, info_list, color_list, size):

    key_list=list(G.nodes())
    trace = pgo.Scatter3d(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           z=[posG[key_list[i]][2] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = 0.8,
                        color = color_list),
                               opacity=0.8,
            ),
        )
    
    return trace

# former : "get_trace_nodes_"
def get_trace_nodes(posG, info_list, color_list, size):

    key_list=list(posG.keys())
    trace = pgo.Scatter3d(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           z=[posG[key_list[i]][2] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color_list)
            ),
        )
    
    return trace


def get_trace_edges_2D(G, posG, color_list):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = posG[edge[0]]
        x1, y1 = posG[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = pgo.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=color_list),
        hoverinfo='none',
        mode='lines')
    
    return edge_trace


'''
Generates edges from 3D coordinates.
Returns a trace of edges.
'''

def get_trace_edges(G, posG, color_list):
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = posG[edge[0]]
        x1, y1, z1 = posG[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    trace_edges = pgo.Scatter3d(
                        x = edge_x, 
                        y = edge_y, 
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.5, color = color_list),
                        opacity = 0.3
                )
    
    return trace_edges



# -------------------------------------------------------------------------------------
# 3D SPHERES 
# -------------------------------------------------------------------------------------

'''
(DOSNES) The Nodes are embedded on a sphere.
Return dict with coordinates (x,y,z)
'''
def embed_tsne_sphere(G, Matrix, momentum, final_momentum, learning_rate, min_gain, max_iter, metric = 'precomputed', rand_state = 42, verb=1):
    model = dosnes.DOSNES(momentum = momentum, final_momentum = final_momentum, learning_rate = learning_rate, min_gain = min_gain,
    max_iter = max_iter, verbose_freq = 10, metric = metric, verbose = verb, random_state=rand_state)
    X_tsne_sphere = model.fit_transform(Matrix)
    
    posG = {}
    cc = 0
    for entz in sorted(G.nodes()):
        posG[entz] = (X_tsne_sphere[cc,0],X_tsne_sphere[cc,1], X_tsne_sphere[cc,2])
        cc += 1
    
    posG_sorted = {key:posG[key] for key in G.nodes()}

    return posG_sorted

'''
Get respective sphere to locate a node based on a parameter dictionary (dict_param) chosen.
Return dict with Node ids (keys) and respective sphere radius assigned based on dict_param.
'''

def assign_radius_to_nodes(G, dict_param):  

    l_rad = []
    count = 1
    for i in set(sorted(dict_param.values())):
        l_rad.append(count)
        count+=1
    
    # bin nodes by dict_param
    # get a dict: radius as keys and dict_param as values 

    d_z = {}
    for k, g in it.groupby(dict_param.values(), key=lambda n:n):
        d_z[k] = list(g)

    d_rad = {}
    for idx,rad in enumerate(l_rad):
        for i,k in enumerate(d_z.keys()):
            if idx == i:
                d_rad[rad] = k

    # assign each node id to a radius based on dict_param 

    d_node_rad = {}
    for k,v in dict_param.items():
        for r, deg in d_rad.items():
            if v == deg:
                d_node_rad[k] = r
    
    d_node_rad_sorted = {key:d_node_rad[key] for key in G.nodes()}
    return d_node_rad_sorted


'''
(DOSNES) From embedded spherical coordinates for each node generate a trace.
Return traces for nodes.
'''
def get_tsne_sphere_trace_nodes(posG, d_node_rad, color_list, size):

    # multiply radius to specific node coordinate based on dict_param bin 
    d_node_coords_trace = {}
    for node,coord in posG.items():
        for nd, rad in d_node_rad.items():
            if node == nd:
                d_node_coords_trace[node] = (rad*coord[0],rad*coord[1],rad*coord[2])
                
    # prep coordinates incl. respective radius for generating a trace  
    x_trace = []
    y_trace = []
    z_trace = []
    for k,v in d_node_coords_trace.items():
        x_trace.append(v[0])
        y_trace.append(v[1])
        z_trace.append(v[2])
        
    # traces nodes
    trace_nodes = pgo.Scatter3d(x = x_trace,
                           y = y_trace,
                           z = z_trace,
                           mode = 'markers',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
            ),
        )

    return trace_nodes


'''
Generate a trace for 3D plotting a sphere (in the background) with chosen radius.
Return list of traces of spheres with respective radii given by design.
'''
def get_sphere_background(radius_list):
    sphere_background = []
    for r in radius_list:
        n = 100j # resolution of sphere contours
        u, v = np.mgrid[0:pi:n, 0:2 * pi:n]

        x=r*np.cos(u)*np.sin(v) 
        y=r*np.sin(u)*np.sin(v) 
        z=r*np.cos(v) 
        sphere_trace = pgo.Surface(
                        x = x, 
                        y = y, 
                        z = z,
                        hoverinfo='skip',
                        contours = {'x': {'show':True, 'start':0, 'end':0, 'size':1, 'color' : 'lightgray'},
                                    'y': {'show':True, 'start':0, 'end':0, 'size':1, 'color' : 'lightgray'},
                                    'z': {'show':True, 'start':0, 'end':0, 'size':1, 'color' : 'lightgray'}
                                }
                        , showscale = False
                        , opacity = 0)
        sphere_background.append(sphere_trace)

    return sphere_background


'''
(UMAP) The Nodes are embedded on a sphere.
Return dict with coordinates (x,y,z)
'''
def embed_umap_sphere(G, Matrix, metric):
    model = umap.UMAP(output_metric=metric)
    sphere_mapper = model.fit(Matrix)
    
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])
    
    posG = {}
    cc = 0
    for entz in sorted(G.nodes()):
        posG[entz] = (x[cc],y[cc], z[cc])
        cc += 1
    
    posG_sorted = {key:posG[key] for key in G.nodes()}

    return posG_sorted


'''
(UMAP) From embedded spherical coordinates for each node generate a trace.
Return traces for nodes.
'''
def get_trace_umap_sphere(posG, info_list, color_list, size3d):
    x=[]
    y=[]
    z=[]
    for coords in posG.values():
        xx=coords[0]
        yy=coords[1]
        zz=coords[2]
        x.append(xx)
        y.append(yy)
        z.append(zz)

    sphere_trace = pgo.Scatter3d( x=x,
                           y=y,
                           z=z,
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           marker = dict(
                color = color_list,
                size = size3d,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color_list)
            ),
        )
    
    return sphere_trace


# -------------------------------------------------------------------------------------
# PLOTLY TRACE : 3D TORUS
# -------------------------------------------------------------------------------------


def torus_euclidean_grad(x, y, torus_dimensions=(2*np.pi,2*np.pi)):
    distance_sqr = 0.0
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        a = abs(x[i] - y[i])
        if 2*a < torus_dimensions[i]:
            distance_sqr += a ** 2
            g[i] = (x[i] - y[i])
        else:
            distance_sqr += (torus_dimensions[i]-a) ** 2
            g[i] = (x[i] - y[i]) * (a - torus_dimensions[i]) / a
    distance = np.sqrt(distance_sqr)
    return distance, g/(1e-6 + distance)


def get_trace_umap_torus(torus_embedded, r, R, color_list, size3d):
    R = 3 # Size of the doughnut circle
    r = 1 # Size of the doughnut cross-section

    x = (R + r * np.cos(torus_embedded.embedding_[:, 0])) * np.cos(torus_embedded.embedding_[:, 1])
    y = (R + r * np.cos(torus_embedded.embedding_[:, 0])) * np.sin(torus_embedded.embedding_[:, 1])
    z = r * np.sin(torus_embedded.embedding_[:, 0])
    
    torus_trace = pgo.Scatter3d( x=x,
                           y=y,
                           z=z,
                           mode = 'markers',
                           marker = dict(
                color = color_list,
                size = size3d,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = color_list)
            ),
        )
    return torus_trace


# --------------------------------------------
# PLOTLY TRACE : 3D
# --------------------------------------------

def get_trace_edges_from_genelist(l_spec_edges, posG, color_list):
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in l_spec_edges:
            x0, y0, z0 = posG[edge[0]]
            x1, y1, z1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)
        
    trace_edges = pgo.Scatter3d(
                        x = edge_x, 
                        y = edge_y, 
                        z = edge_z,
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 1.5, color = color_list),
                        opacity = 0.5
                )
    
    return trace_edges

