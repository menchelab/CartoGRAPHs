
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
from colormap import rgb2hex, rgb2hls, hls2rgb
from colormath.color_objects import sRGBColor, LabColor
#from colormath.color_conversions import convert_color

from fisher import pvalue
from fa2 import ForceAtlas2

from html2image import Html2Image

import itertools as it

import math
import matplotlib.pyplot as plt
import numpy.linalg as la
#%matplotlib inline
import multiprocessing
import mygene

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.generators.degree_seq import expected_degree_graph
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
from numpy import pi, cos, sin, arccos, arange
import numpy.linalg as la
import numba
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

import sys 

import time

import umap 

import warnings
#warnings.filterwarnings("ignore", category=UserWarning)


########################################################################################
#
# F U N C T I O N S   F O R   A N A L Y S I S + C A L C U L A T I O N S
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
    
    num = 1*num_nodes
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


def bin_nodes(data_dict): 
    '''
    Binning nodes based on unique values in dictionary input. 
    Input: 
    - data_dict = dictionary with node id as keys and values of e.g. degree centrality.
    
    Return binned nodes.
    '''
    
    bins = set(data_dict.values())

    d_binned = {}
    for n in bins:
        d_binned[n]=[str(k) for k in data_dict.keys() if data_dict[k] == n]
        
    return d_binned


def rotate_z(x, y, z, theta):
    '''
    Function to make 3D html plot rotating.
    Returns frames, to be used in "pgo.Figure(frames = frames)"
    '''
    
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z 


# -------------------------------------------------------------------------------------
# B E N C H M A R K I N G  specific
# -------------------------------------------------------------------------------------

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


########################################################################################
#
# C O L O R F U N C T I O N S 
#
########################################################################################


def generate_colorlist_nodes(n):
    '''
    Generate color list based on color count (i.e. nodes to be coloured).
    Input:
    - n = number of colors to generate.
    
    Return list of colors.
    '''
    
    colors = [colorsys.hsv_to_rgb(1.0/n*x,1,1) for x in range(n)]
    color_list = []
    for c in colors:
        cc = [int(y*255) for y in c]
        color_list.append('#%02x%02x%02x' % (cc[0],cc[1],cc[2]))
        
    return color_list


def hex_to_rgb(hx):
    hx = hx.lstrip('#')
    hlen = len(hx)
    return tuple(int(hx[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))


def darken_color(r, g, b, factor=0.9):
    return adjust_color_lightness(r, g, b, 1 - factor)


def color_nodes_from_dict_unsort(d_to_be_coloured, palette):
    ''' 
    Generate node colors based on dictionary.
    Input: 
    - d_to_be_coloured = dictionary with nodes as keys and values indicating different colors.
    - palette = sns.color palette e.g. 'YlOrRd' 
    
    Return dictionary (randomly sorted) with nodes as keys and assigned color to each node.
    ''' 

    # Colouringg
    colour_groups = set(d_to_be_coloured.values())
    colour_count = len(colour_groups)
    pal = sns.color_palette(palette, colour_count)
    palette = pal.as_hex()

    d_colourgroups = {}
    for n in colour_groups:
        d_colourgroups[n] = [k for k in d_to_be_coloured.keys() if d_to_be_coloured[k] == n]
        
    d_colourgroups_sorted = {key:d_colourgroups[key] for key in sorted(d_colourgroups.keys())}

    d_val_col = {}
    for idx,val in enumerate(d_colourgroups_sorted):
        for ix,v in enumerate(palette):
            if idx == ix:
                d_val_col[val] = v

    d_node_colour = {}
    for y in d_to_be_coloured.items(): # y[0] = node id, y[1] = val
        for x in d_val_col.items(): # x[0] = val, x[1] = (col,col,col)
            if x[0] == y[1]:
                d_node_colour[y[0]]=x[1]
    
    return d_node_colour # colours


def color_nodes_from_dict(G, d_to_be_coloured, palette): 
    ''' 
    Generate node colors based on dictionary.
    Input: 
    - G = Graph 
    - d_to_be_coloured = dictionary with nodes as keys and values indicating different colors.
    - palette = sns.color palette e.g. 'YlOrRd' 
    
    Return dictionary, sorted according to Graph nodes, with nodes as keys and assigned color to each node.
    ''' 
    
    # Colouring
    colour_groups = set(d_to_be_coloured.values())
    colour_count = len(colour_groups)
    pal = sns.color_palette(palette, colour_count)
    palette = pal.as_hex()

    d_colourgroups = {}
    for n in colour_groups:
        d_colourgroups[n] = [k for k in d_to_be_coloured.keys() if d_to_be_coloured[k] == n]
        
    d_colourgroups_sorted = {key:d_colourgroups[key] for key in sorted(d_colourgroups.keys())}

    d_val_col = {}
    for idx,val in enumerate(d_colourgroups_sorted):
        for ix,v in enumerate(palette):
            if idx == ix:
                d_val_col[val] = v

    d_node_colour = {}
    for y in d_to_be_coloured.items(): # y[0] = node id, y[1] = val
        for x in d_val_col.items(): # x[0] = val, x[1] = (col,col,col)
            if x[0] == y[1]:
                d_node_colour[y[0]]=x[1]

    # SORT dict based on G.nodes
    d_node_colour_sorted = dict([(key, d_node_colour[key]) for key in G.nodes()])
    
    return d_node_colour_sorted


def color_nodes_from_genelist(G, l_genes, color, color_rest):
    '''
    Color (highlight) nodes from specific node list.
    Input: 
    - G = Graph 
    - l_genes = list of nodes 
    - color = string; color to hightlight
    - color_rest = string; color for all other genes
    
    Return node color list sorted based on G.nodes() 
    '''
    
    d_col = {}
    for node in l_genes:
        d_col[str(node)] = color
            
    d_rest = {}
    for g in G.nodes():
        if g not in d_col.keys():
            d_rest[g] = color_rest #'d3d3d3' #'696969', #'dimgrey' # 'rgba(50,50,50,0.5)'
                    
    d_allnodes_col = {**d_col, **d_rest}
    d_allnodes_col_sorted = {key:d_allnodes_col[key] for key in G.nodes()}

    colours = list(d_allnodes_col_sorted.values())
    
    return colours


def color_edges_from_genelist(G, l_genes, color):
    '''
    Color (highlight) edges from specific node list.
    Input: 
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges. 
    '''
    
    edge_lst = [(u,v) for u,v in G.edges(l_genes) if u in l_genes or v in l_genes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color

    return d_col_edges



# -------------------------------------------------------------------------------------
# E S S E N T I A L I T Y   S P E C I F I C  
# -------------------------------------------------------------------------------------


def color_essentiality_nodes(G, essentials, nonessentials, colour1, colour2):
    '''
    Color nodes based on essentiality state.
    Input: 
    - G = graph
    - essentials = list of all essential genes
    - nonessentials = list of all non-essential genes 
    - colour1 = string; to color essential genes
    - colour2 = string; to color non-essential genes 
    All rest genes will be coloured in grey.
    
    Return list of colors for each node in the graph, sorted based on Graph nodes.
    '''

    d_ess = {}
    for node in essentials:
        d_ess[node] = colour1

    d_no_ess = {}
    for node in nonessentials:
        d_no_ess[node] = colour2

    d_essentiality = {**d_ess, **d_no_ess}

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_essentiality.keys():
            d_restnodes[i] = 'grey'

    d_all_essentiality = {**d_essentiality, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_essentiality[key] for key in G.nodes()}   
    
    return d_all_nodes_sorted


def z_landscape_essentiality(G, essential_genes, non_ess_genes, value_ess, value_noness, value_undef):
    '''
    Generate z-heights for each node based on essentiality. 
    Input: 
    - G = graph
    - essential_genes = list of all essential genes
    - non_ess_genes = list of all non-essential genes 
    - value_ess = integer; z-height parameter
    - value_noness = integer; z-height parameter
    - value_undef = integer; z-height parameter
    
    Return dictionary with nodes as keys and z-heights assigned according to essentiality state. 
    '''
    
    d_ess = {}
    for node in essential_genes:
        d_ess[node] = value_ess

    d_no_ess = {}
    for node in non_ess_genes:
        d_no_ess[node] = value_noness

    d_essentiality = {**d_ess, **d_no_ess}

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_essentiality.keys():
            d_restnodes[i] = value_undef
            
    d_all_essentiality = {**d_essentiality, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_essentiality[key] for key in G.nodes()}   
    
    return d_all_nodes_sorted


# -------------------------------------------------------------------------------------
# D I S E A S E   S P E C I F I C
# -------------------------------------------------------------------------------------


def get_disease_genes(G, d_names_do, d_do_genes, disease_category):
    ''' 
    Get disease-specific genes. 
    Input: 
    - G = Graph 
    - d_names_do: dictionary with gene symbol as keys and disease annotations as values 
    - d_do_genes: dictionary with disease as key and list of genes associated with disease as values
    - disease_category: string; specify disease category e.g. 'cancer'
    
    Return a list of genes associated to specified disease as set for no duplicates.
    '''
    
    # get all genes from disease category
    l_disease_genes = [] 
    for d_name in d_names_do.keys():
        if d_name.find(disease_category) != -1:
            try:
                l_genes = d_do_genes[d_names_do[d_name]]
                for gene in l_genes:
                    l_disease_genes.append(gene)
            except:
                    pass
                
    set_disease_genes = set(l_disease_genes)
    
    return set_disease_genes


def color_edges_from_nodelist(G, l_nodes, color_main, color_rest): # former: def color_disease_outgoingedges(G, l_majorcolor_nodes, color)
    '''
    Color (highlight) edges from specific node list.
    Input: 
    - G = Graph 
    - l_nodes = list of nodes 
    - color = color to hightlight
    All other edges will remain in grey.
    
    Return edge list sorted based on G.edges() 
    '''
    
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            if e in l_nodes:
                edge_lst.append(edge)

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color_main

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = color_rest # '#d3d3d3'

    d_edges_all = {**d_col_edges, **d_grey_edges}
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return  d_edges_all



# -------------------------------------------------------------------------------------
# H U B   S P E C I F I C
# -------------------------------------------------------------------------------------


def identify_hubs(degs, closeness, betweens, cutoff):
    '''
    Identify hubs based on a chosen cutoff.
    Input: 
    - degs/closeness/betweens = each > dictionary with nodes as keys and centrality as values.
    - cutoff: integerfor cut off 
    
    Return nodes to be considered as hubs based on cutoff.
    '''
    
    d_deghubs_cutoff = {}
    for node, de in sorted(degs.items(), key = lambda x: x[1], reverse = 1)[:cutoff]:
        d_deghubs_cutoff[node] = de/max(degs.values())

    d_closhubs_cutoff = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1)[:cutoff]:
        d_closhubs_cutoff[node] = cl

    d_betwhubs_cutoff = {}
    for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1)[:cutoff]:
        d_betwhubs_cutoff[node] = be

    # HUBS SCORE 
    overlap = set(d_deghubs_cutoff.keys()) & set(d_closhubs_cutoff.keys()) & set(d_betwhubs_cutoff.keys())

    d_node_hubs = {}
    for node in overlap:
        d_node_hubs[node] = d_deghubs_cutoff[node]+d_betwhubs_cutoff[node]+d_closhubs_cutoff[node]

    
    return d_node_hubs

    
def color_nodes_and_neighbors(G, dict_nodes):
    '''
    Generate colors from nodes and also color their neighbors in a lighter color.
    Input: 
    - G = Graph
    - dict_nodes = list of nodes to color 
    Each node will get one color. It's respective neighbor nodes will show in the same but lighter color.
    
    Return colours for each node in the graph, sorted by graph nodes. 
    '''
    
    n = len(set(dict_nodes))
    color = generate_colorlist_nodes(n)

    # LIGHTER COLORS FOR NEIGHBOURING NODES
    factor = 1.7 # the higher the lighter
    color_neigh = []
    for i in color:
        r,g,b = hex_to_rgb(i)
        color_light = adjust_color_lightness(r,g,b,factor)
        color_neigh.append(color_light)
        
    # major coloured nodes
    d_col_major = {}
    for idx,n in enumerate(dict_nodes.keys()):
            d_col_major[n] = color[idx]

    # direct adjacent nodes for major nodes 
    direct_neigh = {}
    for n in d_col_major.keys():
        l = []
        for pair in G.edges():
            if n == pair[0]:
                l.append(pair[1])
                direct_neigh[n] = l
            elif n == pair[1]:
                l.append(pair[0])
                direct_neigh[n] = l

    d_col_neigh = {}
    for node,col in d_col_major.items():
        for idx, node in enumerate(d_col_major.keys()):
            for nd,neigh in direct_neigh.items():
                for n in neigh:
                    if node==nd and n not in d_col_major.keys():
                        d_col_neigh[n]=color_neigh[idx]

    d_col = {**d_col_major,**d_col_neigh}

    # rest nodes
    d_grey = {}
    for i in G.nodes():
        if i not in d_col.keys():
            d_grey[i] = 'dimgrey'

    d_col_all = {**d_col_major, **d_col_neigh, **d_grey}
    d_nodes_colours = {key:d_col_all[key] for key in G.nodes()}
    
    # Node outgoing edges
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_col_major.keys():
                if e == node:
                    edge_lst.append(edge)

    # Color edges based on hubs
    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_col_major.items():
            if e[0] == node:
                d_col_edges[e]=col
            elif e[1] == node:
                d_col_edges[e]=col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = 'lightgrey'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_colours = {key:d_edges_all[key] for key in G.edges()}

    return d_nodes_colours, d_edges_colours



# delete eventually (nodes + edges are now merged into one function)
'''def color_majornodes_outgoingedges(G, dict_majorcolor_nodes):

    n = len(set(dict_majorcolor_nodes))
    color = generate_colorlist_nodes(n)

    # LIGHTER COLORS FOR NEIGHBOURING NODES
    factor = 1.7 # the higher the lighter
    color_neigh = []
    for i in color:
        r,g,b = hex_to_rgb(i)
        color_light = adjust_color_lightness(r,g,b,factor)
        color_neigh.append(color_light)
        
    # major coloured nodes
    d_col_major = {}
    for idx,n in enumerate(dict_majorcolor_nodes.keys()):
            d_col_major[n] = color[idx]

    # Node outgoing edges
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_col_major.keys():
                if e == node:
                    edge_lst.append(edge)

    # Color edges based on hubs
    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_col_major.items():
            if e[0] == node:
                d_col_edges[e]=col
            elif e[1] == node:
                d_col_edges[e]=col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = 'lightgrey'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return edge_color'''



########################################################################################
#
# N O D E  S I Z E   F U N C T I O N S
#
########################################################################################


# -------------------------------------------------------------------------------------
# DEGREE SPECIFIC
# -------------------------------------------------------------------------------------

def draw_node_degree(G, scalef):
    '''
    Calculate the node degree from graph positions (dict).
    Return list of radii for each node (2D). 
    '''
    
    #x = 20
    #ring_frac = np.sqrt((x-1.)/x)
    #ring_frac = (x-1.)/x

    l_size = {}
    for node in G.nodes():
        k = nx.degree(G, node)
        R = scalef * (1 + k**1.1) 

        l_size[node] = R
        
    return l_size


def draw_node_degree_3D(G, scalef):
    '''
    Calculate the node degree from graph positions (dict).
    Return list of sizes for each node (3D). 
    '''
    
    x = 3
    ring_frac = (x-1.)/x

    deg = dict(G.degree())
    
    d_size = {}
    for i in G.nodes():
        for k,v in deg.items():
            if i == k:
                R = scalef * (1+v**0.9)
                r = ring_frac * R
                d_size[i] = R
    
    return d_size 


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


def embed_umap_2D(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
    '''
    Dimensionality reduction from Matrix using UMAP.
    Return dict (keys: node IDs, values: x,y).
    ''' 
    n_components = 2 

    U = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric, 
        random_state=42)
    
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
# P L O T T I N G 
# -------------------------------------------------------------------------------------

def get_trace2D(x,y,trace_name,colour):
    '''
    Get trace 2D.
    Used for distance functions (2D) 
    ''' 

    trace = pgo.Scatter(name = trace_name,
    x = x,
    y = y,
    mode='markers',
    marker=dict(
        size=6,
        color=colour
    ),)
    
    return trace


def get_trace_nodes_2D(posG, info_list, color_list, size):
    '''
    Get trace of nodes for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opacity = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
    key_list=list(posG.keys())
    trace = pgo.Scatter(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
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


def get_trace_edges_2D(G, posG, color_list, opac = 0.2):
    '''
    Get trace of edges for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opacity = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
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
 
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.2, color = color_list),
                        opacity = opac
                )
    
    return trace_edges


def get_trace_nodes_2D(posG, info_list, color_list, size):
    '''
    Get trace of nodes for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opacity = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
    key_list=list(posG.keys())
    trace = pgo.Scatter(x=[posG[key_list[i]][0] for i in range(len(key_list))],
                           y=[posG[key_list[i]][1] for i in range(len(key_list))],
                           mode = 'markers',
                           text = info_list,
                           hoverinfo = 'text',
                           #textposition='middle center',
                           marker = dict(
                color = color_list,
                size = size,
                symbol = 'circle',
                line = dict(width = 1.0,
                        color = 'dimgrey')
            ),
        )
    
    return trace


def get_trace_edges_2D(G, posG, color_list, opac = 0.2):
    '''
    Get trace of edges for plotting in 2D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opacity = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
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
 
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.2, color = color_list),
                        opacity = opac
                )
    
    return trace_edges


def get_trace_edges_from_genelist2D(l_spec_edges, posG, col, opac=0.1):
    '''
    Get trace of edges for plotting in 3D only for specific edges. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges 
    
    Return a trace of specific edges. 
    '''
    
    edge_x = []
    edge_y = []
    for edge in l_spec_edges:
            x0, y0 = posG[edge[0]]
            x1, y1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.1, color = [col]*len(edge_x)),
                        opacity = opac
                )
    
    return trace_edges


def get_trace_edges_from_genelist2D_(l_spec_edges, posG, col, opac=0.1):
    '''
    Get trace of edges for plotting in 3D only for specific edges. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges 
    
    Return a trace of specific edges. 
    '''
    
    edge_x = []
    edge_y = []
    for edge in l_spec_edges:
            x0, y0 = posG[edge[0]]
            x1, y1 = posG[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            
    trace_edges = pgo.Scatter(
                        x = edge_x, 
                        y = edge_y, 
                        mode = 'lines', hoverinfo='none',
                        line = dict(width = 0.1, color = col),#[col]*len(edge_x)),
                        opacity = opac
                )
    
    return trace_edges



def plot_2D(data,path,fname):
    '''
    Create a 3D plot from traces using plotly.
    Input: 
    - data = list of traces
    - filename = string
    
    Return plot in 2D and file, saved as png.
    '''

    fig = pgo.Figure()
    
    for i in data:
        fig.add_trace(i)
        
    fig.update_layout(template= 'plotly_white', 
                      showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                        ))    
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    # --- show figure ---
    #py.iplot(fig)
    
    # --- get html file ---  
    fig.write_html(path+fname+'.html')
    
    # --- get screenshot image (png) from html --- 
    hti = Html2Image(output_path=path)
    hti.screenshot(html_file = path+fname+'.html', save_as = fname+'.png')
    
    #not working with large file / time ! 
    #fig.write_image(fname+'.png') 
    
    return #py.iplot(fig)




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

def embed_tsne_3D(Matrix, prplxty, density, l_rate, n_iter, metric = 'precomputed'):
    '''
    Dimensionality reduction from Matrix (t-SNE).
    Return dict (keys: node IDs, values: x,y,z).
    '''
    
    tsne3d = TSNE(n_components = 3, random_state = 0, perplexity = prplxty,
                     early_exaggeration = density,  learning_rate = l_rate, n_iter = n_iter, metric = metric)
    embed = tsne3d.fit_transform(Matrix)

    return embed 


def embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
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
        metric = metric)
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


# -------------------------------------------------------------------------------------
# P L O T T I N G 
# -------------------------------------------------------------------------------------


def get_trace_nodes_3D(posG, info_list, color_list, size, opac=0.9):
    '''
    Get trace of nodes for plotting in 3D. 
    Input: 
    - posG = dictionary with nodes as keys and coordinates as values.
    - info_list = hover information for each node, e.g. a list sorted according to the initial graph/posG keys
    - color_list = list of colours obtained from any color function (see above sections).
    - opac = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
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
                        color = color_list),
                opacity = opac,
            ),
        )
    
    return trace


def get_trace_edges_3D(G, posG, color_list, opac = 0.2):
    '''
    Get trace of edges for plotting in 3D. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color_list = list of colours obtained from any color function (see above sections).
    - opac = transparency of edges e.g. 0.2
    
    Return a trace for plotly graph objects plot. 
    '''
    
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
                            line = dict(width = 0.2, color = color_list),
                            opacity = opac
                    )
    return trace_edges



def get_trace_edges_from_genelist3D(l_spec_edges, posG, col, opac=0.2):
    '''
    Get trace of edges for plotting in 3D only for specific edges. 
    Input: 
    - G = Graph
    - posG = dictionary with nodes as keys and coordinates as values.
    - color = string; specific color to highlight specific edges 
    
    Return a trace of specific edges. 
    '''
    
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in l_spec_edges:
            x0, y0,z0 = posG[edge[0]]
            x1, y1,z1 = posG[edge[1]]
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
                        line = dict(width = 1.0, color = [col]*len(edge_x)),
                        opacity = opac
                )
    return trace_edges


def get_trace_edges_landscape(x,y,z0,z):
    '''
    Create trace of vertical connecting edges in between node z0 and node z=parameter (e.g.disease count).
    Return trace with edges.
    '''
    
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


def plot_3D(data, fname, scheme, annotat=None):
    '''
    Create a 3D plot from traces using plotly.
    Input: 
    - data = list of traces
    - filename = string
    - scheme = 'light' or 'dark'
    - annotations = None or plotly annotations
    
    Return plot in 3D and file, saved as html.
    '''

    fig = pgo.Figure()
    
    for i in data:
        fig.add_trace(i)

    if scheme == 'dark' and annotat==None:
        fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='black')),
                            dragmode="turntable"
                        ))
        
    elif scheme == 'dark':    
        fig.update_layout(template='plotly_dark', showlegend=False, autosize = True,
                                  scene=dict(
                                      xaxis_title='',
                                      yaxis_title='',
                                      zaxis_title='',
                                      xaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                      yaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                      zaxis=dict(nticks=0,tickfont=dict(
                                            color='black')),
                                    dragmode="turntable",
                                    annotations=annotat,
                                ))

    elif scheme == 'light' and annotat==None:
        fig.update_layout(template='plotly_white', showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),    
                            dragmode="turntable",
                        ))    
        
    elif scheme == 'light':
         fig.update_layout(template='plotly_white', showlegend=False, width=1200, height=1200,
                          scene=dict(
                              xaxis_title='',
                              yaxis_title='',
                              zaxis_title='',
                              xaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              yaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),
                              zaxis=dict(nticks=0,tickfont=dict(
                                    color='white')),    
                            dragmode="turntable",
                            annotations = annotat
                        ))    


    return plotly.offline.plot(fig, filename = fname+'.html', auto_open=True)


# -------------------------------------------------------------------------------------
# P L O T   A N N O T A T I O N S 
# -------------------------------------------------------------------------------------


def cluster_annotation(d_clusterid_coords, d_genes_per_cluster, mode = 'light'):
    ''' 
    Add Anntation of clusters to 3D plot.
    Input:
    - d_clusterid_coords = dictionary with cluster id and x,y,z coordinates of cluster center.
    - d_genes_per_cluster = dictionary with cluster id and genes counted per cluster 
    - mode = mode of plot (i.e. 'light', 'dark')
    
    Return Annotations for each cluster.
    '''    
    
    l_clus_genecount = list(d_genes_per_cluster.values())

    x=[]
    y=[]
    z=[]
    for i in d_clusterid_coords.values():
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}', 
                                font=dict(
                                    color="dimgrey",
                                    size=8),
                                xanchor="right",
                                ay=-100,
                                ax=-100,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="dimgrey"
                                )
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                x=x[i],
                                y=y[i],
                                z=z[i],
                                showarrow=True,
                                text=f'Cluster: {str(i+1)} <br> total: {str(l_clus_genecount[i])}',
                                font=dict(
                                    color="lightgrey",
                                    size=8),
                                xanchor="right",
                                ay=-100,
                                ax=-100,
                                opacity=0.5,
                                arrowhead=0,
                                arrowwidth=0.5,
                                arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations
        
    else: 
        print('Please choose mode by setting mode="light" or "dark".')

    

def genes_annotation(posG_genes, d_genes, mode = 'light'):
    '''
    Add Anntation of genes to 3D plot.
    Input:
    - posG_genes = dictionary with node id and x,y,z coordinates of cluster center.
    - d_genes = dictionary with node id as keys and symbol (gene symbol) as values. Same order as posG_genes
    - mode of plot (i.e. 'light', 'dark')
    
    Return Annotations for each cluster.
    ''' 
    
    gene_sym = list(d_genes.values())
    
    x = []
    y = []
    z = []
    for k,v in posG_genes.items():
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])

    if mode == 'light':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                    x=x[i],
                                    y=y[i],
                                    z=z[i],
                                    showarrow=True,
                                    text=f'Gene: {gene_sym[i]}',                            
                                    font=dict(
                                        color="black",
                                        size=10),
                                    xanchor="right",
                                    ay=-100,
                                    ax=-100,
                                    opacity=0.5,
                                    arrowhead=0,
                                    arrowwidth=0.5,
                                    arrowcolor="dimgrey")
            i=+1
            annotations.append(annot)
        return annotations

    elif mode == 'dark':
        annotations = []
        for i in range(len(x)):
            annot = dict(
                                    x=x[i],
                                    y=y[i],
                                    z=z[i],
                                    showarrow=True,
                                    text=f'Gene: {gene_sym[i]}',
                                    font=dict(
                                        color="white",
                                        size=10),
                                    xanchor="right",
                                    ay=-100,
                                    ax=-100,
                                    opacity=0.5,
                                    arrowhead=0,
                                    arrowwidth=0.5,
                                    arrowcolor="lightgrey")
            i=+1
            annotations.append(annot)
        return annotations

    else: 
        print('Please choose mode by setting mode="light" or "dark".')


        
########################################################################################
#
# E X P O R T   C O O R D I N A T E S   F U N C T I O N S 
# 
# compatible/for uplad to VRNetzer Platform and Webapp 
#
########################################################################################


def export_to_csv2D(layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 2D layouts. 
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    
    colours_hex2rgb = []
    for j in colours: 
        k = hex_to_rgb(j)
        colours_hex2rgb.append(k)
        
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours_hex2rgb:
        colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]))#*255))
        colours_b.append(int(i[2]))#*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df_2D = pd.DataFrame(posG).T
    df_2D.columns=['X','Y']
    df_2D['Z'] = 0
    df_2D['R'] = colours_r
    df_2D['G'] = colours_g
    df_2D['B'] = colours_b
    df_2D['A'] = colours_a

    df_2D[layout_namespace] = layout_namespace
    df_2D['ID'] = list(posG.keys())

    cols = df_2D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_2D_final = df_2D[cols]
    
    return df_2D_final.to_csv(r'_VR_layouts/'+layout_namespace+'.csv',index=False, header=False)


def export_to_csv3D(layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 3D layouts. 
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    
    colours_hex2rgb = []
    for j in colours: 
        k = hex_to_rgb(j)
        colours_hex2rgb.append(k)
        
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours_hex2rgb:
        colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]))#*255))
        colours_b.append(int(i[2]))#*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df_3D = pd.DataFrame(posG).T
    df_3D.columns=['X','Y','Z']
    df_3D['R'] = colours_r
    df_3D['G'] = colours_g
    df_3D['B'] = colours_b
    df_3D['A'] = colours_a

    df_3D[layout_namespace] = layout_namespace
    df_3D['ID'] = list(posG.keys())

    cols = df_3D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_3D_final = df_3D[cols]
    
    return df_3D_final.to_csv(r'_VR_layouts/'+layout_namespace+'.csv',index=False, header=False)



########################################################################################
# 
# G E N E   I D / S Y M B O L   F U N C T I O N S  
#
########################################################################################


# GENE entrezID <-> Gene Symbol 

def genent2sym():
    '''
    Return two dictionaries.
    First with gene entrezid > symbol. Second with symbol > entrezid. 
    '''
    
    db = mysql.connect("menchelabdb.int.cemm.at","readonly","ra4Roh7ohdee","GenesGO")    

    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    sql = """   SELECT
                    Approved_Symbol,
                    Entrez_Gene_ID_NCBI 
                FROM GenesGO.hgnc_complete
                WHERE Entrez_Gene_ID_NCBI != ''
          """ 

    cursor.execute(sql)
    data = cursor.fetchall()    
#     try: 
#         # execute SQL query using execute() method.
#         cursor.execute(sql)
#         data = cursor.fetchall()
#     except:
#         print('SQL error')
    db.close()

#     t0 = time.time()
    d_sym_ent = {}
    d_ent_sym = {}

    for x in data:
        sym = x[0]
        ent = x[1]
        d_sym_ent[sym] = ent
        d_ent_sym[ent] = sym
#     print(time.time()-t0)
    
    return d_ent_sym, d_sym_ent



# Gene entrezID <-> Gene Symbol 

def convert_symbol_to_entrez(gene_list,name_species):   #name_species must be the official entrez name in string format
    '''
    Get gene list and name of species and
    Return a dict of Gene Symbol and EntrezID
    '''
    
    sym_to_entrez_dict={}    #create a dictionary symbol to entrez
    for gene in gene_list:
        #retrieve gene ID
        handle = Entrez.esearch(db="gene", term=name_species+ "[Orgn] AND " + gene + "[Gene]")
        record = Entrez.read(handle)

        if len(record["IdList"]) > 0:
            sym_to_entrez_dict[gene]=record["IdList"][0]
        else:
            pass
    return sym_to_entrez_dict





########################################################################################
#
# N O T  I N  U S E  ??? 
#
########################################################################################

def color_edges_from_list(G, genelist, col):
    '''
    Color edges based on essentiality state.
    Input: 
    - G = graph
    - genelist = list of all genes to take into consideration 
    - colour = string; to color gene edges 
    All rest edges will be coloured in grey.
    
    Return list of colors for each edge, sorted based on Graph edges.
    '''
    
    # EDGES ------------------------------
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_genes.keys():
                if e == node:
                    edge_lst.append(edge)

    d_col_edges = {}
    for node,col in d_genes.items():
            if e[0] == node:
                d_col_edges[e]= col
            elif e[1] == node:
                d_col_edges[e]= col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = '#d3d3d3'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return edge_color


def color_diseasecategory(G, d_names_do, d_do_genes, disease_category, colour):
    
    # get all genes from disease category
    l_disease_genes = []
    for d_name in d_names_do.keys():
        if d_name.find(disease_category) != -1:
            try:
                l_genes = d_do_genes[d_names_do[d_name]]
                for gene in l_genes:
                    l_disease_genes.append(gene)
            except:
                    pass
                
    set_disease_genes = set(l_disease_genes)
    
    # assign colours to disease cat.(colour1) and other nodes(grey)
    d_col = {}
    for node in set_disease_genes:
        d_col[node] = colour
    
    d_rest = {}
    for i in G.nodes():
        if i not in d_col.keys():
            d_rest[i] = '#303030' # 'dimgrey'
        
    d_allnodes_col = {**d_col, **d_rest}
    d_allnodes_col_sorted = {key:d_allnodes_col[key] for key in G.nodes()}
    
    colours = list(d_allnodes_col_sorted.values())
    
    return colours


def color_nodes_from_list(G, l_nodes, col):
    '''
    Color nodes based on essentiality state.
    Input: 
    - G = graph
    - l_nodes = list of nodes
    - col = string or hex; colour 
    All rest genes will be coloured in grey.
    
    Return list of colors for each node in the graph, sorted based on Graph nodes.
    '''

    d_nodes = {}
    for node in l_nodes:
        d_nodes[node] = col

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_nodes.keys():
            d_restnodes[i] = 'lightgrey'

    d_all_nodes = {**d_nodes, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_nodes[key] for key in G.nodes()}   
    
    return d_all_nodes_sorted


def color_edges_from_nodelist(G, l_nodes, color_main, color_rest): # former: def color_disease_outgoingedges(G, l_majorcolor_nodes, color)
    '''
    Color (highlight) edges from specific node list.
    Input: 
    - G = Graph 
    - l_nodes = list of nodes 
    - color = color to hightlight
    All other edges will remain in grey.
    
    Return edge list sorted based on G.edges() 
    '''
    
    d_col_major = {}
    for n in l_nodes:
            d_col_major[n] = color_main

    edge_lst = []
    for edge in G.edges():
        for e in edge:
            if e in d_col_major.keys():
                #if e == node:
                edge_lst.append(edge)

    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_col_major.items():
            if e[0] == node:
                d_col_edges[e]=col
            elif e[1] == node:
                d_col_edges[e]=col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = color_rest# '#d3d3d3'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    #d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    #edge_color = list(d_edges_all_sorted.values())
    
    return  d_edges_all
