# ============================================
# ANALYSIS + CALCULATIONS
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

# --------------------------------------------
# RANDOM WALK OPERATION
# --------------------------------------------

'''
Random Walk Operator with restart probability.
Return Matrix.
''' 
def rnd_walk_matrix2(A, r, a, num_nodes):

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


# --------------------------------------------
# BINNING
# --------------------------------------------


'''
Binning based on dict.
Return binned nodes.
'''
def bin_nodes(data_dict): 
    bins = set(data_dict.values())

    d_binned = {}
    for n in bins:
        d_binned[n] = [k for k in data_dict.keys() if data_dict[k] == n]
        
    return d_binned


# --------------------------------------------
# HEAT MAP 
# --------------------------------------------

'''
Generate a heatmap + Dendogramm from a Matrix.
Return plot.
'''
def heatmap_from_matrix(Matrix, title = None):

    # Dendogramm
    fig = pylab.figure(figsize=(12,8))

    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    Y = sch.linkage(Matrix, method='average')
    Z = sch.dendrogram(Y, orientation='left')
    axdendro.set_xticks([])
    axdendro.set_yticks([])


    # Plot distance 
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    Dsq = Matrix
    Dsq = Dsq[index,:]
    Dsq = Dsq[:,index]
    im = axmatrix.matshow(Dsq, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])


    # Plot colorbar
    plt.title(title, fontsize= 20)
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    
    plt.savefig('output_plots/' + title + '.png')

    
    return plt.show()


# --------------------------------------------
# CLUSTERING
# --------------------------------------------

def get_node_clusterid(df, X_clusterid, n_clus, n_iterations = 10):
    n = n_clus
    cols = generate_colorlist_nodes(n)
    
    X_clusterid = clusterid[0]
    X_id = df.index
    
    cols_to_clusters = {}
    for ix,col in enumerate(cols):
        for j in X_clusterid:
            if j == ix:
                cols_to_clusters[j]=col

    d_node_clusterid = {k:v for k,v in zip(X_id, X_clusterid)}

    return d_node_clusterid


def get_clustercenter_xy(centers, n_clus, n_iterations = 10):
        
    lx = []
    ly = []
    for clu in range(n_clus):
        for i in centers:
            lx.append(i[clu][0])
            ly.append(i[clu][1])

    return lx, ly


def get_clustercenter_xyz(df, n_clus, n_iterations = 10):

    iterations = range(1,n_iterations)
    centers = []
    kmeans = KMeans(init='random', n_clusters=n_clus, random_state=0, max_iter=n_iterations, n_init=1, verbose=0)
    ids = kmeans.fit_predict(df) 
    centers.append(kmeans.cluster_centers_)
        
    lx = []
    ly = []
    lz = []
    for clu in range(n_clus):
        for i in centers:
            lx.append(i[clu][0])
            ly.append(i[clu][1])
            lz.append(i[clu][2])

    return lx, ly, lz



# --------------------------------------------
# COLORING 
# --------------------------------------------

def color_nodes_from_dict(G, dict_color_nodes):
    n = len(set(dict_color_nodes.values()))
    color = generate_colorlist_nodes(n)

    d_col = {}
    for node,val in dict_color_nodes.items():
        for i,c in enumerate(color):
            if i == val:
                d_col[node] = c       
    #d_col_sorted = {key:d_col[key] for key in G.nodes()}
    #colours = list(d_col_sorted.values())

    return d_col  #colours


def color_nodes_from_dict_same(G, dict_color_nodes, color):
    d_col = {}
    for node,val in dict_color_nodes.items():
        for i,c in enumerate(color):
            if i == val:
                d_col[node] = color[i]      
    #d_col_sorted = {key:d_col[key] for key in G.nodes()}
    #colours = list(d_col_sorted.values())

    return d_col #colours

    
def color_nodes_from_genelist(G, l_genes, color):
    d_col = {}
    for node in l_genes:
        d_col[str(node)] = color
            
    d_rest = {}
    for g in G.nodes():
        if g not in d_col.keys():
            d_rest[g] = 'dimgrey' # 'rgba(50,50,50,0.5)'
                    
    d_allnodes_col = {**d_col, **d_rest}
    d_allnodes_col_sorted = {key:d_allnodes_col[key] for key in G.nodes()}
    colours = list(d_allnodes_col_sorted.values())
    
    return colours


def color_edges_from_genelist(G, l_genes, color):
    edge_lst = []
    for node in l_genes:
        for edge in G.edges():
            if node == edge[0] or node == edge[1]:
                edge_lst.append(edge)

    d_col_edges = {}
    for e in set(edge_lst):
        d_col_edges[e]=color

    return d_col_edges


'''
def color_edges_from_genelist(G, l_genes, color):
    edge_lst = [(u,v) for u,v in G.edges(l_genes) if u in l_genes and v in l_genes]
    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color

    return d_col_edges
'''


# ESSENTIALITY SPECIFIC 

'''
Color nodes based on their Essentiality (Yeast PPI).
Return list of colors sorted based on Graph nodes.
'''
def color_essentiality_nodes(G, essentials, nonessentials, colour1, colour2):
    d_ess = {}
    for node in essentials:
        d_ess['entrez gene/locuslink:'+str(node)] = colour1

    d_no_ess = {}
    for node in nonessentials:
        d_no_ess['entrez gene/locuslink:'+str(node)] = colour2

    d_essentiality = {**d_ess, **d_no_ess}

    d_restnodes = {}
    for i in G.nodes():
        if i not in d_essentiality.keys():
            d_restnodes[i] = 'lightgrey'

    d_all_essentiality = {**d_essentiality, **d_restnodes}
    d_all_nodes_sorted = {key:d_all_essentiality[key] for key in G.nodes()}

    node_color = list(d_all_nodes_sorted.values())
    
    return node_color


'''
Color edges based on their Essentiality (Yeast PPI).
Return list of colors sorted based on Graph edges.
'''
def color_essentiality_edges(G, essentials, nonessentials, colour1, colour2):
    
    # NODES ------------------------------
    d_ess = {}
    for node in essentials:
        d_ess['entrez gene/locuslink:'+str(node)] = colour1

    d_no_ess = {}
    for node in nonessentials:
        d_no_ess['entrez gene/locuslink:'+str(node)] = colour2

    # EDGES ------------------------------
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_ess.keys():
                if e == node:
                    edge_lst.append(edge)

    d_col_edges = {}
    for e in edge_lst:
        for node,col in d_ess.items():
            if e[0] == node:
                d_col_edges[e]= col
            elif e[1] == node:
                d_col_edges[e]= col

    d_grey_edges = {}
    for edge in G.edges():
        if edge not in d_col_edges.keys(): 
            d_grey_edges[edge] = 'silver'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}
    edge_color = list(d_edges_all_sorted.values())
    
    return edge_color


# DISEASE SPECIFIC

def get_disease_genes(G, d_names_do, d_do_genes, disease_category):
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
            d_rest[i] = 'dimgrey'
        
    d_allnodes_col = {**d_col, **d_rest}
    d_allnodes_col_sorted = {key:d_allnodes_col[key] for key in G.nodes()}
    colours = list(d_allnodes_col_sorted.values())
    
    return colours


def color_disease_outgoingedges(G, l_majorcolor_nodes, color):
    d_col_major = {}
    for n in l_majorcolor_nodes:
            d_col_major[n] = color

    # Node outgoing edges
    edge_lst = []
    for edge in G.edges():
        for e in edge:
            for node in d_col_major.keys():
                if e == node:
                    edge_lst.append(edge)

    # Color edges based on major nodes
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
            d_grey_edges[edge] = 'dimgrey'

    d_edges_all = {**d_col_edges, **d_grey_edges}

    # Sort according to G.edges()
    d_edges_all_sorted = {key:d_edges_all[key] for key in G.edges()}

    edge_color = list(d_edges_all_sorted.values())
    
    return edge_color


# HUB SPECIFIC

'''
Identify hubs based on a chosen cutoff.
Return nodes to be considered as hubs based on cutoff.
'''

def identify_hubs(degs, closeness, betweens, cutoff):
    # select how many "important" nodes based on centrality to choose from

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

    
'''
Input a dictionary with nodes to be coloured. 
Return colour list (where nodes of dict. are assigned a color + adjacent nodes are assigned the same color (light version))
'''
def color_majornodes(G, dict_majorcolor_nodes):
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
    d_col_all_sorted = {key:d_col_all[key] for key in G.nodes()}

    l_col_all = list(d_col_all_sorted.values())

    colours = l_col_all
    
    return colours

'''
Input a dictionary with nodes to be coloured. 
Return colour list of edges outgoing from major nodes.
'''

def color_majornodes_outgoingedges(G, dict_majorcolor_nodes):
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
    
    return edge_color



'''
Generate color list based on color count (i.e. nodes to be coloured).
Return list of colors.
'''
def generate_colorlist_nodes(n):
    colors = [colorsys.hsv_to_rgb(1.0/n*x,1,1) for x in range(n)]
    color_list = []
    for c in colors:
        cc = [int(y*255) for y in c]
        color_list.append('#%02x%02x%02x' % (cc[0],cc[1],cc[2]))
        
    return color_list


'''
Transform hex to rgb colors.
Return rgb color values.
'''
def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))


'''
From generated colors, get lighter subcategorical colors.
Return colors in light version.
'''
def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))


'''
From generated colors, get darker subcategorical colors.
Return colors in dark version.
'''
def darken_color(r, g, b, factor=0.9):
    return adjust_color_lightness(r, g, b, 1 - factor)




# --------------------------------------------
# NODE SIZE 
# --------------------------------------------

# CHOOSE BY DICT INPUT 

def draw_node_size(G, d_node_size, scalef):
    
    # sort according to Graph 
    d_node_size_sorted = {key:d_node_size[key] for key in G.nodes()}    
    
    l_size = []
    for nd,val in d_node_size_sorted.items():
        R = scalef * (1 + val**1.1)      
        l_size.append(R)
        
    return l_size


# DEGREE 

'''
Calculate the node degree from graph positions (dict).
Return list of sizes for each node (3D). 
'''
def draw_node_degree_3D(G, l_genes, scalef):
    x = 20
    ring_frac = (x-1.)/x

    deg = dict(G.degree())
    
    d_size = {}
    for i in l_genes:
        for k,v in deg.items():
            if i == k:
                R = scalef * (1+v**1.5)
                r = ring_frac * R
                d_size[i] = r
    
    return d_size

'''
Calculate the node degree from graph positions (dict).
Return list of radii for each node (2D). 
'''
def draw_node_degree(G, scalef):
    #x = 20
    #ring_frac = np.sqrt((x-1.)/x)
    #ring_frac = (x-1.)/x

    l_size = {}
    for node in G.nodes():
        k = nx.degree(G, node)
        R = scalef * (1 + k**1.1)
        #r = ring_frac * R
      
        l_size[node] = R
        
    return l_size
