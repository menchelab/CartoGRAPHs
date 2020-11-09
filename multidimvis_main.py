
# main for Multidim_vis scripts 

# -------------------------------------------------------------------------------------
# LIBRARIES
# -------------------------------------------------------------------------------------

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
import multiprocessing
import mygene

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.generators.degree_seq import expected_degree_graph
from networkx.algorithms.community import greedy_modularity_communities
from node2vec import Node2Vec
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
import plotly.express as px
import plotly.graph_objs as pgo
import plotly.offline as py
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.io as pio
from prettytable import PrettyTable
import pylab
#py.init_notebook_mode(connected = True)
import pymysql as mysql

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
import sklearn
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


########################################################################################
#
# FUNCTIONS FOR ANALYSIS + CALCULATIONS ETC
#
########################################################################################


# GENE entrezID <-> Gene Symbol 
"""
Return two dictionaries.
First with gene entrezid > symbol. Second with symbol > entrezid. 
"""
def genent2sym():

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


'''
Return binned nodes.
'''
def bin_nodes(data_dict): 
    bins = set(data_dict.values())

    d_binned = {}
    for n in bins:
        d_binned[n]=[str(k) for k in data_dict.keys() if data_dict[k] == n]
        
    return d_binned


def tsne_portrait2D_to_csv(posG, colours, organism):
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours:
        colours_r.append(int(i[0]*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]*255))
        colours_b.append(int(i[2]*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df = pd.DataFrame(posG).T
    df.columns=['X','Y']

    df['R'] = colours_r
    df['G'] = colours_g
    df['B'] = colours_b
    df['A'] = colours_a

    return df.to_csv(r'output_layout_csv/2Dportrait_'+organism+'.csv', index = False, header = False)



'''
Function to make 3D html plot rotating.
Returns frames, to be used in "pgo.Figure(frames = frames)"
'''
def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z 


########################################################################################
#
# C O L O R F U N C T I O N S 
#
########################################################################################

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
From generated colors, get lighter or darker subcategorical colors.
Return colors in light/dark version.
'''
def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))


def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))


def darken_color(r, g, b, factor=0.9):
    return adjust_color_lightness(r, g, b, 1 - factor)


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

    return d_col #colours


def color_nodes_from_dict_same(G, dict_color_nodes, color):

    d_col = {}
    for node,val in dict_color_nodes.items():
        for i,c in enumerate(color):
            if i == val:
                d_col[node] = color
                
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
    edge_lst = [(u,v) for u,v in G.edges(l_genes) if u in l_genes and v in l_genes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color

    return d_col_edges


def get_trace_edges_from_genelist_2D(l_spec_edges, posG, color_list):
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
                        line = dict(width = 0.5, color = color_list),
                        opacity = 0.3
                )
    
    return trace_edges


# -------------------------------------------------------------------------------------
# ESSENTIALITY SPECIFIC
# -------------------------------------------------------------------------------------

'''
Color nodes based on their Essentiality (Yeast PPI).
Return list of colors sorted based on Graph nodes.
'''
def color_essentiality_nodes(G, essentials, nonessentials, colour1, colour2):
    
    # NODES ------------------------------
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



# -------------------------------------------------------------------------------------
# DISEASE SPECIFIC
# -------------------------------------------------------------------------------------

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
            d_rest[i] = '#303030' # 'dimgrey'
        
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


# -------------------------------------------------------------------------------------
# HUB SPECIFIC
# -------------------------------------------------------------------------------------

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





########################################################################################
#
# N O D E  S I Z E   F U N C T I O N S
#
########################################################################################


# -------------------------------------------------------------------------------------
# DEGREE SPECIFIC
# -------------------------------------------------------------------------------------

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

        l_size[node] = R
        
    return l_size


'''
Calculate the node degree from graph positions (dict).
Return list of sizes for each node (3D). 
'''
def draw_node_degree_3D(G, scalef):
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
# P L O T T I N G  
#
########################################################################################

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

def embed_umap_2D(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
    n_components = 2 # for 2D

    U = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric)
    embed = U.fit_transform(Matrix)
    
    return embed

'''
Get 2D coordinates for each node.
Return dict with node: x,y coordinates.
''' 
def get_posG_2D(l_nodes, embed):
    posG = {}
    cc = 0
    for entz in l_nodes:
        # posG[str(entz)] = (embed[cc,0],embed[cc,1])
        posG[entz] = (embed[cc,0],embed[cc,1])
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
def calc_dist_2D(posG):
    
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
Validation of Layouts 3D. Calculates distances from layout.
Return list with distances. 
'''
def calc_dist_3D(posG):
    
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

'''
Get trace 2D.
Used for distance functions (2D) 
'''
def get_trace2D(x,y,trace_name,colour):
    trace = pgo.Scatter(name = trace_name,
    x = x,
    y = y,
    mode='markers',
    marker=dict(
        size=6,
        color=colour
    ),)
    return trace



'''
Get trace 2D.
Used for distance functions (2D; benchmarking) 
'''
def get_trace_xy(x,y,trace_name,colour):
    trace = pgo.Scatter(name = trace_name,
    x = x,
    y = y,
    mode='markers',
    marker=dict(
        size=6,
        color=colour
    ),)
    return trace



'''
Generate 3D trace. 
Used for distance functions (3D; benchmarking)
''' 
def get_trace_xyz(x,y,z,name,colour,size):
    
    
    trace = pgo.Scatter3d(
        x = x,
        y = y,
        z = z,
        mode='markers',
        text=name,
        marker=dict(
            size=size,
            color=colour, 
            line_width=0.5,
            line_color = colour,
        ),)
    return trace


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
            color=colour, 
            line_width=0.5,
            line_color = colour,
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
        line=dict(width=0.1, color=color_list),
        hoverinfo='none',
        mode='lines',
        opacity = 0.15)
    
    return edge_trace



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

    '''
    posG = {}
    cc = 0
    for entz in sorted(G.nodes()):
        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1
    posG_sorted = {key:posG[key] for key in G.nodes()}
    
    return posG_sorted'''

def get_posG_3D(l_genes, embed):
    posG = {}
    cc = 0
    for entz in l_genes:
        posG[entz] = (embed[cc,0],embed[cc,1],embed[cc,2])
        cc += 1
    
    return posG



'''
Dimensionality reduction from Matrix (UMAP).
Return dict (keys: node IDs, values: x,y,z).
'''
def embed_umap_3D(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
    n_components = 3 # for 3D

    U_3d = umap.UMAP(
        n_neighbors = n_neighbors,
        spread = spread,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric)
    embed = U_3d.fit_transform(Matrix)
    
    return embed


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


def get_trace_nodes_2D(posG, info_list, color_list, size):

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



def get_trace_nodes_3D(posG, info_list, color_list, size):

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



'''
Generates edges from 3D coordinates.
Returns a trace of edges.
'''
def get_trace_edges_3D(G, posG, color_list, opac = 0.2):
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
                        line = dict(width = 0.1, color = color_list),
                        opacity = opac
                )
    
    return trace_edges



# -------------------------------------------------------------------------------------
# 3D SPHERES 
# -------------------------------------------------------------------------------------

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
Return fitted model. 
'''
def embed_umap_sphere(Matrix, n_neighbors, spread, min_dist, metric='cosine'):
    
    model = umap.UMAP(
        n_neighbors = n_neighbors, 
        spread = spread,
        min_dist = min_dist,
        metric = metric)

    sphere_mapper = model.fit(Matrix)

    return sphere_mapper



'''
(UMAP) Nodes (l_genes) are positioned based on model (sphere_mapper) onto sphere.
Return dict with coordinates (x,y,z).
'''
def get_posG_sphere(l_genes, sphere_mapper):
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])
    
    posG = {}
    cc = 0
    for entz in l_genes:
        posG[entz] = (x[cc],y[cc], z[cc])
        cc += 1
    
    return posG



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


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec




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




# -------------------------------------------------------------------------------------
# 3D TORUS
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







########################################################################################
#
# EXPORT COORDINATES FUNCTIONS
#
########################################################################################


# export for csv --> VR DataDiVR , NEON Webapp 

# export Dataframe
# Format is compatible with VR upload into DataDiVR and for WEBAPP "NEON"

def export_to_csv2D(layout, posG, entrezID_list, colours):
    
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours:
        colours_r.append(int(i[0]*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]*255))
        colours_b.append(int(i[2]*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df_2D = pd.DataFrame(posG).T
    df_2D.columns=['X','Y']
    df_2D['Z'] = 0
    df_2D['R'] = colours_r
    df_2D['G'] = colours_g
    df_2D['B'] = colours_b
    df_2D['A'] = colours_a

    df_2D[layout] = layout
    df_2D['ID'] = entrezID_list

    cols = df_2D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_2D_final = df_2D[cols]
    
    return df_2D_final.to_csv(r'VR_layouts/'+layout+'.csv',index=False, header=False)


def export_to_csv3D(layout, posG, entrezID_list, colours):
    
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours:
        colours_r.append(int(i[0]*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]*255))
        colours_b.append(int(i[2]*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df_3D = pd.DataFrame(posG).T
    df_3D.columns=['X','Y','Z']
    df_3D['R'] = colours_r
    df_3D['G'] = colours_g
    df_3D['B'] = colours_b
    df_3D['A'] = colours_a

    df_3D[layout] = layout
    df_3D['ID'] = entrezID_list

    cols = df_3D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_3D_final = df_3D[cols]
    
    return df_3D_final.to_csv(r'VR_layouts/'+layout+'.csv',index=False, header=False)