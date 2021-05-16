
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

#from html2image import Html2Image

#from igraph import *
import itertools as it

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

import umap 

import warnings
#warnings.filterwarnings("ignore", category=UserWarning)


########################################################################################
#
# F U N C T I O N S   F O R   B E N C H M A R K I N G   S C R I P T S
# 
########################################################################################

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


def pairwise_layout_distance_2D(G,posG):

    dist_layout = {} 
    for p1,p2 in it.combinations(G.nodes(),2):
        dist_layout[(p1,p2)] = np.sqrt((posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2)
        
    return dist_layout


def pairwise_layout_distance_3D(G,posG):

    dist_layout = {} 
    for p1,p2 in it.combinations(G.nodes(),2):
        dist_layout[(p1,p2)] = np.sqrt((posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2 + (posG[p1][1]-posG[p2][2])**2)
        
    return dist_layout
 
    
def pairwise_network_distance(G):
    
    dist_network = {}
    for p1,p2 in it.combinations(G.nodes(),2):
        dist_network[(p1,p2)] = nx.shortest_path_length(G,p1,p2, method='dijkstra')

    return dist_network


def pairwise_network_distance_parts(G,list_of_nodes):
    
    dist_network = {}
    for p1,p2 in it.combinations(list_of_nodes,2):
        dist_network[(p1,p2)] = nx.shortest_path_length(G,p1,p2, method='dijkstra')
    
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


