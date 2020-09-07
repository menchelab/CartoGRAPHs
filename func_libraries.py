
# ==============================
# LIBRARIES
# ==============================



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
import umap 


