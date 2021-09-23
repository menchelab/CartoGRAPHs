
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains F U N C T I O N S   F O R   V I S U A L  P R O P E R T I E S
#
########################################################################################

import colorsys
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances

########################################################################################

def color_nodes(l_genes, color):
    ''' 
    Color nodes of list with same color.
    Returns as dict with node ID and assigned color. 
    ''' 
    d_col = {}
    for node in l_genes:
        d_col[str(node)] = color
    
    return d_col


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


def color_edges_from_nodelist_or(G, l_genes, color):
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


def color_edges_from_nodelist_and(G, l_genes, color):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input:
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF ONE node is in l_genes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(l_genes) if u in l_genes and v in l_genes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color
    return d_col_edges



#def color_edges_from_nodelist_specific(G, l_nodes, color_main): # color_rest): # former: def color_disease_outgoingedges(G, l_majorcolor_nodes, color)
#    '''
#    Color (highlight) edges from specific node list.
#    Input: 
#    - G = Graph 
#    - l_nodes = list of nodes 
#    - color = color to hightlight
#    Return dictionary with coloured edges (not all G.edges!).
#    '''
#    edge_lst = []
#    for edge in G.edges():
#        for e in edge:
#            if e in l_nodes:
#                edge_lst.append(edge)
#    d_col_edges = {}
#    for e in edge_lst:
#        d_col_edges[e]=color_main   
#    return  d_col_edges



def color_edges_from_nodelist_specific(G, l_nodes, color_main):
    '''
    Color (highlight) edges from specific node list exclusively.
    Input:
    - G = Graph 
    - l_nodes = list of nodes 
    - color = string; color to hightlight
    
    Return edge list for selected edges IF ONE node is IN l_genes. 
    '''
    
    edge_lst = [(u,v)for u,v in G.edges(l_genes) if u in l_genes and v in l_genes]

    d_col_edges = {}
    for e in edge_lst:
        d_col_edges[e]=color
    return d_col_edges


def colours_spectralclustering(G, posG, n_clus, n_comp, pal ='gist_rainbow'):
    '''
    Generate node colors based on clustering.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and xy(z) coordinates
    - n_clus = int; number of clusters
    - n_comp = int; number of components (e.g. 10)
    - palette(optional) = string; sns color palette e.g. "gist_rainbow"

    Returns a dictionary with nodes as keys and color values based on clustering method. 
    '''
    
    df_posG = pd.DataFrame(posG).T 

    model = SpectralClustering(n_clusters=n_clus,n_components = n_comp, affinity='nearest_neighbors',random_state=0)
    clusterid = model.fit(df_posG)
    d_node_clusterid = dict(zip(genes, clusterid.labels_))

    colours_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal) #'ocean'
    genes_val = ['#696969']*len(genes_rest)
    colours_rest = dict(zip(genes_rest, genes_val))
    colours_all = {**colours_rest, **colours_unsort}

    d_colours = {key:colours_all[key] for key in G.nodes}
    
    return d_colours



def colours_dbscanclustering(G, DM, posG, epsi, min_sam, pal = 'gist_rainbow', col_rest = '#696969'):
    '''
    Generate node colors based on clustering.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and xy(z) coordinates
    - epsi = int; number of clusters
    - min_sam = int; number of components (e.g. 10)
    - palette(optional) = string; sns color palette e.g. "gist_rainbow"

    Returns a dictionary with nodes as keys and color values based on clustering method. 
    '''
    genes = []
    for i in DM.index:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)
            
    df_posG = pd.DataFrame(posG).T 
    dbscan = DBSCAN(eps=epsi, min_samples=min_sam) 
    clusterid = dbscan.fit(df_posG)
    d_node_clusterid = dict(zip(genes, clusterid.labels_))

    colours_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal)
    genes_val = [col_rest]*len(genes_rest)
    colours_rest = dict(zip(genes_rest, genes_val))
    colours_all = {**colours_rest, **colours_unsort}

    d_colours_sorted = {key:colours_all[key] for key in G.nodes}
    print('Number of Clusters: ', len(set(clusterid.labels_)))
    
    return d_colours_sorted



def kmeansclustering(posG, n_clus):
    
    df_posG = pd.DataFrame(posG).T 
    kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(df_posG)
    centrs = kmeans.cluster_centers_
    
    return kmeans, centrs



def colours_kmeansclustering(G, DM, kmeans, pal = 'gist_rainbow'):
    '''
    Generate node colors based on clustering.
    Input:
    - G = Graph
    - posG = dictionary with nodes as keys and xy(z) coordinates
    - n_clus = int; number of clusters
    - palette(optional) = string; sns color palette e.g. "gist_rainbow"

    Returns a dictionary with nodes as keys and color values based on clustering method. 
    '''

    genes = []
    for i in DM.index:
        if str(i) in G.nodes():
            genes.append(str(i))

    genes_rest = [] 
    for g in G.nodes():
        if str(g) not in genes:
            genes_rest.append(g)
            
    d_node_clusterid = dict(zip(genes, kmeans.labels_))
    colours_unsort = color_nodes_from_dict_unsort(d_node_clusterid, pal ) #'prism'
    
    genes_val = ['#696969']*len(genes_rest)
    colours_rest = dict(zip(genes_rest, genes_val))
    colours_all = {**colours_rest, **colours_unsort}
    d_colours_sorted = {key:colours_all[key] for key in G.nodes}
    
    return d_colours_sorted



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


def zparam_essentiality(G, essential_genes, non_ess_genes, value_ess, value_noness, value_undef):
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


   
def get_hubs(G, max_treshold, min_treshold):
    
    d_degree = dict(nx.degree(G))

    hubs = {}
    for k,v in d_degree.items():
        if v >= min_treshold and v <= max_treshold:
            hubs[k] = v
    #print('Hubs: ',hubs)

    # get their neighbours
    neighbours = {}
    hubs_neigh = []
    for i in hubs.keys():
        for edge in G.edges():
            if edge[0] == i:
                hubs_neigh.append(edge[1])
            elif edge[1] == i:
                hubs_neigh.append(edge[0])
            neighbours[i] = hubs_neigh
    
    
    return hubs,neighbours



def color_nodes_hubs(G, hubs, neighs, hubs_col_nodes, neigh_col_nodes):
    
    rest_col_nodes = '#d3d3d3' 
    rest_col_edges = '#d3d3d3' 

    colours_hubs = {}
    for i in G.nodes():
        if str(i) in hubs.keys():
            colours_hubs[i] = hubs_col_nodes
        elif str(i) in neighs.keys():
            colours_hubs[i] = neigh_col_nodes
        else: 
            colours_hubs[i] = rest_col_nodes

    hubs_all_sorted = {key:colours_hubs[key] for key in G.nodes()}
    #colours = list(hubs_all_sorted.values())
    
    return hubs_all_sorted 



def color_edges_hubs(G, hubs, hub_col_edges, rest_col_edges):

    d_edge_col_ = color_edges_from_genelist(G, list(hubs.keys()), hub_col_edges)
    d_rest_edges={}
    for e in G.edges():
        if str(e) not in d_edge_col_.keys():
            d_rest_edges[e] = rest_col_edges

    d_all_edges = {**d_edge_col_, **d_rest_edges}
    d_all_edges_sort = {key:d_all_edges[key] for key in G.edges()}
    
    return d_all_edges_sort



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