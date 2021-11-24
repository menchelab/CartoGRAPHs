
#########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains F U N C T I O N S   F O R   A N A L Y S I S + C A L C U L A T I O N S
# 
########################################################################################

import numpy as np 
import networkx as nx 

from Bio import Entrez
import pymysql as mysql
import pandas as pd 

from sklearn.preprocessing import normalize
from sklearn import preprocessing

#from shapely import geometry

########################################################################################


def feature_modulation(Struct_matrix, Funct_matrix, scalar_value):
    
    df_max = Struct_matrix.max()
    l_max_visprob = max(list(df_max.values))

    scalar = (1-l_max_visprob)*scalar_value
    
    Matrix_merged = pd.concat([Struct_matrix*(1-scalar), Funct_matrix*scalar]).T
    return Matrix_merged
    
    
    
def compute_centralityfeatures(G):
    '''
    Compute degree,betweenness,closeness and eigenvector centrality
    Input: 
    - G: networkx Graph 
    
    Return a dictionary sorted according to G.nodes with nodeID as keys and four centrality values. 
    ''' 
    
    degs = dict(G.degree())
    d_deghubs = {}
    for node, de in sorted(degs.items(),key = lambda x: x[1], reverse = 1):
        d_deghubs[node] = round(float(de/max(degs.values())),4)

    closeness = nx.closeness_centrality(G)
    d_clos = {}
    for node, cl in sorted(closeness.items(), key = lambda x: x[1], reverse = 1):
        d_clos[node] = round(cl,4)

    betweens = nx.betweenness_centrality(G)
    d_betw = {}
    for node, be in sorted(betweens.items(), key = lambda x: x[1], reverse = 1):
         d_betw[node] = round(be,4)

    eigen = nx.eigenvector_centrality(G)
    d_eigen = {}
    for node, eig in sorted(eigen.items(), key = lambda x: x[1], reverse = 1):
         d_eigen[node] = round(eig,4)

    d_deghubs_sorted = {key:d_deghubs[key] for key in sorted(d_deghubs.keys())}
    d_clos_sorted = {key:d_clos[key] for key in sorted(d_clos.keys())}
    d_betw_sorted = {key:d_betw[key] for key in sorted(d_betw.keys())}
    d_eigen_sorted = {key:d_eigen[key] for key in sorted(d_eigen.keys())}

    feature_dict = dict(zip(d_deghubs_sorted.keys(), zip(d_deghubs_sorted.values(),d_clos_sorted.values(),d_betw_sorted.values(),d_eigen_sorted.values())))

    feature_dict_sorted = {key:feature_dict[key] for key in G.nodes()}
    
    return feature_dict_sorted



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

