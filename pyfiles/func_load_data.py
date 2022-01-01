
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains F U N C T I O N S  T O  L O A D  D A T A 
# 
########################################################################################

import networkx as nx
import pickle 
import pandas as pd
from scipy.spatial import distance

########################################################################################

def load_graph(organism):
    
    if organism == 'yeast':
    
        data = pickle.load( open( "input/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.185.mitab.pickle", "rb" ) )
        filter_score = data[
                            #(data['Interaction Types'] == 'psi-mi:"MI:0915"(physical association)') +
                            (data['Interaction Types'] == 'psi-mi:"MI:0407"(direct interaction)') 
                            #&
                            #(data['Taxid Interactor A'] == "taxid:559292") & 
                            #(data['Taxid Interactor B'] == "taxid:559292") 
        ]

        g = nx.from_pandas_edgelist(filter_score, '#ID Interactor A', 'ID Interactor B')
        g.remove_edges_from(nx.selfloop_edges(g)) #remove self loop

        G_cere = g.subgraph(max(nx.connected_components(g), key=len)) # largest connected component (lcc)
        G = G_cere

        return G
    
    elif organism == 'human':
        
        G = nx.read_edgelist('input/ppi_elist.txt',data=False)
        return G    
    
    else: 
        print('Please choose organism by typing "human" or "yeast"')


def load_genesymbols(G,organism):
    '''
    Load prepared symbols of genes.
    Input: 
    - organism = string; choose from 'human' or 'yeast'

    Return dictionary of geneID (keys) and symbols (values).
    '''  
    if organism == 'yeast':
        df_gID_sym = pd.read_csv('input/DF_gene_symbol_yeast.csv', index_col=0)
        gene_sym = list(df_gID_sym['Sym'])
        gene_id = list(df_gID_sym.index)
        d_gene_sym  = dict(list(zip(gene_id, gene_sym)))
        
        return d_gene_sym 
    
    elif organism == 'human':
        df_gene_sym = pd.read_csv('input/DF_gene_symbol_human.csv', index_col=0)
        sym = list(df_gene_sym['0'])
        l_features = []
        for i in sym:
            l_features.append(i[2:-2])
        d_gene_sym = dict(zip(G.nodes(),l_features))
        
        return d_gene_sym 
  
    else: 
        print('Please choose organism by typing "human" or "yeast"')
        
            
def load_centralities(G,organism):
        '''
        Load prepared centralities of genes.
        Input: 
        - G = Graph
        - organism = string; choose from 'human' or 'yeast'

        Return dictionary with genes as keys and four centrality metrics as values.
        '''
        df_centralities = pd.read_csv('input/Features_centralities_Dataframe_'+organism+'.csv', index_col=0)
        d_deghubs = dict(G.degree()) 
        d_clos = dict(zip(G.nodes(), df_centralities['clos']))
        d_betw = dict(zip(G.nodes(), df_centralities['betw']))
        d_eigen = dict(zip(G.nodes(), df_centralities['eigen']))

        d_centralities = dict(zip(list(G.nodes),zip(d_deghubs.values(),d_clos.values(),d_betw.values(),d_eigen.values())))

        #cent_features = []
        #for i in d_centralities.items():
        #    k=list(i)
        #    cent_features.append(k)
        
        return d_centralities
    
    
def load_essentiality(G, organism):
        '''
        Load prepared essentiality state of organism. 
        Input: 
        - organism = string; choose from 'human' or 'yeast'

        Return lists of genes, split based on essentiality state. 
        '''
        if organism == 'human':
            
            # ESSENTIALITY 
            # get dataframe with ENSG-ID and essentiality state 
            df_human_ess = pd.read_table("input/human_essentiality.txt", delim_whitespace=True)

            # create dict with ENSG-ID:essentiality state 
            ensg_id = list(set(df_human_ess['sciName']))
            gene_ess = list(df_human_ess['locus'])
            d_ensg_ess = dict(zip(ensg_id, gene_ess))

            # match ENSG-ID with entrezID
            # "engs_to_entrezid": entrezIDs were matched with "ensg_id.txt" via "DAVID Database" (https://david.ncifcrf.gov/conversion.jsp)
            df_human_ensg_entrez = pd.read_table('input/ensg_to_entrezid.txt') # delim_whitespace=False)
            df_human_ensg_entrez.dropna()

            df = df_human_ensg_entrez
            df['To'] = df['To'].fillna(0)
            df['To'] = df['To'].astype(int)
            df_human_ensg_entrez = df

            # create dict with ENGS-ID: entrezID
            ensgid = list(df_human_ensg_entrez['From']) #engs ID
            entrezid = list(df_human_ensg_entrez['To']) #entrez ID 

            # dict with engsid : entrezid
            d_ensg_entrez = dict(zip(ensgid, entrezid))

            # create dict with entrezID:essentiality state 
            d_id_ess_unsorted = {}
            for ens,ent in d_ensg_entrez.items():
                for en, ess in d_ensg_ess.items():
                    if ens == en:
                        d_id_ess_unsorted[str(ent)] = ess


            # check if G.nodes match entrezID in dict and sort according to G.nodes 
            d_gid_ess = {}
            for k,v in d_id_ess_unsorted.items():
                if k in G.nodes():
                    d_gid_ess[k]=v

            # create dict with rest of G.nodes not in dict (entrezID:essentiality)
            d_gid_rest = {}
            for g in G.nodes():
                if g not in d_gid_ess.keys():
                    d_gid_rest[g]='not defined'

            #print(len(d_gid_rest)+len(d_gid_ess)) # this should match G.nodes count 

            # merge both dicts
            d_gid_ess_all_unsorted = {**d_gid_ess, **d_gid_rest}

            # sort -> G.nodes()
            d_gID_all = {key:d_gid_ess_all_unsorted[key] for key in G.nodes()}

            essential_genes = []
            non_ess_genes = []
            notdefined_genes = [] 
            for k,v in d_gID_all.items():
                if v == 'E':
                    essential_genes.append(k)
                elif v == 'NE':
                    non_ess_genes.append(k)
                else:
                    notdefined_genes.append(k)
                    
            return essential_genes,non_ess_genes,notdefined_genes
        
        
        elif organism == 'yeast':
            
            # ESSENTIALITY 
            cere_gene =pd.read_csv("input/Saccharomyces cerevisiae.csv",
                       delimiter= ',',
                       skipinitialspace=True)

            cere_sym = list(cere_gene['symbols'])
            cere_ess = list(cere_gene['essentiality status'])
            cere_sym_essentiality = dict(zip(cere_sym, cere_ess))

            d_cere_ess = {}
            d_cere_noess = {}
            d_cere_unknown = {}

            for node,es in cere_sym_essentiality.items():
                if es == 'E':
                    d_cere_ess[node]=es
                elif es == 'NE':
                    d_cere_noess[node]=es
                    
            df_gID_sym = pd.read_csv('input/DF_gene_symbol_yeast.csv', index_col=0)
            gene_sym = list(df_gID_sym['Sym'])
            gene_id = list(df_gID_sym.index)
            g_ID_sym = dict(list(zip(gene_id, gene_sym)))
            
            d_cere_alless = {}
            for nid, sym in g_ID_sym.items():
                for sy,ess in cere_sym_essentiality.items():
                    if sym == sy:
                        d_cere_alless[nid] = ess

            d_cere_unknown = {} 
            for g in G.nodes():
                if g not in d_cere_alless.keys():
                    d_cere_unknown[g]='status unkonwn'

            d_geneID_ess = {**d_cere_unknown, **d_cere_alless}

            d_gID_ess = {}
            d_gID_noess = {}
            d_gID_notdef = {}

            for k,i in d_geneID_ess.items():
                if i == 'E':
                    d_gID_ess[k] = i
                elif i == 'NE':
                    d_gID_noess[k] = i
                else: 
                    d_gID_notdef[k] = 'not defined'

            d_gID_all_unsorted = {**d_gID_ess, **d_gID_noess, **d_gID_notdef}
            d_gID_all = {key:d_gID_all_unsorted[key] for key in G.nodes()}

            essential_genes = []
            non_ess_genes = []
            notdefined_genes = [] 
            for k,v in d_gID_all.items():
                if v == 'E':
                    essential_genes.append(k)
                elif v == 'NE':
                    non_ess_genes.append(k)
                else:
                    notdefined_genes.append(k)
            
            return essential_genes,non_ess_genes,notdefined_genes

        else:
            print('Please choose organism by typing "human" or "yeast"')

            
def load_datamatrix(G,organism,netlayout):
    '''
    Load precalculated Matrix with N genes and M features.
    Input: 
    - path = directory of file location
    - organism = string; choose from 'human' or 'yeast'
    - netlayout = string; choose a network layout e.g. 'local', 'global', 'importance', 'funct-bio', 'funct-cel', 'funct-mol', funct-dis'

    Return Matrix based on choice.
    '''
    path = 'input/'
    
    if netlayout == 'local':
        return pd.read_pickle(path+'Adjacency_Dataframe_'+organism+'.pickle')
    
    elif netlayout == 'global':
        return pd.read_pickle(path+'RWR_Dataframe_'+organism+'.pickle')
    
    elif netlayout == 'importance':
        
        d_centralities = load_centralities(G, organism)
        df_centralities = pd.DataFrame(d_centralities).T

        DM_centralities = pd.DataFrame(distance.squareform(distance.pdist(df_centralities, 'cosine')))
        DM_centralities = round(DM_centralities,6)
        DM_centralities.index = list(G.nodes())
        DM_centralities.columns = list(G.nodes())
        
        return DM_centralities
    
    elif netlayout == 'funct-bio' and organism == 'human':
        return pd.read_pickle('input/DistanceMatrix_goBP_Dataframe_Human_cosine.pickle') #pd.read_pickle('input/Features_BioProc_Dataframe_human.pickle')
    
    elif netlayout == 'funct-mol' and organism == 'human':
        return pd.read_pickle('input/DistanceMatrix_goMF_Dataframe_Human_cosine.pickle') #pd.read_pickle('input/Features_MolFunc_Dataframe_human.pickle')
    
    elif netlayout == 'funct-cel' and organism == 'human':
        return pd.read_pickle('input/DistanceMatrix_goCC_Dataframe_Human_cosine.pickle') #pd.read_pickle('input/Features_GO_CellComp_Dataframe_human.pickle')
    
    elif netlayout == 'funct-dis' and organism == 'human':
        return pd.read_pickle('input/DistanceMatrix_Disease_Dataframe_Human_cosine.pickle') #pd.read_pickle('input/Features_Disease_Dataframe_human.pickle')
    
    else: 
        print('Please type one of the following: "local", "global", "importance", "funct-dis/bio/cel/mol"')

        
        
        