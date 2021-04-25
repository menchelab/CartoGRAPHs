
from multidimvis_main import *

# add functions to multidimvis_main 

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
        n_epochs = n_ep)
    
    embed = U.fit_transform(Matrix)
    
    return embed


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 3)), int(round(m, 0)), int(round(h, 0))
    print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
   
    return m,s


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
    
#---------------------------------------

itr = 50 

d_netsize_branch1 = {#1093:3, 
                    #5461:4} 
                    #9841:3,
                    19531:5,
                    #597871:9,
                    #1398101:4,
                    #5592404:4,
                    #12093235:6
                    }

d_walltime_spring1 = {}
d_corr_spring1 = {}
d_layoutdicts_spring1 = {}
d_graphedges_spring1 = {}

d_walltime_layout1 = {}
d_corr_layout1 = {}
d_layoutdicts_layout1 = {}
d_graphedges_layout1 = {}

for i,branch in d_netsize_branch1.items():
    
    G = nx.full_rary_tree(branch,i)
    #d_graphedges_spring1[i] = list(G.edges())
    #d_graphedges_layout1[i] = list(G.edges())

    #-----------------------------
    
    print('--- Spring ---')
    
    start = time.time()
    posG_spring2D = nx.spring_layout(G, iterations = itr, dim = 2)

    df_posG = pd.DataFrame(posG_spring2D).T
    x = df_posG.values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)

    posG_spring2D_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))

    end = time.time()

    #===========================
    # SPRING save layout as dict + append to dict { netsize : {nodeid:xyz,nodeid:xyz,...}, netsize : {nodeid:xyz, nodeid:xyz,...}}
    #===========================
    #d_layoutdicts_spring1[i] = posG_spring2D_norm

    print('# Nodes (netsize):', i)
    m,s = exec_time(start,end)

    #===========================
    # SPRING save WALLTIME for layout in dict : { netsize : walltime }
    #===========================
    walltime = s+m*60
    d_walltime_spring1[i] = walltime
    print('walltime - spring: ',walltime)

    #-----------------------------
    
    print('--- RWR ---')

    start = time.time()
    r = .9 
    alpha = 1.0

    A = nx.adjacency_matrix(G)
    FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
    DM_rwr = pd.DataFrame(FM_m_array)

    n_neighbors = 20
    spread = 1.0
    min_dist = 0.01
    metric = 'cosine'
    lnr = 1 
    nep = None 

    umap_rwr = embed_umap_2D(DM_rwr, n_neighbors, spread, min_dist, metric, learn_rate = lnr, n_ep = nep)
    posG_umap_rwr = get_posG_2D(list(G.nodes()), umap_rwr)
    posG_complete_umap_rwr = {key:posG_umap_rwr[key] for key in G.nodes()}
    df_posG = pd.DataFrame(posG_complete_umap_rwr).T

    x = df_posG.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_posG_norm = pd.DataFrame(x_scaled)

    posG_complete_umap_rwr_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values)))
    
    end = time.time()
    
    #===========================
    # RWR save layout as dict + append to dict { netsize : {nodeid:xyz,nodeid:xyz,...}, netsize : {nodeid:xyz, nodeid:xyz,...}}
    #===========================
    #d_layoutdicts_layout1[i] = posG_complete_umap_rwr_norm 

    #===========================
    # RWR save WALLTIME for layout in dict : { netsize : walltime }
    #===========================
    m,s = exec_time(start,end)
    walltime = s+m*60

    d_walltime_layout1[i] = walltime
    print('walltime - layout: ',walltime)

    #===========================
    # get and save corr.fact. to dict : { netsize : corr.fact. }
    #===========================
    print('--- calculate SPL ---')
    spl = nx.all_pairs_shortest_path_length(G)
    DM_spl_2D = pd.DataFrame(dict(spl))
    print('--- SPL done ---')

    dist_spring2D = {} 
    for (id1,p1),(id2,p2) in it.combinations(posG_spring2D_norm.items(),2):
        dx,dy = p1[0]-p2[0], p1[1]-p2[1]
        dist_spring2D[id1,id2] = np.sqrt(dx*dx+dy*dy)
    
    dist_network2D = {}
    for p1, p2 in it.combinations(DM_spl_2D.index,2):
        dist_network2D[p1,p2] = DM_spl_2D[p1][p2]    
    
    dist_layout2D = {} 
    for (id1,p1),(id2,p2) in it.combinations(posG_complete_umap_rwr_norm.items(),2):
        dx,dy = p1[0]-p2[0], p1[1]-p2[1]
        dist_layout2D[id1,id2] = np.sqrt(dx*dx+dy*dy)
    
    y_spring = list(dist_spring2D.values())
    x_spring = list(dist_network2D.values())
    
    y_layout = list(dist_layout2D.values())
    x_layout = list(dist_network2D.values())
   
    print('--- calculate correlation factors ---')

    gradient_spring, intercept_spring, corr_spring, p_value_spring, std_err_spring = stats.linregress(x_spring,y_spring)
    d_corr_spring1[i] = corr_spring
    print('Corr Fact - Spring: ', corr_spring)
    
    gradient_layout, intercept_layout, corr_layout, p_value_layout, std_err_layout = stats.linregress(x_layout,y_layout)
    d_corr_layout1[i] = corr_layout
    print('Corr Fact - Layout: ', corr_layout)

    print('--- next graph ---')