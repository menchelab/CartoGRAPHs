 
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains Layouts generated with the STELLARGRAPH module
# 
########################################################################################

import stellargraph as sg
from stellargraph import StellarGraph 
from tensorflow import keras
import gensim
from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification
from stellargraph.mapper import GraphWaveGenerator
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator

########################################################################################


#--------------------
#
# N O D E 2 V E C 
#
#--------------------

def layout_nodevec_tsne(G,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
    walk_lngth = 50
    num_wlks = 10
    wrks = 1
    dmns = 50 
    
    # old version of Node2Vec library (by shenweichen "Graph Embeddings" on github)
    #node2vec = Node2Vec(G, dimensions=dmns, walk_length=walk_lngth, num_walks=num_wlks, workers=wrks, quiet=True)
    #model = node2vec.fit(window=10, min_count=1)
    
    # using Stellargraph Library instead 
    walk_length = 100  # maximum length of a random walk to use throughout this notebook
    stellarG = StellarGraph.from_networkx(G)

    rw = BiasedRandomWalk(stellarG)

    weighted_walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    weighted_model = gensim.models.Word2Vec(weighted_walks, 
                                            vector_size=128, 
                                            window=5, 
                                            min_count=0, 
                                            sg=1, 
                                            workers=1, 
                                            epochs=1)
    
    arr = np.array([weighted_model.wv[x] for x in G.nodes()])
    DM = pd.DataFrame(arr)
    DM.index = list(G.nodes())

    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
def layout_nodevec_umap(G,dim,n_neighbors=20, spread=1.0, min_dist=0.0, metric='cosine'):
    
    walk_lngth = 50
    num_wlks = 10
    wrks = 1
    dmns = 50 
    
    # old version of Node2Vec library (by shenweichen "Graph Embeddings" on github)
    #node2vec = Node2Vec(G, dimensions=dmns, walk_length=walk_lngth, num_walks=num_wlks, workers=wrks, quiet=True)
    #model = node2vec.fit(window=10, min_count=1)
    
    # using Stellargraph Library instead 
    walk_length = 100  # maximum length of a random walk to use throughout this notebook
    stellarG = StellarGraph.from_networkx(G)

    rw = BiasedRandomWalk(stellarG)

    weighted_walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    weighted_model = gensim.models.Word2Vec(weighted_walks, 
                                            vector_size=128, 
                                            window=5, 
                                            min_count=0, 
                                            sg=1, 
                                            workers=1, 
                                            epochs=1)
    
    arr = np.array([weighted_model.wv[x] for x in G.nodes()])
    DM = pd.DataFrame(arr)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


#--------------------
#
# A T T R I 2 V E C 
#
#--------------------

def layout_attrivec_tsne(G,d_features,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
    stellarG = StellarGraph.from_networkx(G, node_features=d_features)
    nodes = list(stellarG.nodes())
    number_of_walks = 4
    length = 5

    unsupervised_samples = UnsupervisedSampler(stellarG, 
                                               nodes=nodes, 
                                               length=length, 
                                               number_of_walks=number_of_walks)

    batch_size = 50
    epochs = 4
    
    generator = Attri2VecLinkGenerator(stellarG, batch_size)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [128]
    attri2vec = Attri2Vec(layer_sizes=layer_sizes, 
                          generator=generator, 
                          bias=False, 
                          normalize=None)

    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.in_out_tensors()

    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = Attri2VecNodeGenerator(stellarG, batch_size).flow(stellarG.nodes())
    embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)
    DM = pd.DataFrame(embeddings)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
        
def layout_attrivec_umap(G,d_features,dim,n_neighbors=20, spread=1.0, min_dist=0.0, metric='cosine'):

    stellarG = StellarGraph.from_networkx(G, node_features=d_features)
    nodes = list(stellarG.nodes())
    number_of_walks = 4
    length = 5

    unsupervised_samples = UnsupervisedSampler(stellarG, 
                                               nodes=nodes, 
                                               length=length, 
                                               number_of_walks=number_of_walks)

    batch_size = 50
    epochs = 4
    
    generator = Attri2VecLinkGenerator(stellarG, batch_size)
    train_gen = generator.flow(unsupervised_samples)
    layer_sizes = [128]
    attri2vec = Attri2Vec(layer_sizes=layer_sizes, 
                          generator=generator, 
                          bias=False, 
                          normalize=None)

    # Build the model and expose input and output sockets of attri2vec, for node pair inputs:
    x_inp, x_out = attri2vec.in_out_tensors()

    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_gen = Attri2VecNodeGenerator(stellarG, batch_size).flow(stellarG.nodes())
    embeddings = embedding_model.predict(node_gen, workers=1, verbose=0)
    DM = pd.DataFrame(embeddings)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
        
#--------------------
#
# G R A P H W A V E 
#
#--------------------


def layout_graphwave_tsne(G,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
    #features = compute_centralityfeatures(G)
    #d_features = pd.DataFrame(features).T
    #d_features.index = list(G.nodes())

    stellarG = StellarGraph.from_networkx(G)#, node_features=d_features)
    
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    degree = 20
    scales = [5, 10]

    generator = GraphWaveGenerator(stellarG, scales=scales, degree=degree)

    embeddings_dataset = generator.flow(
        node_ids=G.nodes(), 
        sample_points=sample_points, 
        batch_size=1, repeat=False)

    embeddings_notstacked = [x.numpy() for x in embeddings_dataset]
    embeddings = np.vstack(embeddings_notstacked)
    DM = pd.DataFrame(embeddings)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')


        
def layout_graphwave_umap(G,dim,n_neighbors=20, spread=1.0, min_dist=0.0, metric='cosine'):
    
    #features = compute_centralityfeatures(G)
    #d_features = pd.DataFrame(features).T
    #d_features.index = list(G.nodes())
    
    stellarG = StellarGraph.from_networkx(G)#, node_features=d_features)
    
    sample_points = np.linspace(0, 100, 50).astype(np.float32)
    degree = 20
    scales = [5, 10]

    generator = GraphWaveGenerator(stellarG, scales=scales, degree=degree)

    embeddings_dataset = generator.flow(
        node_ids=G.nodes(), 
        sample_points=sample_points, 
        batch_size=1, repeat=False)

    embeddings_notstacked = [x.numpy() for x in embeddings_dataset]
    embeddings = np.vstack(embeddings_notstacked)
    DM = pd.DataFrame(embeddings)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
    
#--------------------
#
# M E T A P A T H 2 V E C 
#
#--------------------
        
        
def layout_metapathvec_tsne(G,dim,prplxty=50, density=12, l_rate=200, steps=250, metric='cosine'):
    
    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    A_array = A.toarray()
    DM = pd.DataFrame(A_array, columns = list(G.nodes()), index=list(G.nodes()))
    
    stellarG = StellarGraph.from_networkx(G, node_features=DM)
    nodes = list(stellarG.nodes())
    number_of_walks = 1
    length = 5
    
    unsupervised_samples = UnsupervisedSampler(stellarG, nodes=nodes, length=length, number_of_walks=number_of_walks)
    batch_size = 50
    epochs = 4
    num_samples = [10, 5]
    
    generator = GraphSAGELinkGenerator(stellarG, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    
    layer_sizes = [50, 50]
    graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2")
    
    x_inp, x_out = graphsage.in_out_tensors()
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    
    node_ids = list(G.nodes())
    node_gen = GraphSAGENodeGenerator(stellarG, batch_size, num_samples).flow(node_ids)
    
    embeddings = embedding_model.predict(node_gen, workers=4, verbose=0)
    DM = pd.DataFrame(embeddings)
    DM.index = list(G.nodes())

    if dim == 2:
        r_scale = 1.2
        tsne2D = embed_tsne_2D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_2D_norm(G, DM, tsne2D) #, r_scale)
        
        return posG
    
    elif dim == 3: 
        r_scale = 1.2
        tsne3D = embed_tsne_3D(DM, prplxty, density, l_rate, steps, metric)
        posG = get_posG_3D_norm(G, DM, tsne3D) #, r_scale)

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')

        
        
def layout_metapathvec_umap(G,dim,n_neighbors=20, spread=1.0, min_dist=0.0, metric='cosine'):

    A = nx.adjacency_matrix(G, nodelist=list(G.nodes()))
    A_array = A.toarray()
    DM = pd.DataFrame(A_array, columns = list(G.nodes()), index=list(G.nodes()))
    
    stellarG = StellarGraph.from_networkx(G, node_features=DM)
    nodes = list(stellarG.nodes())
    number_of_walks = 1
    length = 5
    
    unsupervised_samples = UnsupervisedSampler(stellarG, nodes=nodes, length=length, number_of_walks=number_of_walks)
    batch_size = 50
    epochs = 4
    num_samples = [10, 5]
    
    generator = GraphSAGELinkGenerator(stellarG, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
    
    layer_sizes = [50, 50]
    graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2")
    
    x_inp, x_out = graphsage.in_out_tensors()
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    
    node_ids = list(G.nodes())
    node_gen = GraphSAGENodeGenerator(stellarG, batch_size, num_samples).flow(node_ids)
    
    embeddings = embedding_model.predict(node_gen, workers=4, verbose=0)
    DM = pd.DataFrame(embeddings)
    DM.index = list(G.nodes())
    
    if dim == 2:
        r_scale = 1.2
        umap2D = embed_umap_2D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_2D_norm(G, DM, umap2D) #r_scale
        
        return posG
    
    elif dim == 3: 
        umap_3D = embed_umap_3D(DM, n_neighbors, spread, min_dist, metric)
        posG = get_posG_3D_norm(G, DM, umap_3D) #r_scale

        return posG
        
    else:
        print('Please choose dimensions, by either setting dim=2 or dim=3.')
