

######################
#
# TORUS 
# Benchmarking - U P S C A L I N G 
# save network distances to files 
#
#####################

from benchmark_main import *

#100
#i=100
#G = nx.grid_2d_graph(12,8,periodic=True)

#500
#i=500
#G = nx.grid_2d_graph(42,12,periodic=True)

#1k
#i=1000
#G = nx.grid_2d_graph(44,22,periodic=True)

#5k
#i=5000
G = nx.grid_2d_graph(90,55,periodic=True)

#10k
i=10k
G = nx.grid_2d_graph(125,80,periodic=True)

#20k
#i=20k
#G = nx.grid_2d_graph(182,110,periodic=True)

print(len(G.nodes()))

print('calculate network distance')
dist_network = pairwise_network_distance(G)
print('distances network done')

a_file = open('netdist_precalc/dist_network_'+str(i)+'_torus.pkl', "wb")
pickle.dump(dist_network, a_file)
a_file.close()

#b_file = open('dist_network_'+str(i)+'_torus.pkl', "rb")
#dist_network = pickle.load(b_file)
#print(len(dist_network))