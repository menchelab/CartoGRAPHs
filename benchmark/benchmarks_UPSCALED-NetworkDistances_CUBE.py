

######################
#
# CUBE 
# Benchmarking - U P S C A L I N G 
# save network distances to files 
#
#####################

from benchmark_main import *

#100
#G = nx.grid_graph([5,5,5],periodic=False)

#500
#G = nx.grid_graph([8,8,8],periodic=False)

#1k
#G = nx.grid_graph([10,10,10],periodic=False)

#5k
G = nx.grid_graph([18,18,18],periodic=False)

#10k
G = nx.grid_graph([22,22,22],periodic=False)

#20k
#G = nx.grid_graph([28,28,28],periodic=False)

print(len(G.nodes()))

print('calculate network distance')
dist_network = pairwise_network_distance(G)
print('distances network done')

a_file = open('netdist_precalc/dist_network_'+str(len(G.nodes()))+'_cube.pkl', "wb")
pickle.dump(dist_network, a_file)
a_file.close()
print('save done')