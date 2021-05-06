

######################
#
# A D D - O N 
# Benchmarking - U P S C A L I N G - 2D CAYLEY TREE 
#
#####################

from multidimvis_main import *

# 5 k > corr coeff 0.97
branch=4
i=5461


# 10 k > corr coeff 0.98
#branch=3
#i=9841

# 20 k > corr coeff 0.95
#branch=5
#i=19531


G = nx.full_rary_tree(branch,i)

print('Runtime spring for Corr. Coeff. 0.97')

start = time.time()
posG_spring = springlayout_2D(G,itr=1000)
end = time.time()
m,s = exec_time(start,end)

dist_spring = pairwise_layout_distance_2D(G,posG_spring)
print('distances spring done')
dist_network = pairwise_network_distance(G)
print('distances network done')

r_spring = pearson_corrcoef(dist_network, dist_spring)
print('SPRING corr coef: ',r_spring)