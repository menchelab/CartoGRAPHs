from benchmark_main import *
from fa2 import ForceAtlas2


b_file20k = open('netdist_precalc/dist_network_19531_tree.pkl', "rb")
dist_network20k = pickle.load(b_file20k)
print('20k done')

# pearson correlation coefficient 

d_netsize_corr_global = {121:0.975,
                            511:0.992, 
                            1093:0.984,
                            5461:0.979, 
                            9841:0.977,
                            19531:0.952
                        }

# time in seconds 

d_netsize_time = {121:2, 
                    511:5, 
                    1093:6, 
                    5461:43,
                    9841:76, #1min 16s 
                    19531:269, #4min 29s
                    }


print('Run for', d_netsize_time[19531],'seconds')

#########
# 2 0 0 0 0
#########

branch=5
i=19531
G = nx.full_rary_tree(branch,i)

print('SPRING - forceAtlas2')
start = time.time()
forceatlas2 = ForceAtlas2(verbose=False)
posG_spring = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=1000)
end = time.time()
m,s = exec_time(start,end)

start = time.time()
print('distance calculation')
dist_spring = pairwise_layout_distance_2D(G,posG_spring)
end = time.time()
m,s = exec_time(start,end)

print('calculate pearson corr.coef.')

one = pearson_corrcoef_one(dist_network20k, dist_spring)
print('first step done')

two = pearson_corrcoef_two(one)
print('second step done')

r_spring = pearson_corrcoef_two(two)
print('SPRING corr coef: ',r_spring)