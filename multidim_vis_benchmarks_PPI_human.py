from multidimvis_main import *

################################################
#
# H U M A N 
#
################################################

organism = 'Human'

G = nx.read_edgelist('input/ppi_elist.txt',data=False)

#######################
#
# NETWORK DISTANCE PRECALCULATED 
# 
#######################

DM_spl_prec_human_int = pd.read_csv('_output_csv/SPL_Dataframe_int_Human.csv', index_col=0)
DM_spl_prec_human_int.index = list(G.nodes())
DM_spl_prec_human_int.columns = list(G.nodes())
print('reading done')

d_DM_spl = DM_spl_prec_human_int.to_dict()
print('dataframe to dict done')

d_SPL_pairs = {}
for k,d in d_DM_spl.items():
    for n,v in d.items():
        d_SPL_pairs[k,n]=v

print('dict done')
dist_network = d_SPL_pairs

del d_DM_spl

print('Network:', organism)
print('Number of nodes: %s' %len(list(G.nodes())))
print('Number of edges: %s' %len(list(G.edges())))
print('Network density: %.11f%%' %(200.*len(list(G.edges()))/(len(list(G.nodes()))*len(list(G.nodes()))-1)))


# GLOBAL LAYOUT 
#-----------------------------
# UMAP PARAMETERS:
#-----------------------------
n_neighbors = 20 
spread = 0.8
min_dist = 0.0
metric = 'cosine'
lnr = 1
nep = None

feature = 'RWRvis'
r = .9
alpha = 1.0

A = nx.adjacency_matrix(G)
FM_m_array = rnd_walk_matrix2(A, r, alpha, len(G.nodes()))
DM_rwr = pd.DataFrame(FM_m_array).T

del A
del FM_m_array

umap_rwr_3D = embed_umap_3D(DM_rwr, n_neighbors, spread, min_dist, metric)
posG_umap_rwr = get_posG_3D(list(G.nodes()), umap_rwr_3D)
posG_complete_umap_rwr = {key:posG_umap_rwr[key] for key in G.nodes()}

df_posG = pd.DataFrame(posG_complete_umap_rwr).T
x = df_posG.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_posG_norm = pd.DataFrame(x_scaled)

posG = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values,df_posG_norm[2].values)))
    
del DM_rwr
del df_posG

dist_network = d_SPL_pairs

print('prep layout distance')
dist_layout3D = {} 
for p1,p2 in it.combinations(G.nodes(),2):
    dist_layout3D[(p1,p2)] = np.sqrt((posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2 + (posG[p1][1]-posG[p2][2])**2)

print('prep layout / network distance')
d_plot_layout = {}
for spldist in range(1,int(max(dist_network.values()))+1):
    l_s = []
    for k, v in dist_network.items():
        if v == spldist:
            l_s.append(k)
    print('list '+str(spldist)+' done')
    l_xy = []
    for nodes in l_s:
        if nodes in dist_layout3D.keys():
            dxy = dist_layout3D[nodes]
            l_xy.append(dxy)
    d_plot_layout[spldist] = l_xy
    print('dict '+str(spldist)+' done')

max_dist_network = int(max(dist_network.values()))
del dist_network 

print('calculate corr. coeff.')

# GLOBAL RWR Pearson Correlation
l_medians_layout = []
for k, v in d_plot_layout.items():
    l_medians_layout.append(statistics.median(v))
print('list done')

x = np.array(range(1,max_dist_network+1))
y = np.array(l_medians_layout)
r_layout = np.corrcoef(x, y)

print('GLOBAL (RWR) Pearson Correlation Factor: ', r_layout[0][1])

del dist_layout3D 

print('all RWR done')

# SPRING LAYOUT
itr = 50 #20

posG_spring3D = nx.spring_layout(G, iterations = itr, dim = 3)

# NORMALIZED COORDINATES FOR CORRELATION CALCULATION

print('normalize coordinates')
df_posG = pd.DataFrame(posG_spring3D).T
x = df_posG.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_posG_norm = pd.DataFrame(x_scaled)
posG_spring3D_norm = dict(zip(list(G.nodes()),zip(df_posG_norm[0].values,df_posG_norm[1].values, df_posG_norm[2].values)))

print('get layout distances')
posG = posG_spring3D_norm
dist_spring3D = {} 
for p1,p2 in it.combinations(G.nodes(),2):
    #dist_spring2D[(p1,p2)] = (posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2
    dist_spring3D[(p1,p2)] = np.sqrt((posG[p1][0]-posG[p2][0])**2 + (posG[p1][1]-posG[p2][1])**2+(posG[p1][2]-posG[p2][2])**2)

dist_network = d_SPL_pairs

print('prep layout distances')
print(int(max(dist_network.values())))

d_plot_spring = {}
for spldist in range(1,int(max(dist_network.values()))+1):
    l_s = []
    for k, v in dist_network.items():
        if v == spldist:
            l_s.append(k)
    print('list '+str(spldist)+' done')
    l_xy = []
    for nodes in l_s:
        if nodes in dist_spring3D.keys():
            dxy = dist_spring3D[nodes]
            l_xy.append(dxy)
    d_plot_spring[spldist] = l_xy
    print('dict '+str(spldist)+' done')

max_dist_network = int(max(dist_network.values()))
del dist_network 

print('calculate corr. coeff.')

# SPRING Pearson Correlation
l_medians_spring = []
for k, v in d_plot_spring.items():
    l_medians_spring.append(statistics.median(v))
print('list done') 

x = np.array(range(1,max_dist_network+1))
y = np.array(l_medians_spring)
r_spring = np.corrcoef(x, y)

print('SPRING Pearson Correlation Factor: ', r_spring[0][1])

##############
# DIAGRAM 
##############

offset = 0.2 

fig, ax = plt.subplots(figsize =(16,12), dpi=300)
bp_spring = ax.boxplot(d_plot_spring.values(), positions=[i+offset for i in list(d_plot_spring.keys())], widths=0.3, patch_artist=True, showfliers=False)
bp_layout = ax.boxplot(d_plot_layout.values(), positions=[i-offset for i in list(d_plot_layout.keys())], widths=0.3, patch_artist=True, showfliers=False)

for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(bp_spring[element], color='dimgrey', linewidth=0.6)
    plt.setp(bp_layout[element], color='dimgrey', linewidth=0.6)

for patch in bp_spring['boxes']:
    patch.set(facecolor='lightblue', linewidth=0.5)
    
for patch in bp_layout['boxes']:
    patch.set(facecolor='orange', linewidth=0.5)

plt.xlabel('Network Distance', fontsize=22)
plt.ylabel('Layout Distance', fontsize=22)

plt.xticks(range(1,len(d_plot_spring.keys())+1),d_plot_spring.keys(), fontsize=14)
plt.yticks(fontsize=14)


plt.title('3D | '+organism, fontsize=14)
plt.suptitle('Pears corr coef: '+'\n'+'SPRING: '+str(r_spring[0][1])+'\n'+ 'GLOBAL: '+str(r_layout[0][1]), fontsize=10)

plt.show()
fig.savefig('output_plots/benchmark/3Dtoynetwork_NetworkDistances_'+'springitr'+str(itr)+'_'+organism+'.png')