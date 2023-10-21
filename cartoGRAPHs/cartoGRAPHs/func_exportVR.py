  
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains  E X P O R T   C O O R D I N A T E S   F U N C T I O N S 
# compatible/for uplad to VRNetzer Platform and Webapplication "cartoGRAPHs"
#
########################################################################################

from cartoGRAPHs.func_visual_properties import *
import pandas as pd
import json 

########################################################################################

# ----------------------------
# UPLOAD CSV 
# ----------------------------
def exportVR_CSV(filename, G, posG, d_node_colors, d_annotations, linkcolor,clusterlabels = None):
    """
    Export tables for the CSV file uploader of the VRNetzer (beta release, april2023)
        
    G: nx.Graph object
    d_node_colors: dict with keys=nodeID and values=color for each node in hex or rgba
    d_annotations: dict with keys=nodeID and values=annotations for each node as string, divided with ";" 
    linkcolor: hex value or dict with keys=link and values=color for each link in Graph 
    clusterlabels: list of sublist for each cluster, containing clustername (string) and all nodes (IDs) assigned per sublist
    
    Returns:
        CSV tables to be uploaded to the VRNetzer Backend including list of all files generated. 
    """
    
    # modify filename to not contain any spaces :
    filename = filename.replace(" ", "")
    
    # NODE POSITIONS 
    df_nodepos = pd.DataFrame()
    
    if len(list(posG.values())[0]) == 2:
        df_nodepos['x']=[i[0] for i in posG.values()]
        df_nodepos['y']=[i[1] for i in posG.values()]
        df_nodepos['z']=[0 for i in posG.values()]
        df_nodepos.to_csv(filename+'_nodepositions.csv', header=None, index=0)
        #print("Exported File: ", filename+'_nodepositions.csv')

    elif len(list(posG.values())[0]) == 3: 
        df_nodepos['x']=[i[0] for i in posG.values()]
        df_nodepos['y']=[i[1] for i in posG.values()]
        df_nodepos['z']=[i[2] for i in posG.values()]
        df_nodepos.to_csv(filename+'_nodepositions.csv', header=None, index=0)
        #print("Exported File: ", filename+'_nodepositions.csv')


    # NODE COLORS 
    df_nodecol = pd.DataFrame()
    df_nodecol['r']=[hex_to_rgb(i)[0] if type(i)==str else i[0] for i in d_node_colors.values()]
    df_nodecol['g']=[hex_to_rgb(i)[1] if type(i)==str else i[1] for i in d_node_colors.values()]
    df_nodecol['b']=[hex_to_rgb(i)[2] if type(i)==str else i[2] for i in d_node_colors.values()]
    df_nodecol['a']=[100 if type(i)==str else i[3] for i in d_node_colors.values()]

    df_nodecol.to_csv(filename+'_nodecolors.csv', header=None, index=0)
    #print("Exported File: ", filename+'_nodecolors.csv')


    # NODE PROPERTIES
    df_nodeprop = pd.DataFrame()
    df_nodeprop['prop'] = list(d_annotations.values())

    df_nodeprop.to_csv(filename+'_nodeproperties.csv', header=None, index=0)
    #print("Exported File: ", filename+'_nodeproperties.csv')


    # LINKS
    mapping = dict(zip(list(G.nodes()),list(range(0,len(G.nodes())))))
    G = nx.relabel_nodes(G, mapping)
 
    df_links = pd.DataFrame()
    df_links['start'] = [i[0] for i in list(G.edges())]
    df_links['end'] = [i[1] for i in list(G.edges())]

    df_links.to_csv(filename+'_links.csv', header=None, index=0)
    #print("Exported File: ", filename+'_links.csv')


    # LINK COLORS
    df_linkcol = pd.DataFrame()
    df_linkcol['r']=[hex_to_rgb(i)[0] if type(i)==str else i[0] for i in linkcolor.values()]
    df_linkcol['g']=[hex_to_rgb(i)[1] if type(i)==str else i[1] for i in linkcolor.values()]
    df_linkcol['b']=[hex_to_rgb(i)[2] if type(i)==str else i[2] for i in linkcolor.values()]
    df_linkcol['a']=[80 if type(i)==str else i[3] for i in linkcolor.values()]

    df_linkcol.to_csv(filename+'_linkcolors.csv', header=None, index=0)
    #print("Exported File: ", filename+'_linkcolors.csv')
    

    # CLUSTER LABELS
    if clusterlabels != None:
        df_labels = pd.DataFrame(clusterlabels)
        df_labels.to_csv(filename+'_clusterlabels.csv', header=None, index=0)
        #print("Exported File: ", filename+'_clusterlabels.csv')
            
    else:
        pass

    return print("Export done.")



# ----------------------------
# UPLOAD JSON 
# ----------------------------
def exportVR_JSON(filename, G,posG,d_node_colors, d_annotations, linkcolor, dict_for_cluster=None):
    """
    Export a Graph including attributes for the JSON file uploader of the VRNetzer (beta release, april2023)
        
    G: nx.Graph object
    dict_for_cluster: dict with keys=nodeID and values=cluster assigned
    d_node_colors: dict with keys=nodeID and values=color for each node in hex or rgba
    d_annotations: dict with keys=nodeID and values=annotations for each node as string, divided with ";" 
    linkcolor: hex value or dict with keys=link and values=color for each link in Graph 
    
    Returns:
        Graph including attributes as JSON file to be uploaded to the VRNetzer Backend. 
    """
    
    # modify filename to not contain any spaces :
    filename = filename.replace(" ", "")
    
    if len(list(posG.values())[0]) == 2:
        new_posG = {}
        for k,v in posG.items():
            new_posG[k] = (v[0],v[1],0)
    else:
        new_posG = posG
        
    # set graph attributes 
    nx.set_node_attributes(G, new_posG, name="pos")
    nx.set_node_attributes(G, d_node_colors, name="nodecolor")
    nx.set_node_attributes(G, d_annotations, name="annotation") 
    nx.set_edge_attributes(G, linkcolor, name="linkcolor")
    
    if dict_for_cluster != None:
        nx.set_node_attributes(G, dict_for_cluster, name="cluster")
    else:
        pass

    mapping = dict(zip(list(G.nodes()),list(range(0,len(G.nodes())))))
    G = nx.relabel_nodes(G, mapping)

    G_json = json.dumps(nx.node_link_data(G))

    with open(filename+".json", "w") as outfile:
        outfile.write(G_json)
    
    return print("Exported File: \n", [filename+".json"])









# -----------------------------------
# O L D 
# -----------------------------------

# def export_to_csv2D(path, layout_namespace, posG, colors = None):
#     '''
#     Generate csv for upload to VRnetzer plaform for 2D layouts. 
#     Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
#     '''
    
#     if colors is None: 
#         colors_hex2rgb = list((255,0,0) for i in list(posG.keys()))
    
#     else: 
#         colors_hex2rgb = []
#         for j in colors: 
#             k = hex_to_rgb(j)
#             colors_hex2rgb.append(k)

#     colors_r = []
#     colors_g = []
#     colors_b = []
#     colors_a = []
#     for i in colors_hex2rgb:
#         colors_r.append(int(i[0]))#*255)) # color values should be integers within 0-255
#         colors_g.append(int(i[1]))#*255))
#         colors_b.append(int(i[2]))#*255))
#         colors_a.append(100) # 0-100 shows normal colors in VR, 128-200 is glowing mode

#     df_2D = pd.DataFrame(posG).T
#     df_2D.columns=['X','Y']
#     df_2D['Z'] = 0
#     df_2D['R'] = colors_r
#     df_2D['G'] = colors_g
#     df_2D['B'] = colors_b
#     df_2D['A'] = colors_a

#     df_2D[layout_namespace] = layout_namespace
#     df_2D['ID'] = list(posG.keys())

#     cols = df_2D.columns.tolist()
#     cols = cols[-1:] + cols[:-1]
#     df_2D_final = df_2D[cols]
    
#     return df_2D_final.to_csv(r''+path+layout_namespace+'_layout.csv',index=False, header=False)
    

# def export_to_csv3D(path, layout_namespace, posG, colors = None):
#     '''
#     Generate csv for upload to VRnetzer plaform for 3D layouts. 
#     Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
#     '''
    
#     if colors is None: 
#         colors_hex2rgb = list((255,0,0) for i in list(posG.keys()))
    
#     else: 
#         colors_hex2rgb = []
#         for j in colors: 
#             k = hex_to_rgb(j)
#             colors_hex2rgb.append(k)
            
#     colors_r = []
#     colors_g = []
#     colors_b = []
#     colors_a = []
#     for i in colors_hex2rgb:
#         colors_r.append(int(i[0]))#*255)) # color values should be integers within 0-255
#         colors_g.append(int(i[1]))#*255))
#         colors_b.append(int(i[2]))#*255))
#         colors_a.append(100) # 0-100 shows normal colors in VR, 128-200 is glowing mode
        
#     df_3D = pd.DataFrame(posG).T
#     df_3D.columns=['X','Y','Z']
#     df_3D['R'] = colors_r
#     df_3D['G'] = colors_g
#     df_3D['B'] = colors_b
#     df_3D['A'] = colors_a

#     df_3D[layout_namespace] = layout_namespace
#     df_3D['ID'] = list(posG.keys())

#     cols = df_3D.columns.tolist()
#     cols = cols[-1:] + cols[:-1]
#     df_3D_final = df_3D[cols]
    
#     return df_3D_final.to_csv(r''+path+layout_namespace+'_layout.csv',index=False, header=False)



# def import_vrnetzer_csv2D(G,file):

#     df = pd.read_csv(file, header=None)

#     df.columns = ['id','x','y','z','r','g','b','a','namespace']

#     df_vrnetzer = df.set_index('id')
#     df_vrnetzer.index.name = None

#     ids = [str(i) for i in list(df['id'])]
#     x = list(df['x'])
#     y = list(df['y'])
#     z = list(df['z'])
#     posG = dict(zip(ids,zip(x,y))) #,z)))

#     r_list = list(df['r'])
#     g_list = list(df['g'])
#     b_list = list(df['b'])
#     a_list = list(df['a'])

#     colors = list(zip(r_list,g_list,b_list,a_list))
    
#     return posG,colors



# def import_vrnetzer_csv3D(G,file):

#     df = pd.read_csv(file, header=None)

#     df.columns = ['id','x','y','z','r','g','b','a','namespace']

#     df_vrnetzer = df.set_index('id')
#     df_vrnetzer.index.name = None

#     ids = [str(i) for i in list(df['id'])]
#     x = list(df['x'])
#     y = list(df['y'])
#     z = list(df['z'])
#     posG = dict(zip(ids,zip(x,y,z)))

#     r_list = list(df['r'])
#     g_list = list(df['g'])
#     b_list = list(df['b'])
#     a_list = list(df['a'])

#     colors = list(zip(r_list,g_list,b_list,a_list))
    
#     return posG,colors