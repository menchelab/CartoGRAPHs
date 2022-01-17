  
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains  E X P O R T   C O O R D I N A T E S   F U N C T I O N S 
# compatible/for uplad to VRNetzer Platform and Webapplication "cartoGRAPHs"
#
########################################################################################

from cartoGRAPHs.func_visual_properties import *

import pandas as pd

########################################################################################

def export_to_csv2D(path, layout_namespace, posG, colors = None):
    '''
    Generate csv for upload to VRnetzer plaform for 2D layouts. 
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    
    if colors is None: 
        colors_hex2rgb = list((255,0,0) for i in list(posG.keys()))
    
    else: 
        colors_hex2rgb = []
        for j in colors: 
            k = hex_to_rgb(j)
            colors_hex2rgb.append(k)

    colors_r = []
    colors_g = []
    colors_b = []
    colors_a = []
    for i in colors_hex2rgb:
        colors_r.append(int(i[0]))#*255)) # color values should be integers within 0-255
        colors_g.append(int(i[1]))#*255))
        colors_b.append(int(i[2]))#*255))
        colors_a.append(100) # 0-100 shows normal colors in VR, 128-200 is glowing mode

    df_2D = pd.DataFrame(posG).T
    df_2D.columns=['X','Y']
    df_2D['Z'] = 0
    df_2D['R'] = colors_r
    df_2D['G'] = colors_g
    df_2D['B'] = colors_b
    df_2D['A'] = colors_a

    df_2D[layout_namespace] = layout_namespace
    df_2D['ID'] = list(posG.keys())

    cols = df_2D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_2D_final = df_2D[cols]
    
    return df_2D_final.to_csv(r''+path+layout_namespace+'_layout.csv',index=False, header=False)
    

def export_to_csv3D(path, layout_namespace, posG, colors = None):
    '''
    Generate csv for upload to VRnetzer plaform for 3D layouts. 
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    
    if colors is None: 
        colors_hex2rgb = list((255,0,0) for i in list(posG.keys()))
    
    else: 
        colors_hex2rgb = []
        for j in colors: 
            k = hex_to_rgb(j)
            colors_hex2rgb.append(k)
            
    colors_r = []
    colors_g = []
    colors_b = []
    colors_a = []
    for i in colors_hex2rgb:
        colors_r.append(int(i[0]))#*255)) # color values should be integers within 0-255
        colors_g.append(int(i[1]))#*255))
        colors_b.append(int(i[2]))#*255))
        colors_a.append(100) # 0-100 shows normal colors in VR, 128-200 is glowing mode
        
    df_3D = pd.DataFrame(posG).T
    df_3D.columns=['X','Y','Z']
    df_3D['R'] = colors_r
    df_3D['G'] = colors_g
    df_3D['B'] = colors_b
    df_3D['A'] = colors_a

    df_3D[layout_namespace] = layout_namespace
    df_3D['ID'] = list(posG.keys())

    cols = df_3D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_3D_final = df_3D[cols]
    
    return df_3D_final.to_csv(r''+path+layout_namespace+'_layout.csv',index=False, header=False)



def import_vrnetzer_csv2D(G,file):

    df = pd.read_csv(file, header=None)

    df.columns = ['id','x','y','z','r','g','b','a','namespace']

    df_vrnetzer = df.set_index('id')
    df_vrnetzer.index.name = None

    ids = [str(i) for i in list(df['id'])]
    x = list(df['x'])
    y = list(df['y'])
    z = list(df['z'])
    posG = dict(zip(ids,zip(x,y))) #,z)))

    r_list = list(df['r'])
    g_list = list(df['g'])
    b_list = list(df['b'])
    a_list = list(df['a'])

    colors = list(zip(r_list,g_list,b_list,a_list))
    
    return posG,colors



def import_vrnetzer_csv3D(G,file):

    df = pd.read_csv(file, header=None)

    df.columns = ['id','x','y','z','r','g','b','a','namespace']

    df_vrnetzer = df.set_index('id')
    df_vrnetzer.index.name = None

    ids = [str(i) for i in list(df['id'])]
    x = list(df['x'])
    y = list(df['y'])
    z = list(df['z'])
    posG = dict(zip(ids,zip(x,y,z)))

    r_list = list(df['r'])
    g_list = list(df['g'])
    b_list = list(df['b'])
    a_list = list(df['a'])

    colors = list(zip(r_list,g_list,b_list,a_list))
    
    return posG,colors