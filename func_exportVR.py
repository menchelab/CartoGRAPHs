  
########################################################################################
#
# This python file is part of the Project "cartoGRAPHs"
# and contains  E X P O R T   C O O R D I N A T E S   F U N C T I O N S 
# compatible/for uplad to VRNetzer Platform and Webapplication "cartoGRAPHs"
#
########################################################################################

import pandas as pd

########################################################################################

def export_to_csv2D(path, layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 2D layouts. 
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    
    colours_hex2rgb = []
    for j in colours: 
        k = hex_to_rgb(j)
        colours_hex2rgb.append(k)
        
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours_hex2rgb:
        colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]))#*255))
        colours_b.append(int(i[2]))#*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df_2D = pd.DataFrame(posG).T
    df_2D.columns=['X','Y']
    df_2D['Z'] = 0
    df_2D['R'] = colours_r
    df_2D['G'] = colours_g
    df_2D['B'] = colours_b
    df_2D['A'] = colours_a

    df_2D[layout_namespace] = layout_namespace
    df_2D['ID'] = list(posG.keys())

    cols = df_2D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_2D_final = df_2D[cols]
    
    return df_2D_final.to_csv(r''+path+layout_namespace+'_layout.csv',index=False, header=False)
    

def export_to_csv3D(path, layout_namespace, posG, colours):
    '''
    Generate csv for upload to VRnetzer plaform for 3D layouts. 
    Return dataframe with ID,X,Y,Z,R,G,B,A,layout_namespace.
    '''
    
    colours_hex2rgb = []
    for j in colours: 
        k = hex_to_rgb(j)
        colours_hex2rgb.append(k)
        
    colours_r = []
    colours_g = []
    colours_b = []
    colours_a = []
    for i in colours_hex2rgb:
        colours_r.append(int(i[0]))#*255)) # colour values should be integers within 0-255
        colours_g.append(int(i[1]))#*255))
        colours_b.append(int(i[2]))#*255))
        colours_a.append(100) # 0-100 shows normal colours in VR, 128-200 is glowing mode
        
    df_3D = pd.DataFrame(posG).T
    df_3D.columns=['X','Y','Z']
    df_3D['R'] = colours_r
    df_3D['G'] = colours_g
    df_3D['B'] = colours_b
    df_3D['A'] = colours_a

    df_3D[layout_namespace] = layout_namespace
    df_3D['ID'] = list(posG.keys())

    cols = df_3D.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_3D_final = df_3D[cols]
    
    return df_3D_final.to_csv(r''+path+layout_namespace+'_layout.csv',index=False, header=False)


