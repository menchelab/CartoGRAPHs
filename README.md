# CartoGRAPHs
### A Framework for Interpretable Network Visualizations

![3D Network Portrait](img/3DNetworkPortrait.png)


Networks offer an intuitive visual representation of complex systems. Important network
characteristics can often be recognized by eye and, in turn, patterns that stand out
visually often have a meaningful interpretation. However, conventional network layouts
are difficult to interpret, as they offer no direct connection between node position and
network structure. Here, we propose an approach for directly encoding arbitrary
structural or functional network characteristics into node positions. We introduce a
series of two and three-dimensional layouts, benchmark their efficiency for model
networks, and demonstrate their power for elucidating s tructure to function 
relationships in large-scale biological networks.


### REPOSITORY STRUCTURE
The structue of the project Github repo is as described below:
```
menchelab/cartoGRAPHs
- cartoGRAPHs_main.py ---> main functions used in the jupyter notebook
- cartoGRAPHs_main.ipynb ---> jupyter notebook to visualize networks (examples are human interactome and yeast interactome, as introduced in the manuscript)
├── input ---> all input data required to reproduce figures in jupyter notebook and python files
├── benchmark ---> benchmark evaluations 
├── output_plots ---> folder for saving produced plots during notebook sessions 
└── .gitignore
```

### **HOW TO USE THE FRAMEWORK**

At the moment the code can be run using Jupyter Notebook/Lab and will soon be available as a python package. 
The main script for producing Layouts for the Protein-Protein Interaction Network (*homo sapiens* or *saccharomyces cerevisiae*) 
is entitled "cartographs_main.ipynb" with the python functions included in "cartographs_main.py". Input files essential to run the scripts can be downloaded
[here](https://drive.google.com/file/d/1_FR-It9-h9ZZ1Pn-ErwGqxxIlMCHG_54/view?usp=sharing) and shall be unpacked in the location of the jupyter notebook. 
A web-based application will be available to the public soon. 


### **NETWORK LAYOUTS**
Four different Layouts can be produced. 

+ local layout (based on node pairwise adjacencies)
+ global layout (based on network propagation)
+ importance layout (based on network centrality metrics, such as degree, closeness, betweenness and eigenvector centrality)
+ functional layout (e.g. based on a NxM matrix including N nodes in the network and M features)

### ''NETWORK CATEGORIES**

Four different Layout Categories are implemented. 
+ 2D Network portrait
+ 3D Network portrait
+ 3D Topographic Network Map
+ 3D Geodesic Network Map




### **MODEL NETWORKS FOR BENCHMARKING**

To benchmark the framework, model networks with well-known architecture, such as Cayley Tree, Cubic Grid and Torus Lattice were used.
The code to run and reproduce layouts with aforementioned model networks can be viewed in the folder "benchmark". The respective scripts are partitioned based on model network and precalculated files, for network distance comparison can be downloaded [here](https://drive.google.com/file/d/1_Fhc6pbW8TfCB9jYUQGG-8I5qLs1niUZ/view?usp=sharing). Please unzip and place the folder in the directory of the benchmarking scripts (i.e. in the "benchmark" folder). 

*Please note : This project is work in progress and will be updated/improved frequently.*

