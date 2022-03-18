# CARTOGRAPHS 
Visual Network Exploration in two and three dimensions


![cartographs](img/cartographs_img02.png)

---

Networks offer an intuitive visual representation of complex systems. Important network
characteristics can often be recognized by eye and, in turn, patterns that stand out
visually often have a meaningful interpretation. However, conventional network layouts
are difficult to interpret, as they offer no direct connection between node position and
network structure. Here, we propose an approach for directly encoding arbitrary
structural or functional network characteristics into node positions. We introduce a
series of two and three-dimensional layouts, benchmark their efficiency for model
networks, and demonstrate their power for elucidating structure to function 
relationships in large-scale biological networks.


---

### ABOUT CARTOGRAPHS

CartoGRAPHs is a python package to generate two- and three-dimensional layouts of networks. 
Here you will find Jupyter Notebooks to use our method of visualizing different network characteristics based on 
feature modulation and dimensionality reduction.

To get a first glance on the framework, we provide a Quickstarter Notebook with an exemplary graph. Additionally 
one can dive deeper into real world networks focusing on the Protein Protein Interaction Network.

---

### INSTALLATION

Install the package e.g. in a virtual environment: 

+ create a virtual environment 
```
python3 -m venv name_of_env
```

+ activate it 
```
source name_of_env/bin/activate 
```

+ install requirements packages
```
python3 -m pip install -r requirements.txt
```

+ install cartoGRAPHs 
```
python3 -m pip install cartoGRAPHs
```

+ to use environment with jupyter notebooks
```
ipython kernel install --user --name=name_of_env
```

More information here: https://pypi.org/project/cartoGRAPHs/1.0.1/

---

### NETWORK LAYOUTS

The Network Layouts are themed based on different characteristics of a Network. Those can be of structural or functional nature. Additionally we came up with a method to modulate between both, structural and functional features (please find a "hands-on" example in the Notebook "cartoGRAPHs_FeatureModulation.ipynb"). 

An Overview on the layouts included within the framework: 

+ **local layout** > based on node pairwise adjacencies
+ **global layout** > based on network propagation
+ **importance layout** > based on network centrality metrics, such as degree, closeness, betweenness and eigenvector centrality
+ **functional layout** > e.g. based on a *NxM* matrix including *N* nodes in the network and *M* features
+ **combined layouts** > based on modulation between structural and functional features

---

### NETWORK CATEGORIES

To experiment with a diversity of two- and three-dimensional visualizations, we 
came up with four different Layout Categories, named after their natural appearance.

+ **2D Network portrait**
+ **3D Network portrait**
+ **3D Topographic Network Map**
+ **3D Geodesic Network Map**

---

### HOW TO CREATE NETWORK VISUALIZATIONS

**Quickstarter** | *cartoGRAPHs_AQuickStarter.ipynb*
The Quickstarter Notebook contains basic functions to get familiar with the framework and 
test different layouts quickly using small network models. 

**More Detailed Example** | *cartoGRAPHs_ExemplaryNotebook.ipynb*

**Focus: Feature Modulation** | *cartoGRAPHs_FeatureModulation.ipynb*

**A Biological Network: Human PPI** | *cartoGRAPHs_ManuscriptFigure*.ipynb* 

---

FOLDER STRUCTURE 

```
├── input: please download all input files from our zenodo repository and deposit the "input" folder into your directory of notebooks
├── benchmark: benchmark evaluations with network models; input files can as well be downloaded through zenodo and should be placed into the "benchmark" folder 
├── img: diagrams and images
└── pyfiles: python files to create layouts and visualizations 
```

PYTHON FILES (folder *pyfiles*)

- cartoGRAPHs.py > contains the actual layout functions
- func_load_data.py > loading precalculated data, to be found in /input/
- func_calculations.py > contains functions for calculations 
- func_embed_plot.py > contains spatial embedding and visualization functions 
- func_visual_properties.py > contains additional node- and edge visual property settings/functions
- func_exportVR.py > contains export functions for 2D/3D layouts e.g. for a VR platform

---

**A diagrammatic overview of the functions included in this framework:** 

![cartographs](img/Codestructure_diagram.png)

---

### **INPUT DATA**

To use the Jupyter Notebooks provided for the Human PPI, please download the input files from [Zenodo](https://doi.org/10.5281/zenodo.5883000).
The files are located in the "input" folder and should be unzipped at the location of the Notebooks. 
The folder includes a PPI edgelist, Disease Ontology files, Gene Ontology files and Gene lists, 
such as Essentiality, Rare Diseases, Early Developmental Genes. 
Additionally one can find preprocessed Feature Matrices for diverse layouts, due to exceeding calculation time for large networks. 

---

### **MODEL NETWORKS FOR BENCHMARKING**

To benchmark the framework, model networks with well-known architecture, such as Cayley Tree, Cubic Grid and Torus Lattice were used.
The code to run and reproduce layouts with aforementioned model networks can be viewed in the folder "benchmark". The respective scripts are partitioned based on model network and precalculated files, for network distance comparison can be obtained from the Zenodo repository. Please unzip and place the "benchmark" folder in the directory of the benchmarking scripts for testing.

---

### **EXPLORING NETWORKS THROUGH A WEB APPLICATION**
To explore the framework without any coding required, please check out our web application here: www.cartographs.xyz
It is frequently updated, check out our github repository for more [here](https://github.com/menchelab/cartoGRAPHs_app)

---

### SYSTEM REQUIREMENTS
All Visualizations with networks up to 20,000 nodes and ~300,000 links in the main jupyter notebook were carried out on a machine with a 2 GHz Quad-Core Intel Core i5 and 16GB or Memory. 
Heavier computation, such as performed during layout benchmarking, was calculated on a cluster and stored in the benchmark/netdist_precalc folder. 

*Please note : This project is work in progress and will be updated/improved frequently.*

