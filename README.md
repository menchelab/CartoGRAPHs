# CartoGRAPHs
## A Framework for Interpretable Network Visualizations


Networks offer an intuitive visual representation of complex systems. Important network
characteristics can often be recognized by eye and, in turn, patterns that stand out
visually often have a meaningful interpretation. However, conventional network layouts
are difficult to interpret, as they offer no direct connection between node position and
network structure. Here, we propose an approach for directly encoding arbitrary
structural or functional network characteristics into node positions. We introduce a
series of two and three-dimensional layouts, benchmark their efficiency for model
networks, and demonstrate their power for elucidating s tructure to function 
relationships in large-scale biological networks.

___
### **How to use the Framework**
___

*Please note : This project is work in progress and will be updated/improved frequently.*

At the moment the code can be run using Jupyter Notebook/Lab and will soon be available as a python package. 
The main script for producing Layouts for the Protein-Protein Interaction Network (*homo sapiens* or *saccharomyces cerevisiae*) 
is entitled "cartographs_main.ipynb" with the python functions included in "cartographs_main.py". 

A web-based application will be available to the public soon. 


### **Which Layouts can be produced**

Four different Layout Categories are implemented. 
+ 2D Network portrait
+ 3D Network portrait
+ 3D Topographic Network Map
+ 3D Geodesic Network Map

### **Benchmarking with Model Networks**

To benchmark the framework, model networks with well-known architecture, such as Cayley Tree, Cubic Grid and Torus Lattice were used. The scripts regarding all benchmarking figures can be found in the folder "benchmarks". The network distances for all model network sizes, which are required to run the scripts were precalculated and can be downloaded from here and shall be located in the folder "benchmark" after unzipping.  
