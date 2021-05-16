# CartoGRAPHs
### A Framework for Interpretable Network Visualizations

Networks offer an intuitive visual representation of complex systems. Important network
characteristics can often be recognized by eye and, in turn, patterns that stand out
visually often have a meaningful interpretation. However, conventional network layouts
are difficult to interpret, as they offer no direct connection between node position and
network structure. Here, we propose an approach for directly encoding arbitrary
structural or functional network characteristics into node positions. We introduce a
series of two and three-dimensional layouts, benchmark their efficiency for model
networks, and demonstrate their power for elucidating s tructure to function relationships
in large-scale biological networks.


### **How to use**

At the moment the code can be run using Jupyter Notebook/Lab and will soon be available as a python package. 
The main script for producing Layouts for the Protein-Protein Interaction Network (homo sapiens or saccharomyces cerevisiae) 
is entitled "cartographs_main.ipynb" with the python functions included in "cartographs_main.py". 


### **Benchmarks** 

Model networks with well-known architecture, such as Cayley Tree, Cubic Grid and Torus Lattice were used to benchmark the framework. The scripts regarding all benchmarking figures can be found in the folder "benchmarks". The network distances for all model network sizes, which are required to run the scripts were precalculated and can be downloaded from here and shall be located in the folder "benchmark" after unzipping.  

