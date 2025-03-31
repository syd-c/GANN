# Graph attention neural networks for interpretable and generalizable prediction of Janus III-VI van der Waals heterostructures
## Abstract
Graph neural networks (GNNs) have been widely used in materials science due to their ability to process complex graph-structured data by capturing the relationships between atoms, molecules, or crystal structures within materials. However, due to the lack of interpretability, GNN is acted as a “black box” model in most cases. In this work, by introducing the attention mechanism into GNN, a graph attention neural network (GANN) model for two-dimensional (2D) materials is proposed, which not only reach accurate prediction performances but also show interpretability via attention mechanism analysis. Taking Janus III-VI van der Waals heterostructures as a representative case, the MAE for predicting formation energy, lattice constants, PBE bandgap, and HSE bandgap are 0.012 eV, 0.004 Å, 0.119 eV, and 0.168 eV, respectively. Remarkably, the GANN model shows outstanding generalizable ability that can achieve accurate prediction using guessed input structures without fully structural relaxation to reach the ground state for Janus III-VI vdW heterostructures. Furthermore, the attention mechanisms of integration enable qualitative analysis of the contributions of the nearest neighboring atoms, endowing the GNN model with enhanced interpretability. Our findings offer novel perspectives for AI-driven two-dimensional material design by establishing an optimal balance between predictive accuracy and model interpretability in graph neural network approaches.
- Project path: GANN<br>
`cd GANN`<br>
- Dependence packages：<br>
`pip install numpy`<br>
`pip install tqdm`<br>
`pip install pandas`<br>
`pip install scikit-learn`<br>
`pip install tensorboard`<br>
`pip install pymatgen`<br>
- Simply run the following code:<br>
`python main.py --train_HE_3321X4_hou_HSE --target 2 --train-ratio 0.8  --val-ratio 0.1 --test-ratio 0.1`
