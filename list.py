#imports
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import csv
import matplotlib.pyplot as plt 
from karateclub import DeepWalk
import os

# number of nodes in the graphs
n_nodes = 9

graphs = [
[(0, 1), (0, 4), (0, 2), (0, 5), (1, 5), (1, 8), (1, 7), (2, 4), (2, 7), (2, 3), (4, 3), (4, 5), (3, 8), (3, 6), (8, 6), (8, 7), (7, 6), (5, 6)],
[(0, 2), (0, 4), (0, 3), (0, 5), (2, 5), (2, 8), (2, 7), (3, 4), (3, 7), (3, 1), (4, 1), (4, 5), (1, 8), (1, 6), (8, 6), (8, 7), (7, 6), (5, 6)]
]
graph_type = '4-Regular'

# translate first und second graph into vectors feature_vec_1 and feature_vec_2 using the DeepWalk algorithm as encoding method
def prepare_graphs(first, second):
    g = nx.from_edgelist(first)
    model = DeepWalk(dimensions=1)
    model.fit(g)
    embedding1 = model.get_embedding()
    feature_vec_1 = []

    for i in range(len(embedding1)):
        feature_vec_1.append(embedding1[i][0])

    h = nx.from_edgelist(second)
    feature_vec_2 = []
    model1 = DeepWalk(dimensions=1)
    model1.fit(h)
    embedding2 = model1.get_embedding()
    for i in range(n_nodes):
        feature_vec_2.append(embedding2[i][0])
    return feature_vec_1, feature_vec_2

# one variational layer
def layer(qubits, weights):
    for j in range(qubits):
        qml.RY(weights[0][j], wires=j)
        qml.RZ(weights[1][j], wires=j)
    for k in range(qubits):
        qml.CZ(wires=[k, (k+1) % qubits])

# inverse variational layer
def inverse_layer(qubits, weights):
    qml.adjoint(layer)(qubits, weights)

wires = range(n_nodes)
dev = qml.device('default.qubit', n_nodes)

# list circuit
@qml.qnode(dev, diff_method='backprop')
def circuit(weights, feature_vec):
    val = feature_vec
    qml.AngleEmbedding(val, wires)
    for i in range(1):
        layer(n_nodes, weights[i*2:i*2+2])
    return qml.state()

# list circuit using the inversion test
@qml.qnode(dev)
def inverse_circuit(weights, feature_vec1, feature_vec2):
    val1 = feature_vec1
    val2 = feature_vec2
    qml.AngleEmbedding(val1, wires)
    for i in range(1):
        layer(n_nodes, weights[i*2:i*2+2])    
    inverse_weights = weights[::-1]
    for i in range(1):
        a = inverse_weights[i*2:i*2+2]
        inverse_layer(n_nodes, a[::-1])
    qml.adjoint(qml.AngleEmbedding(val2, wires))
    return qml.state()

# cost function
def list_costs(weights):
    fidelity = inverse_circuit(weights, feature_vec_1, feature_vec_2)[0]
    return 1 - np.sqrt(np.real(fidelity)**2+np.imag(fidelity)**2) **2

# Specify the directory path
directory = "Data"

# optimization
for i in range(1): # number of runs
    np.random.seed()
    weights_init = np.random.randn(2*2, n_nodes, requires_grad=True)
    angle = [weights_init]
    feature_vec_1, feature_vec_2 = prepare_graphs(graphs[0], graphs[1])        
    list_cost = [list_costs(weights_init)] 
    opt = qml.AdamOptimizer()
    max_iterations = 100
    conv_tol = 1e-06
    print(qml.draw(inverse_circuit, expansion_strategy="device")(weights_init, feature_vec_1, feature_vec_2))

    for n in range(max_iterations):
        weights_init, prev_cost = opt.step_and_cost(list_costs, weights_init)
        list_cost.append(prev_cost)
        angle.append(weights_init)
        conv = np.abs(list_cost[-1] - prev_cost)
        if n % 10 == 0:
            print(f"Step = {n},  Cost function = {list_cost[-1]:.8f} ")
        
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
    
    with open ('Data/QGK_' + graph_type + '_List_Angles' + '.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(angle)  
        file.close() 

    with open ('Data/QGK_' + graph_type + '_List_Cost' + '.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list_cost)  
        file.close()
