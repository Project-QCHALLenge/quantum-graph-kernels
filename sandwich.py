#imports
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import csv
import matplotlib.pyplot as plt 
from karateclub import DeepWalk
import os

#number of nodes in the graph
n_nodes = 9

graphs = [
[(0, 1), (0, 4), (0, 2), (0, 5), (1, 5), (1, 8), (1, 7), (2, 4), (2, 7), (2, 3), (4, 3), (4, 5), (3, 8), (3, 6), (8, 6), (8, 7), (7, 6), (5, 6)],
[(0, 2), (0, 4), (0, 3), (0, 5), (2, 5), (2, 8), (2, 7), (3, 4), (3, 7), (3, 1), (4, 1), (4, 5), (1, 8), (1, 6), (8, 6), (8, 7), (7, 6), (5, 6)]
]
graph_type = '4-Regular'

#  translate first und second graph into vectors feature_vec_1 and feature_vec_2 using the DeepWalk algorithm as encoding method
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
def sandwich_layer(qubits, weights):
    for j in range(int(qubits)):
        qml.RY(weights[0][j], wires=j)
        qml.RZ(weights[1][j], wires=j)
    for k in range(qubits):
        qml.CZ(wires=[k, (k+1) % qubits])

# inverse variational layer
def inverse_sandwich_layer(qubits, weights):
    qml.adjoint(sandwich_layer)(qubits, weights)

wires = range(int(np.sqrt(n_nodes)))
dev = qml.device('default.qubit', wires)

# sandwich circuit
@qml.qnode(dev)
def sandwich_circuit(weights, val):
    qubits = int(np.sqrt(n_nodes))
    for i in range(qubits):
        qml.AngleEmbedding(val[i*qubits:(i*qubits)+qubits], wires)
        sandwich_layer(qubits, weights[i*2:i*2+2])
    return qml.state()

# sandwich circuit using the inversion test
@qml.qnode(dev)
def inverse_sandwich_circuit(weights, feature_vec1, feature_vec2):
    val1 = feature_vec1
    val2 = feature_vec2
    qubits = int(np.sqrt(n_nodes))
    for i in range(qubits):
        qml.AngleEmbedding(val1[i*qubits:(i*qubits)+qubits], wires)
        sandwich_layer(qubits, weights[i*2:i*2+2])
    inverse_weights = weights[::-1]
    for i in range(qubits):
        a = inverse_weights[i*2:i*2+2]
        inverse_sandwich_layer(qubits, a[::-1])
        qml.adjoint(qml.AngleEmbedding(val2[(qubits-i-1)*qubits:((qubits-i-1)*qubits)+qubits], wires))
    return qml.state()
    
# cost function    
def sandwich_costs(weights):
    fidelity = inverse_sandwich_circuit(weights, feature_vec_1, feature_vec_2)[0]
    return 1 - np.sqrt(np.real(fidelity)**2+np.imag(fidelity)**2) **2

# Specify the directory path
directory = "Data"

# optimization
for i in range(1): # number of runs
    np.random.seed()
    weights_init = np.random.randn(2*int(np.sqrt(n_nodes))*2, int(np.sqrt(n_nodes)), requires_grad=True)
    angle = [weights_init]
    feature_vec_1, feature_vec_2 = prepare_graphs(graphs[0], graphs[1])
    sandwich_cost = [sandwich_costs(weights_init)]
    opt = qml.AdamOptimizer()
    max_iterations = 100
    conv_tol = 1e-06

    print(qml.draw(inverse_sandwich_circuit, expansion_strategy='device')(weights_init, feature_vec_1, feature_vec_2))
    
    for n in range(max_iterations):
        weights_init, prev_cost = opt.step_and_cost(sandwich_costs, weights_init)
        sandwich_cost.append(prev_cost)
        angle.append(weights_init)

        conv = np.abs(sandwich_cost[-1] - prev_cost)
        if n % 10 == 0:
            print(f"Step = {n},  Cost function = {sandwich_cost[-1]:.8f} ")


    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
      
    with open ('Data/QGK_' + graph_type + '_Sandwich_Angles' + '.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(angle)  
        file.close() 

    with open ('Data/QGK_' + graph_type + '_Sandwich_Cost' + '.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(sandwich_cost)  
        file.close() 
