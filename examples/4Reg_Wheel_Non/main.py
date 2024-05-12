#imports
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import csv
import matplotlib.pyplot as plt 
from karateclub import DeepWalk

n_nodes = 9

graphs = [
[(0, 1), (0, 4), (0, 2), (0, 5), (1, 5), (1, 8), (1, 7), (2, 4), (2, 7), (2, 3), (4, 3), (4, 5), (3, 8), (3, 6), (8, 6), (8, 7), (7, 6), (5, 6)],
[(0, 2), (0, 4), (0, 3), (0, 5), (2, 5), (2, 8), (2, 7), (3, 4), (3, 7), (3, 1), (4, 1), (4, 5), (1, 8), (1, 6), (8, 6), (8, 7), (7, 6), (5, 6)],
[(0, 1), (0, 3), (0, 4), (0, 5), (1, 5), (1, 2), (1, 7), (4, 3), (4, 7), (4, 8), (3, 8), (3, 5), (8, 2), (8, 6), (2, 6), (2, 7), (7, 6), (5, 6)],
[(2, 1), (2, 5), (2, 4), (2, 7), (1, 7), (1, 8), (1, 0), (4, 5), (4, 0), (4, 3), (5, 3), (5, 7), (3, 8), (3, 6), (8, 6), (8, 0), (0, 6), (7, 6)],
[(5, 3), (5, 4), (5, 2), (5, 1), (3, 1), (3, 8), (3, 7), (2, 4), (2, 7), (2, 0), (4, 0), (4, 1), (0, 8), (0, 6), (8, 6), (8, 7), (7, 6), (1, 6)],
[(0, 1), (0, 4), (0, 8), (0, 5), (1, 5), (1, 7), (1, 6), (8, 4), (8, 6), (8, 3), (4, 3), (4, 5), (3, 7), (3, 2), (7, 2), (7, 6), (6, 2), (5, 2)],
[(5, 1), (5, 4), (5, 0), (5, 2), (1, 2), (1, 8), (1, 7), (0, 4), (0, 7), (0, 3), (4, 3), (4, 2), (3, 8), (3, 6), (8, 6), (8, 7), (7, 6), (2, 6)],
[(0, 5), (0, 8), (0, 2), (0, 4), (5, 4), (5, 1), (5, 7), (2, 8), (2, 7), (2, 3), (8, 3), (8, 4), (3, 1), (3, 6), (1, 6), (1, 7), (7, 6), (4, 6)],
[(0, 1), (0, 7), (0, 4), (0, 5), (1, 5), (1, 8), (1, 3), (4, 7), (4, 3), (4, 2), (7, 2), (7, 5), (2, 8), (2, 6), (8, 6), (8, 3), (3, 6), (5, 6)],
[(0, 4), (0, 8), (0, 2), (0, 5), (4, 5), (4, 1), (4, 7), (2, 8), (2, 7), (2, 3), (8, 3), (8, 5), (3, 1), (3, 6), (1, 6), (1, 7), (7, 6), (5, 6)],
[(0, 1), (0, 4), (0, 5), (0, 3), (1, 3), (1, 2), (1, 7), (5, 4), (5, 7), (5, 8), (4, 8), (4, 3), (8, 2), (8, 6), (2, 6), (2, 7), (7, 6), (3, 6)],
[(0, 1), (0, 6), (0, 2), (0, 5), (1, 5), (1, 8), (1, 7), (2, 6), (2, 7), (2, 4), (6, 4), (6, 5), (4, 8), (4, 3), (8, 3), (8, 7), (7, 3), (5, 3)],
[(4, 1), (4, 2), (4, 6), (4, 5), (1, 5), (1, 0), (1, 7), (6, 2), (6, 7), (6, 3), (2, 3), (2, 5), (3, 0), (3, 8), (0, 8), (0, 7), (7, 8), (5, 8)],
[(0, 1), (0, 3), (0, 2), (0, 7), (1, 7), (1, 8), (1, 4), (2, 3), (2, 4), (2, 5), (3, 5), (3, 7), (5, 8), (5, 6), (8, 6), (8, 4), (4, 6), (7, 6)],
[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (1, 2), (1, 8), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
[(0, 2), (0, 3), (0, 1), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (2, 3), (2, 8), (3, 1), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
[(0, 1), (0, 4), (0, 8), (0, 3), (0, 5), (0, 6), (0, 7), (0, 2), (1, 4), (1, 2), (4, 8), (8, 3), (3, 5), (5, 6), (6, 7), (7, 2)],
[(2, 1), (2, 4), (2, 3), (2, 5), (2, 7), (2, 6), (2, 0), (2, 8), (1, 4), (1, 8), (4, 3), (3, 5), (5, 7), (7, 6), (6, 0), (0, 8)],
[(5, 3), (5, 2), (5, 0), (5, 4), (5, 1), (5, 6), (5, 7), (5, 8), (3, 2), (3, 8), (2, 0), (0, 4), (4, 1), (1, 6), (6, 7), (7, 8)],
[(0, 1), (0, 8), (0, 3), (0, 4), (0, 5), (0, 2), (0, 6), (0, 7), (1, 8), (1, 7), (8, 3), (3, 4), (4, 5), (5, 2), (2, 6), (6, 7)],
[(5, 1), (5, 0), (5, 3), (5, 4), (5, 2), (5, 6), (5, 7), (5, 8), (1, 0), (1, 8), (0, 3), (3, 4), (4, 2), (2, 6), (6, 7), (7, 8)],
[(0, 5), (0, 2), (0, 3), (0, 8), (0, 4), (0, 6), (0, 7), (0, 1), (5, 2), (5, 1), (2, 3), (3, 8), (8, 4), (4, 6), (6, 7), (7, 1)]
]
graph_type = '4-Reg_Wheel'

# translate graphs into vectors 
def prepare_graphs(first, second):
    g = nx.from_edgelist(first)
    #degree_centrality = nx.degree_centrality(g)
    #degree_centrality = nx.eigenvector_centrality(g)
    model = DeepWalk(dimensions=1)
    model.fit(g)
    embedding1 = model.get_embedding()
    feature_vec_1 = []

    for i in range(len(embedding1)):
        feature_vec_1.append(embedding1[i][0])

    h = nx.from_edgelist(second)
    #degree_centrality = nx.degree_centrality(h)
    #degree_centrality = nx.eigenvector_centrality(g)
    feature_vec_2 = []
    model1 = DeepWalk(dimensions=1)
    model1.fit(h)
    embedding2 = model1.get_embedding()
    for i in range(n_nodes):
        feature_vec_2.append(embedding2[i][0])
    return feature_vec_1, feature_vec_2

# variational layer
def layer(qubits, weights):
    for j in range(qubits):
        qml.RY(weights[0][j], wires=j)
        qml.RZ(weights[1][j], wires=j)
    for k in range(qubits):
        qml.CZ(wires=[k, (k+1) % qubits])

# inverse variational layer
def inverse_layer(qubits, weights):
    qml.adjoint(layer)(qubits, weights)
    # for i in range(qubits):
    #     qml.CZ(wires= [qubits-(i+1), (qubits-i) % qubits])
    # for j in range(qubits):
    #     qml.RZ(-weights[1][j], wires=j)
    #     qml.RY(-weights[0][j], wires=j)

wires = range(n_nodes)
dev = qml.device('default.qubit', n_nodes)

# list circuit
@qml.qnode(dev, diff_method='backprop')
def circuit(weights, feature_vec):
    val = feature_vec
    qml.AngleEmbedding(val, wires)
    #for i in range(int(np.sqrt(n_nodes))):
    for i in range(1):
        layer(n_nodes, weights[i*2:i*2+2])
    return qml.state()

@qml.qnode(dev)
def inverse_circuit(weights, feature_vec1, feature_vec2):
    val1 = feature_vec1
    val2 = feature_vec2
    qml.AngleEmbedding(val1, wires)
    #for i in range(int(np.sqrt(n_nodes))):
    for i in range(1):
        layer(n_nodes, weights[i*2:i*2+2])    
    inverse_weights = weights[::-1]
    #for i in range(int(np.sqrt(n_nodes))):
    for i in range(1):
        a = inverse_weights[i*2:i*2+2]
        inverse_layer(n_nodes, a[::-1])
    qml.adjoint(qml.AngleEmbedding(val2, wires))
    return qml.state()

# list architecture
#weights = np.repeat(np.pi, 2*int(np.sqrt(n_nodes))*n_nodes)
#weights = np.reshape(weights, (2*int(np.sqrt(n_nodes)), n_nodes))

#a = circuit(weights, feature_vec_1)[0]
#b = circuit(weights, feature_vec_2)[0]

## dot product equals 1 if equal vectors, dot product equals 0 if orthogonal vectors
#fidelity = np.dot(a,b)
#fidelity
#print(qml.draw(circuit, expansion_strategy="device")(weights, feature_vec_1))
#print(qml.draw(inverse_circuit, expansion_strategy="device")(weights, feature_vec_1, feature_vec_2))

#inverse_circuit(weights, feature_vec_1, feature_vec_2)[0]

def list_costs(weights):
    #fidelity = np.dot(circuit(weights, feature_vec_1)[0], circuit(weights, feature_vec_2)[0])
    #fidelity = np.dot(inverse_circuit(weights, feature_vec_1, feature_vec_2)[0], inverse_circuit(weights, feature_vec_1, feature_vec_2)[0])
    fidelity = inverse_circuit(weights, feature_vec_1, feature_vec_2)[-1]
    return np.sqrt(np.real(fidelity)**2+np.imag(fidelity)**2) **2


for i in range(1):
    for j in range(len(graphs)):
        for k in range(len(graphs)):
            np.random.seed()
            weights_init = np.random.randn(2*2, n_nodes, requires_grad=True)
            angle = [weights_init]
            feature_vec_1, feature_vec_2 = prepare_graphs(graphs[j], graphs[k])
            list_cost = [list_costs(weights_init)] 
            opt = qml.AdamOptimizer()
            max_iterations = 100
            conv_tol = 1e-06
            print(qml.draw(inverse_circuit, expansion_strategy="device")(weights_init, feature_vec_1, feature_vec_2))
    # ### circuit results in tensor array size 2^9, but only need value at position 0 
            for n in range(max_iterations):
                weights_init, prev_cost = opt.step_and_cost(list_costs, weights_init)
                list_cost.append(prev_cost)
                angle.append(weights_init)

                conv = np.abs(list_cost[-1] - prev_cost)
                if n % 10 == 0:
                    print(f"Step = {n},  Cost function = {list_cost[-1]:.8f} ")
                # if conv <= conv_tol:
                #     break

            with open ('../Data-Non/QGK_' + graph_type + '_List_Angles_Graphs_' + str(j) + '_' + str(k) + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(angle)  
                file.close() 

            with open ('../Data-Non/QGK_' + graph_type + '_List_Cost_Graphs_' + str(j) + '_' + str(k) + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list_cost)  
                file.close()

def sandwich_layer(qubits, weights):
    for j in range(int(qubits)):
        qml.RY(weights[0][j], wires=j)
        qml.RZ(weights[1][j], wires=j)
    for k in range(qubits):
        qml.CZ(wires=[k, (k+1) % qubits])

def inverse_sandwich_layer(qubits, weights):
    qml.adjoint(sandwich_layer)(qubits, weights)
    # for i in range(qubits):
    #     qml.CZ(wires= [qubits-(i+1), (qubits-i) % qubits])
    # for j in range(qubits):
    #     qml.RZ(weights[1][j], wires=j)
    #     qml.RY(weights[0][j], wires=j)

## sandwich architecture circuit
wires = range(int(np.sqrt(n_nodes)))
dev = qml.device('default.qubit', wires)

@qml.qnode(dev)
def sandwich_circuit(weights, val):
    qubits = int(np.sqrt(n_nodes))
    for i in range(qubits):
        qml.AngleEmbedding(val[i*qubits:(i*qubits)+qubits], wires)
        sandwich_layer(qubits, weights[i*2:i*2+2])

    return qml.state()

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
    #return qml.expval(qml.PauliZ(wires=[0])), qml.expval(qml.PauliZ(wires=[1]))

#print(qml.draw(inverse_sandwich_circuit, expansion_strategy='device')(weights, feature_vec_1, feature_vec_2))

# sandwich architecture
#weights = np.repeat(np.pi, 18)
#weights = np.reshape(weights, (6, 3))

#a = sandwich_circuit(weights, feature_vec_1)[0]
#b = sandwich_circuit(weights, feature_vec_2)[0]

## dot product equals 1 if equal vectors, dot product equals 0 if orthogonal vectors
#fidelity = np.dot(a,b)
#print(qml.draw(sandwich_circuit, expansion_strategy="device")(weights, feature_vec_1))
#print(qml.draw(inverse_sandwich_circuit, expansion_strategy='device')(weights, feature_vec_1, feature_vec_2))

def sandwich_costs(weights):
    fidelity = inverse_sandwich_circuit(weights, feature_vec_1, feature_vec_2)[-1]
    #fidelity = np.dot(inverse_sandwich_circuit(weights, feature_vec_1, feature_vec_2)[0], inverse_sandwich_circuit(weights, feature_vec_1, feature_vec_2)[0])
    #fidelity = np.dot(sandwich_circuit(weights, feature_vec_1)[0], sandwich_circuit(weights, feature_vec_2)[0])
    return np.sqrt(np.real(fidelity)**2+np.imag(fidelity)**2) **2

for i in range(1):
    for j in range(len(graphs)):
        for k in range(len(graphs)):
            print('sandwich', wires)
            np.random.seed()
            weights_init = np.random.randn(2*int(np.sqrt(n_nodes))*2, int(np.sqrt(n_nodes)), requires_grad=True)
            angle = [weights_init]
            feature_vec_1, feature_vec_2 = prepare_graphs(graphs[j], graphs[k])
            sandwich_cost = [sandwich_costs(weights_init)]
            opt = qml.AdamOptimizer()
            max_iterations = 100
            conv_tol = 1e-06

            print(qml.draw(inverse_sandwich_circuit, expansion_strategy='device')(weights_init, feature_vec_1, feature_vec_2))
    ### circuit results in tensor array size 2^9, but only need value at position 0 
            for n in range(max_iterations):
                weights_init, prev_cost = opt.step_and_cost(sandwich_costs, weights_init)
                sandwich_cost.append(prev_cost)
                angle.append(weights_init)

                conv = np.abs(sandwich_cost[-1] - prev_cost)
                if n % 10 == 0:
                    print(f"Step = {n},  Cost function = {sandwich_cost[-1]:.8f} ")
                #if conv <= conv_tol:
                #    break
            
            with open ('../Data-Non/QGK_' + graph_type + '_Sandwich_Angles_Graphs_' + str(j) + '_' + str(k) + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(angle)  
                file.close() 

            with open ('../Data-Non/QGK_' + graph_type + '_Sandwich_Cost_Graphs_' + str(j) + '_' + str(k) + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(sandwich_cost)  
                file.close() 
