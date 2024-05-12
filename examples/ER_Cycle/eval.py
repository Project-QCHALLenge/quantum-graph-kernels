import csv
import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
from karateclub import DeepWalk

graphs = [
[(0, 2), (0, 3), (0, 4), (0, 7), (0, 8), (1, 2), (1, 3), (1, 8), (2, 3), (2, 4), (2, 5), (2, 7), (3, 8), (4, 5), (4, 6), (4, 8), (5, 7), (5, 8)],
[(0, 3), (0, 1), (0, 4), (0, 7), (0, 8), (2, 3), (2, 1), (2, 8), (3, 1), (3, 4), (3, 5), (3, 7), (1, 8), (4, 5), (4, 6), (4, 8), (5, 7), (5, 8)],
[(0, 4), (0, 8), (0, 3), (0, 7), (0, 2), (1, 4), (1, 8), (1, 2), (4, 8), (4, 3), (4, 5), (4, 7), (8, 2), (3, 5), (3, 6), (3, 2), (5, 7), (5, 2)],
[(2, 4), (2, 3), (2, 5), (2, 0), (2, 8), (1, 4), (1, 3), (1, 8), (4, 3), (4, 5), (4, 7), (4, 0), (3, 8), (5, 7), (5, 6), (5, 8), (7, 0), (7, 8)],
[(5, 2), (5, 0), (5, 4), (5, 7), (5, 8), (3, 2), (3, 0), (3, 8), (2, 0), (2, 4), (2, 1), (2, 7), (0, 8), (4, 1), (4, 6), (4, 8), (1, 7), (1, 8)],
[(0, 8), (0, 3), (0, 4), (0, 6), (0, 7), (1, 8), (1, 3), (1, 7), (8, 3), (8, 4), (8, 5), (8, 6), (3, 7), (4, 5), (4, 2), (4, 7), (5, 6), (5, 7)],
[(5, 0), (5, 3), (5, 4), (5, 7), (5, 8), (1, 0), (1, 3), (1, 8), (0, 3), (0, 4), (0, 2), (0, 7), (3, 8), (4, 2), (4, 6), (4, 8), (2, 7), (2, 8)],
[(0, 2), (0, 3), (0, 8), (0, 7), (0, 1), (5, 2), (5, 3), (5, 1), (2, 3), (2, 8), (2, 4), (2, 7), (3, 1), (8, 4), (8, 6), (8, 1), (4, 7), (4, 1)],
[(0, 4), (0, 2), (0, 7), (0, 3), (0, 8), (1, 4), (1, 2), (1, 8), (4, 2), (4, 7), (4, 5), (4, 3), (2, 8), (7, 5), (7, 6), (7, 8), (5, 3), (5, 8)],
[(0, 2), (0, 3), (0, 8), (0, 7), (0, 1), (4, 2), (4, 3), (4, 1), (2, 3), (2, 8), (2, 5), (2, 7), (3, 1), (8, 5), (8, 6), (8, 1), (5, 7), (5, 1)],
[(0, 5), (0, 8), (0, 4), (0, 7), (0, 2), (1, 5), (1, 8), (1, 2), (5, 8), (5, 4), (5, 3), (5, 7), (8, 2), (4, 3), (4, 6), (4, 2), (3, 7), (3, 2)],
[(0, 2), (0, 4), (0, 6), (0, 7), (0, 8), (1, 2), (1, 4), (1, 8), (2, 4), (2, 6), (2, 5), (2, 7), (4, 8), (6, 5), (6, 3), (6, 8), (5, 7), (5, 8)],
[(4, 6), (4, 3), (4, 2), (4, 7), (4, 0), (1, 6), (1, 3), (1, 0), (6, 3), (6, 2), (6, 5), (6, 7), (3, 0), (2, 5), (2, 8), (2, 0), (5, 7), (5, 0)],
[(0, 2), (0, 5), (0, 3), (0, 4), (0, 8), (1, 2), (1, 5), (1, 8), (2, 5), (2, 3), (2, 7), (2, 4), (5, 8), (3, 7), (3, 6), (3, 8), (7, 4), (7, 8)],
[(0, 1), (0, 8), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
[(0, 2), (0, 8), (2, 3), (3, 1), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
[(0, 1), (0, 2), (1, 4), (4, 8), (8, 3), (3, 5), (5, 6), (6, 7), (7, 2)],
[(2, 1), (2, 8), (1, 4), (4, 3), (3, 5), (5, 7), (7, 6), (6, 0), (0, 8)],
[(5, 3), (5, 8), (3, 2), (2, 0), (0, 4), (4, 1), (1, 6), (6, 7), (7, 8)],
[(0, 1), (0, 7), (1, 8), (8, 3), (3, 4), (4, 5), (5, 2), (2, 6), (6, 7)],
[(5, 1), (5, 8), (1, 0), (0, 3), (3, 4), (4, 2), (2, 6), (6, 7), (7, 8)],
[(0, 5), (0, 1), (5, 2), (2, 3), (3, 8), (8, 4), (4, 6), (6, 7), (7, 1)]
]
graph_type = 'ER_Cycle'

n_nodes = 9

def prepare_sandwich_weights(j, k):
    if k > 13 or j > 13:
        with open ('../Data-Non/QGK_' + graph_type + '_Sandwich_Angles_Graphs_' + str(j) + '_' + str(k) + '.csv') as file:
            reader = file.read()
    else:
        with open ('../Data/QGK_' + graph_type + '_Sandwich_Angles_Graphs_' + str(j) + '_' + str(k) + '.csv') as file:
            reader = file.read()
    weights = []

    reader = reader.split('"')
    x=True
    while(x):
        if '\n' in reader:
            reader.remove('\n')
        if '' in reader:
            reader.remove('')
        if ',' in reader:
            reader.remove(',')
        else:
            x = False

    a = reader[-1].split(' ')
    x=True
    while(x):
        if '[' in a:
            a.remove('[')
        if ']\n' in a:
            a.remove(']\n')
        if '[[' in a:
            a.remove('[[')
        if ',' in a:
            a.remove(',')
        if '[' in a:
            a.remove('[')
        if '' in a:
            a.remove('')
        else:
            x = False
    

    weights = [[float(a[0].replace('\n', '').replace('[[', '')), float(a[1].replace('\n', '')), float(a[2].replace(']\n', ''))],
            [float(a[3].replace('\n', '').replace('[', '')), float(a[4].replace('\n', '')), float(a[5].replace(']\n', ''))],
            [float(a[6].replace('\n', '').replace('[', '')), float(a[7].replace('\n', '')), float(a[8].replace(']\n', ''))],
            [float(a[9].replace('\n', '').replace('[', '')), float(a[10].replace('\n', '')), float(a[11].replace(']\n', ''))],
            [float(a[12].replace('\n', '').replace('[', '')), float(a[13].replace('\n', '')), float(a[14].replace(']\n', ''))],
            [float(a[15].replace('\n', '').replace('[', '')), float(a[16].replace('\n', '')), float(a[17].replace(']\n', ''))],
            [float(a[18].replace('\n', '').replace('[', '')), float(a[19].replace('\n', '')), float(a[20].replace(']\n', ''))],
            [float(a[21].replace('\n', '').replace('[', '')), float(a[22].replace('\n', '')), float(a[23].replace(']\n', ''))],
            [float(a[24].replace('\n', '').replace('[', '')), float(a[25].replace('\n', '')), float(a[26].replace(']\n', ''))],
            [float(a[27].replace('\n', '').replace('[', '')), float(a[28].replace('\n', '')), float(a[29].replace(']\n', ''))],
            [float(a[30].replace('\n', '').replace('[', '')), float(a[31].replace('\n', '')), float(a[32].replace(']\n', ''))],
            [float(a[33].replace('\n', '').replace('[', '')), float(a[34].replace('\n', '')), float(a[35].replace(']]', ''))]]
    return weights

def prepare_list_weights(j, k):
    if k > 13 or j > 13:
        with open ('../Data-Non/QGK_' + graph_type + '_List_Angles_Graphs_' + str(j) + '_' + str(k) + '.csv') as file:
            reader = file.read()
    else:
        with open ('../Data/QGK_' + graph_type + '_List_Angles_Graphs_' + str(j) + '_' + str(k) + '.csv') as file:
            reader = file.read()
    weights = []
    reader = reader.split('"')
    x=True
    while(x):
        if '\n' in reader:
            reader.remove('\n')
        if '' in reader:
            reader.remove('')
        if ',' in reader:
            reader.remove(',')
        else:
            x = False

    a = reader[-1].split(' ')
    x=True
    while(x):
        if '[' in a:
            a.remove('[')
        if ']\n' in a:
            a.remove(']\n')
        if '[[' in a:
            a.remove('[[')
        if ',' in a:
            a.remove(',')
        if '[' in a:
            a.remove('[')
        if '' in a:
            a.remove('')
        else:
            x = False
    a
    
    weights = [[float(a[0].replace('\n', '').replace('[[', '')), float(a[1].replace('\n', '')), float(a[2].replace('\n', '')), float(a[3].replace('\n', '')), float(a[4].replace('\n', '')), float(a[5].replace('\n', '')), float(a[6].replace('\n', '')), float(a[7].replace('\n', '')), float(a[8].replace(']\n', ''))],
            [float(a[9].replace('\n', '').replace('[', '')), float(a[10].replace('\n', '')), float(a[11].replace('\n', '')), float(a[12].replace('\n', '')), float(a[13].replace('\n', '')), float(a[14].replace('\n', '')), float(a[15].replace('\n', '')), float(a[16].replace('\n', '')), float(a[17].replace(']\n', ''))]
            ]

    return weights

def prepare_feature_vecs(graph_0, graph_1):
    g = nx.from_edgelist(graph_0)
    #degree_centrality = nx.degree_centrality(g)
    #degree_centrality = nx.eigenvector_centrality(g)
    model = DeepWalk(dimensions=1)
    model.fit(g)
    embedding1 = model.get_embedding()
    feature_vec_1 = []

    for i in range(len(embedding1)):
        feature_vec_1.append(embedding1[i][0])

    h = nx.from_edgelist(graph_1)
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

all_data = []
for j in range(14):
    dataframe = []
    for k in range(14):
        float_probs = []
        for l in range(len(graphs)):
            data = []
            for m in range(len(graphs)):
                graph_0 = graphs[l]
                graph_1 = graphs[m]
                weights = prepare_list_weights(j, k)
                feature_vec_1, feature_vec_2 = prepare_feature_vecs(graph_0, graph_1)
                probs = inverse_circuit(weights, feature_vec_1, feature_vec_2)[0]
                data.append(float(np.sqrt(np.real(probs)**2+np.imag(probs)**2) **2))
            float_probs.append(data)
        dataframe.append(float_probs)
    all_data.append(dataframe)

for j in range(14):
    for k in range(14):
        df = pd.DataFrame(all_data[j][k])
        svm = sns.heatmap(df, annot=True, annot_kws={"size": 5})
        figure = svm.get_figure()    
        figure.savefig('../Test/Heatmap_' + graph_type + '_List_Training_' + str(j) + '_' + str(k) + '.png', dpi=400)
        figure.clf()

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

# all_sandwich_data = []
# for j in range(14):
#     sandwich_dataframe = []
#     for k in range(14):
#         float_probs_sandwich = []
#         for l in range(len(graphs)):
#             data = []
#             for m in range(len(graphs)):
#                 graph_0 = graphs[l]
#                 graph_1 = graphs[m]
#                 weights = prepare_sandwich_weights(j, k)
#                 feature_vec_1, feature_vec_2 = prepare_feature_vecs(graph_0, graph_1)
#                 probs = inverse_sandwich_circuit(weights, feature_vec_1, feature_vec_2)[0]
#                 data.append(float(np.sqrt(np.real(probs)**2+np.imag(probs)**2) **2))
#             float_probs_sandwich.append(data)
#         sandwich_dataframe.append(float_probs_sandwich)
#     all_sandwich_data.append(sandwich_dataframe)

# for j in range(14):
#     for k in range(14):
#         sandwich_df = pd.DataFrame(all_sandwich_data[j][k])
#         sig = sns.heatmap(sandwich_df, annot=True, annot_kws={'size': 5})
#         figure = sig.get_figure()    
#         figure.savefig('../Test/Heatmap_' + graph_type + '_Sandwich_Training_' + str(j) + '_' + str(k) + '.png', dpi=400)
#         figure.clf()        

