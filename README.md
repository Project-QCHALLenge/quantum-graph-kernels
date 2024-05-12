# Quantum Graph Kernels
The graph isomorphism problem is problem of great interest and NP-hard. We introduce a quantum based graph kernel solution to get an indication if two graphs are isomorphic or not. We implemented two different architectures (list and sandwich) to solve the problem via an inversion test. Our algorithm starts with a translation of the graphs into one vector, respectively, which are angle embedded in the next step. These embeddings are used in our architectures to gain a final measurement, which gives us the likelihood of isomorphism.

## Requirements
python 3
pennylane 0.32.0
numpy 1.22.3
networkx 2.6.3
matplotlib 3.7.1
karateclub 1.3.3
seaborn 0.13.2
pandas 1.3.5

## Contents
- dev: first approaches to develop the code
- example: contains example execution with different graphs
- list.py: code for list architecture
- sandwich.py: code for sandwich architecture

## Maintainers
- Sabrina Egger [ru45jiy]

