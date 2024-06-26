# Example Experiments

This folder contains code for training and evaluation on the following experiments:
- 4-Regular Graphs and Cycle Graphs
- 4-Regular Graphs and Erdös-Renyi Graphs
- 4-Regular Graphs and Watts-Strogatz Graphs
- 4-Regular Graphs and Wheel Graphs
- Cycle Graphs and 4-Regular Graphs
- Cycle Graphs and Erdös-Renyi Graphs
- Cycle Graphs and Watts-Strogatz Graphs
- Cycle Graphs and Wheel Graphs
- Erdös-Renyi Graphs and 4-Regular Graphs
- Erdös-Renyi Graphs and Cycle Graphs
- Erdös-Renyi Graphs and Watts-Strogatz Graphs
- Erdös-Renyi Graphs and Wheel Graphs
- Watts-Strogatz Graphs and 4-Regular Graphs
- Watts-Strogatz Graphs and Cycle Graphs
- Watts-Strogatz Graphs and Erdös-Renyi Graphs
- Watts-Strogatz Graphs and Wheel Graphs
- Wheel Graphs and 4-Regular Graphs
- Wheel Graphs and Cycle Graphs
- Wheel Graphs and Erdös-Renyi Graphs
- Wheel Graphs and Watts-Strogatz Graphs

The main.py file can be used for training and the eval.py for evaluation.
Execute the files with ```python main.py``` or ```python eval.py``` in the command line/terminal. 
The ```myslurmscript.sh```can be used for execution using slurm (please adjust the file with personal settings).
For evaluation, data generated by the folders with '-Non' postfix are required, as these folders contain the code for the non-isomorphic cases (different cost function).
Data from training is stored in a 'Data'-folder, which is used for evaluation. Heatmaps visualizing the results from the evaluation are stored in a 'Test' folder.
The first 14 and the last 8 entries in the graphs array are isomorphic, respectively. 


Requirements:
python 3
pennylane 0.32.0
networkx 2.6.3
matplotlib 3.7.1
karateclub 1.3.3
seaborn 0.13.2
pandas 1.3.5