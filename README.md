# Traveling Salesman Problem (TSP) Solver
This repository contains a Python implementation of a solver for the Traveling Salesman Problem (TSP) using two different algorithms:

* Slime Mold Algorithm
* Nearest Neighbor Algorithm
![solution](https://imgur.com/QVmGSAV)
## Table of Contents
* Introduction
* Requirements
* Usage
* Visualization
* Contributing
### Introduction
The Traveling Salesman Problem (TSP) is an NP-hard optimization problem that seeks to find the shortest possible route for a salesman to visit a given set of cities and return to the origin city. In this repository, we implemented two algorithms to solve the TSP: the Slime Mold algorithm, which is a nature-inspired metaheuristic, and the Nearest Neighbor algorithm, which is a greedy algorithm.

### Requirements
To run the TSP Solver, you will need the following Python packages:

```
Numpy
Matplotlib
```
You can install them using pip:
```
pip install numpy matplotlib
```
### Usage
To use the TSP Solver, you will need to define your TSP graph with cities and their pairwise distances. Then, you can run the solver with the Slime Mold and Nearest Neighbor algorithms. Here's an example:
```
from tsp_solver import TSPGraph, run_simulation, nearest_neighbor_algorithm, plot_solution

cities = ["A", "B", "C", "D", "E", "F", "G", "H"]
distances = [
    [0, 12, 29, 22, 13, 24, 30, 25],
    [12, 0, 19, 3, 25, 41, 23, 11],
    [29, 19, 0, 21, 46, 25, 50, 37],
    [22, 3, 21, 0, 23, 39, 22, 14],
    [13, 25, 46, 23, 0, 25, 19, 36],
    [24, 41, 25, 39, 25, 0, 29, 27],
    [30, 23, 50, 22, 19, 29, 0, 12],
    [25, 11, 37, 14, 36, 27, 12, 0]
]

tsp_graph = TSPGraph(cities, distances)
starting_city = "A"

# Slime Mold Algorithm
slime_mold_path, slime_mold_distance = run_simulation(tsp_graph, n_agents=100, n_iterations=2000, alpha=3, beta=5, decay_rate=0.1, starting_city=starting_city)
print("Slime Mold path:", slime_mold_path)
print("Slime Mold distance:", slime_mold_distance)

# Nearest Neighbor Algorithm
nearest_neighbor_path, nearest_neighbor_distance = nearest_neighbor_algorithm(tsp_graph, starting_city)
print("Nearest Neighbor path:", nearest_neighbor_path)
print("Nearest Neighbor distance:", nearest_neighbor_distance)
```
