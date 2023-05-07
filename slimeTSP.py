import numpy as np
import random
import matplotlib.pyplot as plt

# Traveling Salesman Problem (TSP) Graph class
class TSPGraph:
    def __init__(self, cities, distances):
        self.cities = cities
        self.distances = distances

    # Retrieve the distance between two cities
    def get_distance(self, city1, city2):
        return self.distances[self.cities.index(city1)][self.cities.index(city2)]

    # Get the total number of cities in the graph
    def number_of_cities(self):
        return len(self.cities)

# Sample TSP problem
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

city_coords = {
    "A": (0.1, 0.3),
    "B": (0.7, 0.5),
    "C": (0.4, 0.9),
    "D": (0.6, 0.2),
    "E": (0.2, 0.8),
    "F": (0.9, 0.7),
    "G": (0.8, 0.1),
    "H": (0.5, 0.4),
}

# Slime Mold agent class for TSP
class SlimeMold:
    def __init__(self, graph, alpha, beta, decay_rate):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.decay_rate = decay_rate
        self.current_city = random.choice(graph.cities)
        self.visited_cities = [self.current_city]
        self.unvisited_cities = [city for city in graph.cities if city != self.current_city]
        self.pheromones = [[1 for _ in range(graph.number_of_cities())] for _ in range(graph.number_of_cities())]

    # Move the agent to a new city
    def move(self):
        probabilities = self._compute_probabilities()
        self.current_city = np.random.choice(self.unvisited_cities, p=probabilities)
        self.visited_cities.append(self.current_city)
        self.unvisited_cities.remove(self.current_city)

    # Compute the probabilities of moving to each unvisited city
    def _compute_probabilities(self):
        probabilities = []
        for city in self.unvisited_cities:
            last_city = self.visited_cities[-1]
            pheromone = self.pheromones[self.graph.cities.index(last_city)][self.graph.cities.index(city)]
            distance = self.graph.get_distance(last_city, city)
            probabilities.append((pheromone ** self.alpha) * ((1 / distance) ** self.beta))
        return [prob / sum(probabilities) for prob in probabilities]

    # Deposit pheromones on the path the agent has taken
    def deposit_pheromones(self):
        for i in range(len(self.visited_cities) - 1):
            city1 = self.visited_cities[i]
            city2 = self.visited_cities[i + 1]
            delta_pheromone = 1 / self.graph.get_distance(city1, city2)
            self.pheromones[self.graph.cities.index(city1)][self.graph.cities.index(city2)] += delta_pheromone
            self.pheromones[self.graph.cities.index(city2)][self.graph.cities.index(city1)] += delta_pheromone

    # Update pheromones on the path by applying the decay rate
    def update_pheromones(self):
        for i in range(len(self.visited_cities) - 1):
            city1 = self.visited_cities[i]
            city2 = self.visited_cities[i + 1]
            self.pheromones[self.graph.cities.index(city1)][self.graph.cities.index(city2)] *= 1 - self.decay_rate
            self.pheromones[self.graph.cities.index(city2)][self.graph.cities.index(city1)] *= 1 - self.decay_rate

# Run the Slime Mold simulation for TSP
def run_simulation(graph, n_agents, n_iterations, alpha, beta, decay_rate, starting_city=None):
    agents = [SlimeMold(graph, alpha, beta, decay_rate) for _ in range(n_agents)]
    best_path = None
    best_distance = float('inf')
    if not starting_city:
        starting_city = random.choice(graph.cities)

    for _ in range(n_iterations):
        for agent in agents:
            agent.current_city = starting_city  # Set the starting city
            agent.visited_cities = [agent.current_city]
            agent.unvisited_cities = [city for city in graph.cities if city != agent.current_city]
            
            while agent.unvisited_cities:
                agent.move()

            # Close the tour by returning to the starting city
            agent.visited_cities.append(agent.visited_cities[0])
            distance = sum([graph.get_distance(agent.visited_cities[i], agent.visited_cities[i + 1]) for i in range(len(agent.visited_cities) - 1)])

            # Update the best path and distance found so far
            if distance < best_distance:
                best_distance = distance
                best_path = agent.visited_cities

            agent.update_pheromones()
            agent.deposit_pheromones()

    return best_path, best_distance

# Simulation parameters
n_agents = 100
n_iterations = 2000
alpha = 3
beta = 5
decay_rate = 0.1

starting_city = "A"

best_path, best_distance = run_simulation(tsp_graph, n_agents, n_iterations, alpha, beta, decay_rate, starting_city)
print("Slime Mold best path found:", best_path)
print("Slime Mold best distance:", best_distance)


# Visualize the TSP solution
def plot_solution(graph, best_path, city_coords):
    x = [city_coords[city][0] for city in graph.cities]
    y = [city_coords[city][1] for city in graph.cities]

    plt.scatter(x, y)

    for i, city in enumerate(graph.cities):
        plt.annotate(city, (x[i], y[i]))

    for i in range(len(best_path) - 1):
        city1 = best_path[i]
        city2 = best_path[i + 1]
        city1_index = graph.cities.index(city1)
        city2_index = graph.cities.index(city2)
        plt.plot([x[city1_index], x[city2_index]], [y[city1_index], y[city2_index]], 'r--')

    plt.show()


def nearest_neighbor_algorithm(graph, starting_city=None):
    if not starting_city:
        starting_city = random.choice(graph.cities)
    current_city = starting_city
    unvisited_cities = [city for city in graph.cities if city != current_city]
    visited_cities = [current_city]

    while unvisited_cities:
        min_distance = float('inf')
        next_city = None
        for city in unvisited_cities:
            distance = graph.get_distance(current_city, city)
            if distance < min_distance:
                min_distance = distance
                next_city = city
        current_city = next_city
        visited_cities.append(current_city)
        unvisited_cities.remove(current_city)

    visited_cities.append(visited_cities[0])
    total_distance = sum([graph.get_distance(visited_cities[i], visited_cities[i + 1]) for i in range(len(visited_cities) - 1)])
    return visited_cities, total_distance

nearest_neighbor_path, nearest_neighbor_distance = nearest_neighbor_algorithm(tsp_graph, starting_city)
print("Nearest Neighbor path:", nearest_neighbor_path)
print("Nearest Neighbor distance:", nearest_neighbor_distance)

# Plot the Slime Mold solution
plot_solution(tsp_graph, best_path, city_coords)

# Plot the Nearest Neighbor solution
plot_solution(tsp_graph, nearest_neighbor_path, city_coords)