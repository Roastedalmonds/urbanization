import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import math
from collections import defaultdict
from sklearn.cluster import KMeans

def distance(a, b):
  return np.linalg.norm(np.array(a)-np.array(b))

def shift(location, radius, dimension):
  x, y = location
  if(eee[x][y]):
    radialX, radialY = [x-radius if x-radius >= 0 else 0, x+radius if x+radius <= dimension[0] else dimension[0]], [y-radius if y-radius >= 0 else 0, y+radius if y+radius <= dimension[1] else dimension[1]]
    if(np.all(eee[radialX[0]:radialX[1]][radialY[0]:radialY[1]])): return False
    nearest_value, nearest_point = 99999, None
    for i in range(radialX[0], radialX[1]):
      for j in range(radialY[0], radialY[1]):
        temp = distance(location, [i,j])
        if temp <= nearest_value:
          nearest_value = temp
          nearest_point = [i,j]
    return nearest_point
  
  else: return location

def stations(world, city, radius):
  stations = pd.DataFrame(columns = ['x', 'y', 'present', 'i', 'j'])
  edges, points = [], []
  i = 0
  for y in range(radius, world[1][0], 2*radius):
    i += 1
    j = 0
    for x in range(radius, world[1][1], 2*radius):
      j += 1
      result = shift([x,y], radius, world[1])
      present = 1 if x in range(city[0][0], city[1][0]) and y in range(city[0][1], city[1][1]) else 0
      if result != False:
        stations.loc[len(stations.index),:] = [result[0], result[1], present, i, j]
        points.append([result[0], result[1]])
        srcI = stations[stations['i'] == i-1][stations['j'] == j].index.tolist()
        srcJ = stations[stations['i'] == i][stations['j'] == j-1].index.tolist()
        if j != 0 and srcJ: edges.append([srcJ[0], stations.index.tolist()[-1]])
        if i != 0 and srcI: edges.append([srcI[0], stations.index.tolist()[-1]])
  return [stations[['x','y','present']], edges]

def coordinator(station, centers):
  df_group = station.groupby('labels')
  coordinates = pd.DataFrame(columns = list(range(len(centers))), index = list(range(len(centers))))
  for label, center in enumerate(centers):
    cs = centers.tolist().copy()
    cs.pop(label)
    for cluster in cs:
      bestPoint, bestScore = None, 0
      for index, row in df_group.get_group(label).iterrows():
        if row['present'] == 0: continue
        parameter = distance([row['x'], row['y']], cluster) / distance([row['x'], row['y']], center.tolist())
        if parameter > bestScore:
          bestScore = parameter
          bestPoint = index
      coordinates.loc[centers.tolist().index(cluster), centers.tolist().index(center.tolist())] = bestPoint

  points, edges = [], []
  for i, row in station.iterrows():
    points.append([row['x'], row['y']])
  for i in range(len(centers)):
    for j in range(len(centers)):
      if i == j: continue
      edges.append([coordinates.loc[i,j], coordinates.loc[j,i]])
  return points, edges, coordinates

def plot(points, graph):
    edges = []
    for src in station.index:
        for dest in station.index:
            if graph.does_edge_exist(src, dest):
                edges.append([src, dest])
    edges = edges + route
    points, edges = np.array(points), np.array(edges)
    x = points[:,0].flatten()
    y = points[:,1].flatten()

    plt.plot(x[edges.T], y[edges.T], linestyle='-', color='y', markerfacecolor='red', marker='o') 

    plt.show()

# Dijkstra's Algorithm
class Dijkstra:

  # A utility function to find the vertex with minimum dist value, from the set of vertices still in queue
  def minDistance(self,dist,queue):
    # Initialize min value and min_index as -1
    minimum = float("Inf")
    min_index = -1
    
    # from the dist array,pick one which has min value and is till in queue
    for i in range(len(dist)):
      if dist[i] < minimum and i in queue:
        minimum = dist[i]
        min_index = i
    return min_index


  # Function to print shortest path from source to j using parent array
  def printPath(self, parent, j, solution=[]):
    if parent[j] == -1 :
      solution.append(j)
      # print(j,end=" ")
      return j
    
    solution.append(j)
    self.printPath(parent , parent[j], solution)
    # print(j,end=" ")
    return solution
		

	# A utility function to print the constructed distance array
	# def printSolution(self, dist, parent):
	# 	src = 0
	# 	print("Vertex \t\tDistance from Source\tPath")
	# 	for i in range(1, len(dist)):
	# 		print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i]),end=" ")
	# 		self.printPath(parent,i)

  def dijkstra(self, graph, src, dest):
    row = len(graph)
    col = len(graph[0])

    # The output array. dist[i] will hold the shortest distance from src to i Initialize all distances as INFINITE
    dist = [float("Inf")] * row

    #Parent array to store shortest path tree
    parent = [-1] * row

    # Distance of source vertex from itself is always 0
    dist[src] = 0

    # Add all vertices in queue
    queue = []
    for i in range(row):
      queue.append(i)
      
    #Find shortest path for all vertices
    while queue:

      # Pick the minimum dist vertex from the set of vertices still in queue
      u = self.minDistance(dist,queue)

      # remove min element	
      queue.remove(u)

      # Update dist value and parent index of the adjacent vertices of the picked vertex. Consider only those vertices which are still in queue
      for i in range(col):
        if graph[u][i] and i in queue:
          if dist[u] + graph[u][i] < dist[i]:
            dist[i] = dist[u] + graph[u][i]
            parent[i] = u


    # print the constructed distance array
    return self.printPath(parent, dest, solution=[])

class Graph:
    def __init__(self):
        # dictionary containing keys that map to the corresponding vertex object
        self.vertices = {}
 
    def add_vertex(self, key):
        """Add a vertex with the given key to the graph."""
        vertex = Vertex(key)
        self.vertices[key] = vertex
 
    def get_vertex(self, key):
        """Return vertex object with the corresponding key."""
        return self.vertices[key]
 
    def __contains__(self, key):
        return key in self.vertices
 
    def add_edge(self, src_key, dest_key, weight=1):
        """Add edge from src_key to dest_key with given weight."""
        self.vertices[src_key].add_neighbour(self.vertices[dest_key], weight)
 
    def does_vertex_exist(self, key):
        return key in self.vertices
 
    def does_edge_exist(self, src_key, dest_key):
        """Return True if there is an edge from src_key to dest_key."""
        return self.vertices[src_key].does_it_point_to(self.vertices[dest_key])
 
    def display(self):
        print('Vertices: ', end='')
        for v in self:
            print(v.get_key(), end=' ')
        print()
 
        print('Edges: ')
        for v in self:
            for dest in v.get_neighbours():
                w = v.get_weight(dest)
                print('(src={}, dest={}, weight={}) '.format(v.get_key(),
                                                             dest.get_key(), w))
 
    def __len__(self):
        return len(self.vertices)
 
    def __iter__(self):
        return iter(self.vertices.values())

class Vertex:
    def __init__(self, key):
        self.key = key
        self.points_to = {}
 
    def get_key(self):
        """Return key corresponding to this vertex object."""
        return self.key
 
    def add_neighbour(self, dest, weight):
        """Make this vertex point to dest with given edge weight."""
        self.points_to[dest] = weight
 
    def get_neighbours(self):
        """Return all vertices pointed to by this vertex."""
        return self.points_to.keys()
 
    def get_weight(self, dest):
        """Get weight of edge from this vertex to dest."""
        return self.points_to[dest]
 
    def does_it_point_to(self, dest):
        """Return True if this vertex points to dest."""
        return dest in self.points_to
 
def mst_krusal(g):
    """Return a minimum cost spanning tree of the connected graph g."""
    mst = Graph() # create new Graph object to hold the MST
 
    if len(g) == 1:
        u = next(iter(g)) # get the single vertex
        mst.add_vertex(u.get_key()) # add a copy of it to mst
        return mst
 
    # get all the edges in a list
    edges = []
    for v in g:
        for n in v.get_neighbours():
            # avoid adding two edges for each edge of the undirected graph
            if v.get_key() < n.get_key():
                edges.append((v, n))
 
    # sort edges
    edges.sort(key=lambda edge: edge[0].get_weight(edge[1]))
 
    # initially, each vertex is in its own component
    component = {}
    for i, v in enumerate(g):
        component[v] = i
 
    # next edge to try
    edge_index = 0
 
    # loop until mst has the same number of vertices as g
    while len(mst) < len(g):
        u, v = edges[edge_index]
        edge_index += 1
 
        # if adding edge (u, v) will not form a cycle
        if component[u] != component[v]:
            # add to mst
          if not mst.does_vertex_exist(u.get_key()):
              mst.add_vertex(u.get_key())
          if not mst.does_vertex_exist(v.get_key()):
              mst.add_vertex(v.get_key())
          mst.add_edge(u.get_key(), v.get_key(), u.get_weight(v))
          mst.add_edge(v.get_key(), u.get_key(), u.get_weight(v))

        # merge components of u and v
        for w in g:
            if component[w] == component[v]:
                component[w] = component[u]
 
    return mst

eee = cv2.cvtColor(cv2.imread('3e.png'), cv2.COLOR_BGR2GRAY)

# Foresight
world_dimensions = [[0,0],[300, 300]]
# Bounding Box
city_dimensions = [[50,50],[250,250]]

station, connections = stations(world_dimensions, city_dimensions, 25)

kmeans = KMeans(int(math.sqrt(len(station.index))))
kmeans.fit(station[['x','y']])
station['labels'] = kmeans.labels_
centers = kmeans.cluster_centers_

points, edges, coordinates = coordinator(station, centers)


# graph = [[None]*len(station.index)]*len(station.index)
g = Dijkstra()
graph = [[None for j in range(len(station.index))] for i in range(len(station.index))]
for src, dest in connections:
  dist = distance(station.loc[src, ['x', 'y']].tolist(), station.loc[dest, ['x', 'y']].tolist())
  graph[src][dest] = graph[dest][src] = dist
route = []
for i in range(len(coordinates.index)):
  for j in range(i+1, len(coordinates.index)):
    path = g.dijkstra(graph,coordinates.loc[i,j], coordinates.loc[j,i])
    for s in range(len(path)):
      if s+1 != len(path): route.append([path[s],path[s+1]])

g = Graph()

df_group = station.groupby('labels')
for label in station.labels.unique().tolist():
  for key in df_group.get_group(label).index:
    g.add_vertex(key)
    keyX, keyY = station.loc[key, ['x','y']]
    for src in df_group.get_group(label).index:
        if src == key: break
        x, y = station.loc[src, ['x','y']]
        g.add_edge(src, key, distance([keyX, keyY],[x,y]))

mst = mst_krusal(g)
plot(points, graph=mst)

