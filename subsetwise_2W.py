import random
import networkx as nx
import math
import sys
import time
from input_functions import *

def clustering(G, cluster_size):
  U = list(G.nodes())
  cluster_arr = []
  Gc = nx.Graph()
  for v in G.nodes():
    unclustered_nghbors = []
    for v2 in G.neighbors(v):
      if v2 in U:
        unclustered_nghbors.append(v2)
    if len(unclustered_nghbors)>=cluster_size:
      unclustered_nghbors = unclustered_nghbors[:cluster_size]
      cluster = []
      for v2 in unclustered_nghbors:
        U.remove(v2)
        cluster.append(v2)
        Gc.add_edge(v, v2, weight=G[v][v2]["weight"])
        for v3 in unclustered_nghbors:
          if not v2==v3:
            if G.has_edge(v2, v3):
              Gc.add_edge(v2, v3, weight=G[v2][v3]["weight"])
      cluster_arr.append(cluster)
  for v in U:
    for v2 in G.neighbors(v):
      Gc.add_edge(v, v2, weight=G[v][v2]["weight"])
  return Gc, cluster_arr

def has_improvement(Gs, pth, cluster):
  pth_arr = [nx.shortest_path(Gs, source=pth[0], target=cluster[i], weight="weight") for i in range(len(cluster))]
  graph_dis = min([len(pth2)-1 for pth2 in pth_arr])
  pth_dis = -1
  for i, w in enumerate(pth):
    pth_dis = pth_dis + 1
    if w in cluster:break
  if pth_dis<graph_dis:
    return True
  return False

def subsetwise(G, S):
  #W = 10
  W = find_max_weight(G)
  p = math.sqrt(len(S)*W)
  Gs, cluster_arr = clustering(G, math.ceil(p))
  #print(Gs.edges(), cluster_arr)
  for u in S:
    for v in S:
      if u!=v:
        pth = nx.shortest_path(G, source=u, target=v, weight="weight")
        cst = 0
        for i, w in enumerate(pth):
          if i>0:
            if not Gs.has_edge(pth[i-1], w):
              cst = cst + 1
        val = 0
        for cluster in cluster_arr:
          if len(set(cluster).intersection(set(pth)))>0:
            if has_improvement(Gs, pth, cluster):
              val = val + 1
            else:
              pth.reverse()
              if has_improvement(Gs, pth, cluster):
                val = val + 1
        if cst <= (2*W + 1)*val:
          Gs.add_weighted_edges_from([(x, y, G[x][y]["weight"]) for x, y in zip(pth[:len(pth)-1], pth[1:])])
  #print(Gs.edges())
  subset_cost = 0
  for u, v in Gs.edges():
    #subset_cost = subset_cost + Gs[u][v]["weight"]
    subset_cost = subset_cost + 1
  return Gs, subset_cost

##G = nx.cycle_graph(10)
#G = nx.complete_graph(10)
#for u, v in G.edges():
#  G[u][v]["weight"] = random.randint(1, 10)
#S = [1, 3, 5, 9]
#subsetwise(G, S)

def verify_spanner_with_checker(G_S, G, all_pairs, check_stretch, param):
        for i in range(len(all_pairs)):
                if not (all_pairs[i][0] in G_S.nodes() and all_pairs[i][1] in G_S.nodes()):
                        return False
                if not nx.has_path(G_S, all_pairs[i][0], all_pairs[i][1]):
                        return False
                sp = nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight')
                #if not check_stretch(nx.dijkstra_path_length(G_S, all_pairs[i][0], all_pairs[i][1]), sp, param):
                if not check_stretch(nx.shortest_path_length(G_S, all_pairs[i][0], all_pairs[i][1], 'weight'), sp, param):
                        return False
        return True

def all_pairs_from_subset(s):
        s = list(s)
        all_pairs = []
        for i in range(len(s)):
                for j in range(i+1, len(s)):
                        p = []
                        p.append(s[i])
                        p.append(s[j])
                        all_pairs.append(p)
        return all_pairs

def check_stretch(spanner_sp, graph_sp, param):
  if spanner_sp<=(graph_sp+param):
    return True
  return False

def find_max_weight(G):
  mx = 0
  for e in G.edges():
    u, v = e
    wgt = G.get_edge_data(u, v, 'weight')['weight']
    if mx<wgt:
      mx = wgt
  return mx

def multi_level_spanner(G, subset_arr, single_level_solver):
  total_cost = 0
  l = len(subset_arr)
  E = set()
  for i, T in enumerate(subset_arr):
    Gs, subset_cost = single_level_solver(G, T)
    subset_cost = 0
    for u, v in Gs.edges():
      if ((u, v) not in E) and ((v, u) not in E):
        subset_cost = subset_cost + 1
        E.add((u, v))
    #print(T, subset_cost)
    total_cost = total_cost + l*subset_cost
    l = l - 1
  return total_cost

def check_subsetwise(folder_name, file_name_without_ext, output_file):
  filename = folder_name + '/' + file_name_without_ext +'.txt'
  G, subset_arr = build_networkx_graph(filename)
  #subset_arr.reverse()
  start_time = time.time()
  #Gs, subset_cost = subsetwise(G, subset_arr[0])
  subset_cost = multi_level_spanner(G, subset_arr, subsetwise)
  total_time = time.time() - start_time
  # append the outputs to a csv file
  #f = open(folder_name + '/' + output_file, 'a')
  #f.write(folder_name + ';' + file_name_without_ext + ';' + str(kruskal_cost) + ';\n')
  #f.close()
  #print(folder_name + ';' + file_name_without_ext + ';' + str(subset_cost) + ';' + str(total_time) + ';\n')
  st = set(subset_arr[0])
  #print(folder_name + ';' + file_name_without_ext + ';' + str(verify_spanner_with_checker(Gs, G, all_pairs_from_subset(st), check_stretch, 2*find_max_weight(G))) + ';' + str(total_time) + ';\n')
  print(folder_name + ';' + file_name_without_ext + ';' + str(subset_cost) + ';' + str(total_time) + ';\n')

if __name__ == '__main__':
  # 1. Folder name
  # 2. File name
  # 3. Output File name
  check_subsetwise(sys.argv[1], sys.argv[2], sys.argv[3])


