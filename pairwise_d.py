import sys
import time
from input_functions import *
import networkx as nx
import random
import math

def check_stretch(spanner_sp, graph_sp, param):
  if spanner_sp<=(graph_sp+param):
    return True
  return False

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

def d_light_initialization(G, d):
  H = nx.Graph()
  for u in G.nodes():
    edges = []
    for v in G.neighbors(u):
      edges.append([u, v, G.get_edge_data(u, v, 'weight')])
    #print(edges)
    edges = sorted(edges, key = lambda e:e[2]['weight'])
    for i in range(min(d,len(edges))):
      H.add_weighted_edges_from([(edges[i][0], edges[i][1], edges[i][2]['weight'])])
  return H

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

def proper_edge_order(a, b):
  if a>b:
    t = a
    a = b
    b = t
  return a, b

def get_missing_edges(pth, H, l):
  cur_missing_edges = set()
  for i in range(1, len(pth)):
    u = pth[i-1]
    v = pth[i]
    u, v = proper_edge_order(u, v)
    if (u, v) not in H.edges():
      cur_missing_edges.add((u, v))
    if len(cur_missing_edges)==l:
      break
  return cur_missing_edges

def get_last_missing_edges(pth, H, l):
  cur_missing_edges = set()
  i = len(pth) - 2
  while True:
    u = pth[i]
    v = pth[i+1]
    u, v = proper_edge_order(u, v)
    if (u, v) not in H.edges():
      cur_missing_edges.add((u, v))
    i = i - 1
    if i<0:
      break
    if len(cur_missing_edges)==l:
      break
  return cur_missing_edges

def shortest_path_phase(H, P, path, l):
  missing_edges_type1 = set()
  missing_edges_type2 = set()
  missing_edges_type3 = set()
  missing_edges_type4 = set()
  for s, t in P:
    pth = path[s][t]
    cur_missing_edges = get_missing_edges(pth, H, len(pth))
    if len(cur_missing_edges)<=l:
      missing_edges_type1 = missing_edges_type1.union(cur_missing_edges)
    else:
      missing_edges_type2 = missing_edges_type2.union(cur_missing_edges)
      cur_missing_edges = get_missing_edges(pth, H, l)
      missing_edges_type3 = missing_edges_type3.union(cur_missing_edges)
      cur_missing_edges = get_last_missing_edges(pth, H, l)
      missing_edges_type4 = missing_edges_type4.union(cur_missing_edges)
  return missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4

def add_missing_edges(H, missing_edges, G):
  for u, v in missing_edges:
    H.add_weighted_edges_from([(u, v, G[u][v]["weight"])])

def pairwise2W(G, S, d_frac):
  P = all_pairs_from_subset(S)
  d = math.ceil(math.pow(len(P), 1/3)/d_frac)
  valid_spanner = False
  H = d_light_initialization(G, d)
  path = dict(nx.all_pairs_dijkstra_path(G))
  n = len(list(G.nodes()))
  if len(P)>0:
    l = math.ceil(n/math.pow(len(P), 2/3))
    prob = 1/(d*l)
    missing_edges_type1, missing_edges_type2, _, _ = shortest_path_phase(H, P, path, l)
    total_missing_edges = missing_edges_type1.union(missing_edges_type2)
    while len(total_missing_edges)>(n*d):
      if verify_spanner_with_checker(Gs, G, all_pairs_from_subset(st), check_stretch, 2*find_max_weight(G)):
        valid_spanner = True
        break
      add_missing_edges(H, missing_edges_type1, G)
      for r in G.nodes():
        if random.uniform(0, 1)<=prob:
          for v in G.nodes():
            pth = path[r][v]
            cur_missing_edges = get_missing_edges(pth, H, len(pth))
            for p, q in cur_missing_edges:
              H.add_weighted_edges_from([(p, q, G[p][q]["weight"])])
      missing_edges_type1, missing_edges_type2, _, _ = shortest_path_phase(H, P, path, l)
      total_missing_edges = missing_edges_type1.union(missing_edges_type2)
    if not valid_spanner:
      add_missing_edges(H, total_missing_edges, G)
  total_cost = 0
  for u, v in H.edges():
    total_cost = total_cost + H[u][v]["weight"]
  return H, total_cost

def pairwise4W(G, S, d_frac):
  P = all_pairs_from_subset(S)
  d = math.ceil(math.pow(len(P), 2/7)/d_frac)
  valid_spanner = False
  H = d_light_initialization(G, d)
  path = dict(nx.all_pairs_dijkstra_path(G))
  n = len(list(G.nodes()))
  if len(P)>0:
    l = math.ceil(n/math.pow(len(P), 5/7))
    prob1 = d/n
    prob2 = 1/(d*l)
    missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4 = shortest_path_phase(H, P, path, l)
    total_missing_edges = missing_edges_type1.union(missing_edges_type2)
    while len(total_missing_edges)>(n*d):
      if verify_spanner_with_checker(Gs, G, all_pairs_from_subset(st), check_stretch, 4*find_max_weight(G)):
        valid_spanner = True
        break
      add_missing_edges(H, missing_edges_type1, G)
      add_missing_edges(H, missing_edges_type3, G)
      add_missing_edges(H, missing_edges_type4, G)
      for r in G.nodes():
        if random.uniform(0, 1)<=prob1:
          for v in G.nodes():
            pth = path[r][v]
            cur_missing_edges = get_missing_edges(pth, H, len(pth))
            for p, q in cur_missing_edges:
              H.add_weighted_edges_from([(p, q, G[p][q]["weight"])])
      # intermediate edges
      sample_nodes = []
      for r in G.nodes():
        if random.uniform(0, 1)<=prob2:
          sample_nodes.append(r)
      sample_pairs = all_pairs_from_subset(sample_nodes)
      for r1, r2 in sample_pairs:
        cur_missing_edges = get_missing_edges(path[r1][r2], H, len(path[r1][r2]))
        if len(cur_missing_edges)<=(n/(d*d)):
          add_missing_edges(H, cur_missing_edges, G)
      missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4 = shortest_path_phase(H, P, path, l)
      total_missing_edges = missing_edges_type1.union(missing_edges_type2)
    if not valid_spanner:
      add_missing_edges(H, missing_edges_type1, G)
      add_missing_edges(H, missing_edges_type3, G)
      add_missing_edges(H, missing_edges_type4, G)
  total_cost = 0
  for u, v in H.edges():
    total_cost = total_cost + H[u][v]["weight"]
  return H, total_cost

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

def find_max_weight(G):
  mx = 0
  for e in G.edges():
    u, v = e
    wgt = G.get_edge_data(u, v, 'weight')['weight']
    if mx<wgt:
      mx = wgt
  return mx

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

def pairwise6W(G, S, d_frac):
  P = all_pairs_from_subset(S)
  d = math.ceil(math.pow(len(P), 1/4)/d_frac)
  valid_spanner = False
  H = d_light_initialization(G, d)
  path = dict(nx.all_pairs_dijkstra_path(G))
  n = len(list(G.nodes()))
  if len(P)>0:
    l = math.ceil(n/math.pow(len(P), 3/4))
    prob = 1/(d*l)
    missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4 = shortest_path_phase(H, P, path, l)
    total_missing_edges = missing_edges_type1.union(missing_edges_type2)
    while len(total_missing_edges)>(n*d):
      if verify_spanner_with_checker(Gs, G, all_pairs_from_subset(st), check_stretch, 6*find_max_weight(G)):
        valid_spanner = True
        break
      add_missing_edges(H, missing_edges_type1, G)
      add_missing_edges(H, missing_edges_type3, G)
      add_missing_edges(H, missing_edges_type4, G)
      sample_nodes = []
      for r in G.nodes():
        if random.uniform(0, 1)<=prob:
          sample_nodes.append(r)
      G_subset, _ = subsetwise(G, sample_nodes)
      add_missing_edges(H, G_subset.edges(), G)
      missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4 = shortest_path_phase(H, P, path, l)
      total_missing_edges = missing_edges_type1.union(missing_edges_type2)
    if not valid_spanner:
      add_missing_edges(H, missing_edges_type1, G)
      add_missing_edges(H, missing_edges_type3, G)
      add_missing_edges(H, missing_edges_type4, G)
  total_cost = 0
  for u, v in H.edges():
    total_cost = total_cost + H[u][v]["weight"]
  return H, total_cost

def is_near_connected(G, H, u, v, G_sp):
  H_sp = dict(nx.shortest_path_length(H, weight="weight"))
  for x in H.neighbors(u):
    if G_sp[v][x]==H_sp[v][x]:
      return True
  for x in H.neighbors(v):
    if G_sp[u][x]==H_sp[u][x]:
      return True
  return False

def compute_subsetwise_spanner(G, S):
  d = math.ceil(math.sqrt(len(S)))
  H = d_light_initialization(G, d)
  G_sp = dict(nx.shortest_path_length(G, weight="weight"))
  for s in S:
    for t in S:
      if s==t:
        continue
      distance_preserved = False
      if is_near_connected(G, H, s, t, G_sp):
        distance_preserved = True
      else:
        sp = nx.shortest_path(G, s, t)
        for i in range(1, len(sp)-1):
          x = sp[i]
          if is_near_connected(G, H, s, x, G_sp) and is_near_connected(G, H, x, t, G_sp):
            distance_preserved = True
            break
      if not distance_preserved:
        for i in range(0, len(sp)-1):
          H.add_weighted_edges_from([(sp[i], sp[i+1], G.get_edge_data(sp[i], sp[i+1], 'weight'))])
  total_cost = 0
  for u, v in H.edges():
    total_cost = total_cost + H[u][v]["weight"]
  return H, total_cost

def pairwise8W(G, S, d_frac):
  P = all_pairs_from_subset(S)
  d = math.ceil(math.pow(len(P), 1/4)/d_frac)
  valid_spanner = False
  H = d_light_initialization(G, d)
  path = dict(nx.all_pairs_dijkstra_path(G))
  n = len(list(G.nodes()))
  l = math.ceil(n/math.pow(len(P), 3/4))
  prob = 1/(d*l)
  missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4 = shortest_path_phase(H, P, path, l)
  total_missing_edges = missing_edges_type1.union(missing_edges_type2)
  while len(total_missing_edges)>(n*d):
    if verify_spanner_with_checker(Gs, G, all_pairs_from_subset(st), check_stretch, 8*find_max_weight(G)):
      valid_spanner = True
      break
    add_missing_edges(H, missing_edges_type1, G)
    add_missing_edges(H, missing_edges_type3, G)
    add_missing_edges(H, missing_edges_type4, G)
    sample_nodes = []
    for r in G.nodes():
      if random.uniform(0, 1)<=prob:
        sample_nodes.append(r)
    G_subset, _ = compute_subsetwise_spanner(G, sample_nodes)
    add_missing_edges(H, G_subset.edges(), G)
    missing_edges_type1, missing_edges_type2, missing_edges_type3, missing_edges_type4 = shortest_path_phase(H, P, path, l)
    total_missing_edges = missing_edges_type1.union(missing_edges_type2)
  if not valid_spanner:
    add_missing_edges(H, missing_edges_type1, G)
    add_missing_edges(H, missing_edges_type3, G)
    add_missing_edges(H, missing_edges_type4, G)
  total_cost = 0
  for u, v in H.edges():
    total_cost = total_cost + H[u][v]["weight"]
  return H, total_cost


def multi_level_spanner(G, subset_arr, stretch, d_frac):
  total_cost = 0
  l = len(subset_arr)
  E = set()
  for i, T in enumerate(subset_arr):
    if stretch==2:
      Gs, cost = pairwise2W(G, T, d_frac)
    elif stretch==4:
      Gs, cost = pairwise4W(G, T, d_frac)
    elif stretch==6:
      Gs, cost = pairwise6W(G, T, d_frac)
    elif stretch==8:
      Gs, cost = pairwise8W(G, T, d_frac)
    subset_cost = 0 
    for u, v in Gs.edges():
      if ((u, v) not in E) and ((v, u) not in E):
        subset_cost = subset_cost + 1
        E.add((u, v))
    #print(T, subset_cost)
    total_cost = total_cost + l*subset_cost
    l = l - 1
  return total_cost

def check_pairwise(folder_name, file_name_without_ext, stretch, d_frac):
  filename = folder_name + '/' + file_name_without_ext +'.txt'
  G, subset_arr = build_networkx_graph(filename)
  #subset_arr.reverse()
  start_time = time.time()
  #Gs, subset_cost = subsetwise(G, subset_arr[0])
  subset_cost = multi_level_spanner(G, subset_arr, stretch, d_frac)
  total_time = time.time() - start_time
  # append the outputs to a csv file
  #f = open(folder_name + '/' + output_file, 'a')
  #f.write(folder_name + ';' + file_name_without_ext + ';' + str(kruskal_cost) + ';\n')
  #f.close()
  #print(folder_name + ';' + file_name_without_ext + ';' + str(subset_cost) + ';' + str(total_time) + ';\n')
  #st = set(subset_arr[0])
  #print(folder_name + ';' + file_name_without_ext + ';' + str(verify_spanner_with_checker(Gs, G, all_pairs_from_subset(st), check_stretch, 2*find_max_weight(G))) + ';' + str(total_time) + ';\n')
  print(folder_name + ';' + file_name_without_ext + ';' + str(subset_cost) + ';' + str(total_time) + ';\n')

if __name__ == '__main__':
  # 1. Folder name
  # 2. File name
  # 3. Output File name
  check_pairwise(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

