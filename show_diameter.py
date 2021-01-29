import networkx as nx
from input_functions import *
import sys

#calculate_or_plot = "calculate"
#calculate_or_plot = "plot"
calculate_or_plot = sys.argv[1]

path_to_plots_directory = "exp_all_wgt_large/"
if calculate_or_plot == "calculate":
 #diam_arr = []
 #rad_arr = []
 #path_to_plots_directory = "exp_all_wgt_large/"
 map_file = path_to_plots_directory + "id_to_file.csv"
 f = open(map_file, 'r')
 radius_file = path_to_plots_directory + "radius.txt"
 f_write = open(radius_file, "w")
 f_write.close()
 for i in range(4000):
 #for i in range(4):
  #f.readline()
  arr = f.readline().split(';')
  ID = arr[0]
  CODE_FILE = arr[1]
  ROOT_FOLDER = arr[2]
  FILE_NAME = arr[3]
  OUTPUT_FILE = arr[4]
  G, subset_arr = build_networkx_graph(path_to_plots_directory + FILE_NAME + ".txt")
  #diam_arr.append(nx.diameter(G))
  #rad_arr.append(nx.radius(G))
  diam = nx.diameter(G)
  rad = nx.radius(G)
  f_write = open(radius_file, "a")
  f_write.write(ID + ';' + FILE_NAME + ';' + str(diam) + ';' + str(rad) + "\n")
  f_write.close()
 f.close()

if calculate_or_plot == "plot":
 from matplotlib import pyplot
 import matplotlib
 matplotlib.use('Agg')

 diam_arr = []
 rad_arr = []
 radius_file = path_to_plots_directory + "radius.txt"
 f = open(radius_file, 'r')
 for i in range(4000):
 #for i in range(2000):
  arr = f.readline().split(';')
  FILE_NAME = arr[1]
  diam = int(arr[2])
  rad = int(arr[3])
  diam_arr.append(diam)
  rad_arr.append(rad)
 f.close()

 graph_type = sys.argv[2]
 distance_type = sys.argv[3]
 if graph_type == "WS":
  start = 0
  end = 1000
 elif graph_type == "ER":
  start = 1000
  end = 2000
 elif graph_type == "BA":
  start = 2000
  end = 3000
 elif graph_type == "GE":
  start = 3000
  end = 4000
 if distance_type == "radius":
  data = rad_arr[start:end]
 elif distance_type == "diameter":
  data = diam_arr[start:end]
 pyplot.hist(data)
 pyplot.title(graph_type + ',' + distance_type)
 pyplot.savefig(path_to_plots_directory + graph_type + '_' + distance_type + ".png", bbox_inches='tight')


