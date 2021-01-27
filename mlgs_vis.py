import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import matplotlib.patches as mpatches
from input_functions import *
from matplotlib.ticker import MaxNLocator
import random
import os
from input_functions import *
import math

folder_name = "exp_all_wgt_subset_6"
# ER
#y_min, y_max = 1.9, 15.0
# WS
#y_min, y_max = 2.9, 14.0
# BA
#y_min, y_max = 2.9, 16.0
# GE
#y_min, y_max = 0.9, 11.0
# ER pair 2W
#y_min, y_max = 1.9, 8.0
# WS pair 2W
#y_min, y_max = 1.9, 9.0
# BA pair 2W
#y_min, y_max = 1.9, 10.0
# GE pair 2W
#y_min, y_max = 0.9, 7.0
# ER global vs local
#y_min, y_max = 1.9, 15.0
# WS global vs local
#y_min, y_max = 1.9, 14.0
# BA global vs local
#y_min, y_max = 1.9, 16.0
# GE global vs local
#y_min, y_max = 0.9, 11.0
# ER pair 4W
#y_min, y_max = 0.9, 8.0
# WS pair 4W
#y_min, y_max = 1.9, 9.0
# BA pair 4W
#y_min, y_max = 0.9, 10.0
# GE pair 4W
#y_min, y_max = 0.9, 8.0
# ER local vs local
#y_min, y_max = 0.9, 8.0
# WS local vs local
#y_min, y_max = 1.9, 9.0
# BA local vs local
#y_min, y_max = 0.9, 10.0
#GE local vs local
#y_min, y_max = 0.9, 8.0
# ER pair 6W
#y_min, y_max = 0.9, 8.0
# WS pair 6W
#y_min, y_max = 1.9, 9.0
# BA pair 6W
#y_min, y_max = 0.9, 9.0
# GE pair 6W
y_min, y_max = 0.9, 7.0
# ER global vs global
#y_min, y_max = 0.9, 15.0
# WS global vs global
#y_min, y_max = 1.9, 14.0
# BA global vs global
#y_min, y_max = 0.9, 16.0
# GE global vs global
#y_min, y_max = 0.9, 11.0
#folder_name = "exp_all_wgt_large"
#y_min, y_max = 0.9, 11.0

def parse_id_csv(experiment_folder, graph_type):
  ids = []
  folders = []
  stretch = []
  file_names = []
  level = []
  nlevel = []
  node = []
  f = open(experiment_folder + 'id_to_file.csv', 'r')
  line_number = 1
  while True:
   line = f.readline()
   if line=='':
    break
   arr = line.split(';')
   FILE_NAME = arr[3]
   if not graph_type in FILE_NAME:
    continue
   file_names.append(FILE_NAME)
   ids.append(arr[0])
   CODE_FILE = arr[1]
   ROOT_FOLDER = arr[2]
   folders.append(ROOT_FOLDER)
   STRETCH_FACTOR = float(arr[4])
   stretch.append(str(STRETCH_FACTOR))
   line_number += 1
  f.close()
  for i in range(len(folders)):
   node.append(int(file_names[i].split('_')[5]))
   level.append(file_names[i].split('_')[4])
   nlevel.append(int(file_names[i].split('_')[3]))
  return ids, folders, stretch, file_names, level, node, nlevel

def read_output_file(out_dir, file_name, stretch):
      solution_value = -1
      f = open(out_dir+'print_log_'+file_name+'_stretch_'+str(stretch)+'.txt')
      #print(f.read())
      s = f.read()
      if 'Solution value  = ' in s:
        arr = s.split('Solution value  = ')
        val = float(arr[1].strip())
        solution_value = val
      return solution_value

def parse_time_data(file_name):
  f = open(file_name)
  s = str(f.read())
  if s=='':return -1
  s = s.split('Total (root+branch&cut) =')
  total_time = 0
  for i in range(1, len(s)):
    total_time += float(s[i].split('sec.')[0].strip())
  f.close()
  return total_time

def get_approx_info(file_name):
  f = open(file_name)
  s = str(f.read())
  if s=='':return -1
  s = s.split(';')
  edges = float(s[2])
  time = float(s[3])
  return edges, time

#print(read_output_file('exp_ER_wgt_subset_3/', 'graph_ER_300_1_L_10_0', '2.0'))
#print(parse_id_csv("exp_ER_wgt_subset_3/"))
#print(parse_time_data("exp_ER_wgt_subset_3/log_folder/output_51.dat"))
#print(get_approx_info("exp_ER_wgt_subset_3/log_folder_subset/output_1.dat"))

def line_plot_old(input_folder, plot_folder, graph_type, X_plot_type):
  ids, folders, stretch, file_names, level, node, nlevel = parse_id_csv(input_folder+"/", graph_type)
  max_dict = OrderedDict()
  min_dict = OrderedDict()
  sum_dict = OrderedDict()
  count_dict = OrderedDict()
  for i, id in enumerate(ids):
    #if i==50:
    #  break
    #if node[i]>30:
    if nlevel[i]>1:
      continue
    apprx_val, apprx_time = get_approx_info(folders[i] + '/' + "log_folder_subset" + '/' + "output_" + ids[i] + ".dat")
    exact_out = get_approx_info(folders[i] + '/' + "log_folder" + '/' + "output_" + ids[i] + ".dat")
    if exact_out==-1:
      continue
    exact_val, exact_time = exact_out
    #exact_val = read_output_file(folders[i] + '/', file_names[i], stretch[i])
    #exact_time = parse_time_data(folders[i] + '/' + "log_folder" + '/' + "output_" + ids[i] + ".dat")
    if X_plot_type=="ratio":
      rat = apprx_val/exact_val
    elif X_plot_type=="apprx_time":
      rat = apprx_time
    elif X_plot_type=="exact_time":
      rat = exact_time
    #print(apprx_val, exact_val)
    #print(apprx_time, exact_time)
    dependent_var = node
    #dependent_var = nlevel
    if dependent_var[i] in max_dict.keys():
        if max_dict[dependent_var[i]]<rat:
          max_dict[dependent_var[i]] = rat
        if min_dict[dependent_var[i]]>rat:
          min_dict[dependent_var[i]] = rat
        sum_dict[dependent_var[i]] = sum_dict[dependent_var[i]] + rat
        count_dict[dependent_var[i]] = count_dict[dependent_var[i]] + 1
    else:
        max_dict[dependent_var[i]] = rat
        min_dict[dependent_var[i]] = rat
        sum_dict[dependent_var[i]] = rat
        count_dict[dependent_var[i]] = 1

    if X_plot_type=="ratio":
      path_to_plots_directory, file_name = plot_folder + "/", "NVR.png"
    elif X_plot_type=="apprx_time":
      path_to_plots_directory, file_name = plot_folder + "/", "NVT_apprx.png"
    elif X_plot_type=="exact_time":
      path_to_plots_directory, file_name = plot_folder + "/", "NVT_exact.png"
    dependent='node'
    #dependent='level'
    label = []
    max_ratios = []
    min_ratios = []
    avg_ratios = []
    for k in max_dict.keys():
      label.append(k)
      max_ratios.append(max_dict[k])
      min_ratios.append(min_dict[k])
      avg_ratios.append(sum_dict[k]/count_dict[k])
    print(label, max_ratios, avg_ratios, min_ratios)

    #if dependent=='level':
    #  ax = plt.figure(fig_count).gca()
    #  fig_count = fig_count + 1
    #  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.plot(label, max_ratios, 'r--', label, avg_ratios, 'bs', label, min_ratios, 'g^')
    plt.plot(label, max_ratios, 'ro', label, avg_ratios, 'bs', label, min_ratios, 'g^')
    if dependent=='node':
      plt.xlabel('Number of vertices', fontsize=20)
    elif dependent=='level':
      plt.xlabel('Number of levels', fontsize=20)
    elif dependent=='stretch':
      plt.xlabel('Stretch factors', fontsize=20)
    if X_plot_type=="ratio":
      plt.ylabel('Ratio', fontsize=20)
    else:
      plt.ylabel('Time (seconds)', fontsize=20)
    #plt.ylim(1,max_label)
    #plt.ylim(.94,1.8)
    #plt.ylim(.94,1.9)
    #plt.ylim(y_min, y_max)
    plt.legend(['max', 'avg', 'min'], loc='upper right', fontsize=16)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    #plt.show()
    plt.savefig(path_to_plots_directory+file_name, bbox_inches='tight')
    plt.close()  

#line_plot("exp_ER_wgt_subset_3", "plots", '')
#line_plot("exp_GE_wgt_subset_5", "plots_GE", '')
#line_plot("exp_all_wgt_subset_6", "plots_WS", 'WS', 'ratio')
#line_plot("exp_all_wgt_subset_6", "plots_WS", 'WS', 'apprx_time')
#line_plot("exp_all_wgt_subset_6", "plots_WS", 'WS', 'exact_time')
#line_plot("exp_all_wgt_subset_6", "plots_WS", 'ER', 'ratio')

fig_count = 0
my_inf = 10000000

def read_id_file(filename, graph_type):
  res = dict()
  f = open(filename, 'r')
  l = f.readline()
  while len(l)>0:
    if len(l)<=1:
      l = f.readline()
      continue
    arr = l.split(";")
    if graph_type not in arr[3]:
      l = f.readline()
      continue
    res[arr[3]] = arr[0]
    l = f.readline()
  f.close()
  return res

def parse_output_file(log_folder, log_folder_heu, specific_node_size, graph_type):
  global folder_name
  ids = read_id_file(folder_name+"/id_to_file.csv", graph_type)
  node = []
  level = []
  terminal = []
  number_of_terminals = []
  obj = []
  heu_obj = []
  #print(ids)
  for graph in ids.keys():
    filename = folder_name + "/" + log_folder + "/output_" + ids[graph] + ".dat"
    #print(filename)
    f = open(filename, 'r')
    #l = f.readline()
    l = str(f.read())
    if l=='':continue
    arr = l.split(";")
    info = arr[1].split("_") #graph_ER_30_2_L_10_0
    if not specific_node_size == None:
      if int(info[5])!=specific_node_size:
        continue
    node.append(int(info[5]))
    level.append(int(info[3]))
    size = int(info[5])
    levels = int(info[3])
    if info[4]=='L':
      terminal.append(0)
      if size<10:
        steiner_nodes = size-2
      else:
        steiner_nodes = int(size*(levels)/(levels+1))
      number_of_terminals.append(steiner_nodes)
    elif info[4]=='E':
      terminal.append(1)
      if size<10:
        steiner_nodes = size-2
      else:
        steiner_nodes = int(math.ceil(size/2.0))
      number_of_terminals.append(steiner_nodes)
    obj.append(float(arr[2]))
    f.close()

    filename = folder_name + "/" + log_folder_heu + "/output_" + ids[graph] + ".dat"
    f = open(filename, 'r')
    l = f.readline()
    arr = l.split(";")
    info = arr[1].split("_") #graph_ER_30_2_L_10_0
    heu_obj.append(float(arr[2]))
    f.close()
  return node, level, terminal, number_of_terminals, obj, heu_obj

def parse_output_file_apprx(log_folder_heu, specific_node_size, graph_type):
  global folder_name
  ids = read_id_file(folder_name+"/id_to_file.csv", graph_type)
  node = []
  level = []
  terminal = []
  number_of_terminals = []
  heu_obj = []
  #print(ids)
  for graph in ids.keys():
    filename = folder_name + "/" + log_folder_heu + "/output_" + ids[graph] + ".dat"
    #print(filename)
    f = open(filename, 'r')
    #l = f.readline()
    l = str(f.read())
    #if l=='':continue
    #print(l, len(l))
    if len(l)<5:
      node.append(-1)
      level.append(-1)
      terminal.append(-1)
      number_of_terminals.append(-1)
      heu_obj.append(my_inf)
      continue
    arr = l.split(";")
    info = arr[1].split("_") #graph_ER_30_2_L_10_0
    if not specific_node_size == None:
      if int(info[5])!=specific_node_size:
        continue
    node.append(int(info[5]))
    level.append(int(info[3]))
    size = int(info[5])
    levels = int(info[3])
    if info[4]=='L':
      terminal.append(0)
      if size<10:
        steiner_nodes = size-2
      else:
        steiner_nodes = int(size*(levels)/(levels+1))
      number_of_terminals.append(steiner_nodes)
    elif info[4]=='E':
      terminal.append(1)
      if size<10:
        steiner_nodes = size-2
      else:
        steiner_nodes = int(math.ceil(size/2.0))
      number_of_terminals.append(steiner_nodes)
    heu_obj.append(float(arr[2]))
    f.close()

  return node, level, terminal, number_of_terminals, heu_obj

def line_plot(file_name, log_folder, out_dir, dependent, y_min, y_max, graph_type, specific_node_size=None):
  global fig_count
  global folder_name
  experiment_folder = folder_name + '/'
  path_to_plots_directory = experiment_folder + 'plots/'
  node, level, terminal, number_of_terminals, obj, heu_obj = parse_output_file(log_folder, out_dir, specific_node_size, graph_type)
  #print(obj)
  #print(heu_obj)
  #max_dict = OrderedDict()
  #min_dict = OrderedDict()
  #sum_dict = OrderedDict()
  #count_dict = OrderedDict()
  max_dict = dict()
  min_dict = dict()
  sum_dict = dict()
  count_dict = dict()
  if dependent=='node':
    dependent_var = node
  elif dependent=='level':
    dependent_var = level
  elif dependent=='terminal':
    dependent_var = terminal
  elif dependent=='number_of_terminals':
    dependent_var = number_of_terminals
  for i in range(len(obj)):
    if obj[i]!=-1 and heu_obj[i]!=-1:
      rat = heu_obj[i]/obj[i]
      if rat<1.0:rat=1.0
      if dependent_var[i] in max_dict.keys():
        if max_dict[dependent_var[i]]<rat:
          max_dict[dependent_var[i]] = rat
        if min_dict[dependent_var[i]]>rat:
          min_dict[dependent_var[i]] = rat
        sum_dict[dependent_var[i]] = sum_dict[dependent_var[i]] + rat
        count_dict[dependent_var[i]] = count_dict[dependent_var[i]] + 1
      else:
        max_dict[dependent_var[i]] = rat
        min_dict[dependent_var[i]] = rat
        sum_dict[dependent_var[i]] = rat
        count_dict[dependent_var[i]] = 1
  label = []
  max_ratios = []
  min_ratios = []
  avg_ratios = []
  for k in max_dict.keys():
    if dependent=="terminal":
      if k==0:
        label.append("Linear")
      else:
        label.append("Exponential")
    else:
      label.append(k)
    max_ratios.append(max_dict[k])
    min_ratios.append(min_dict[k])
    avg_ratios.append(sum_dict[k]/count_dict[k])
  zip_x = zip(label, max_ratios, avg_ratios, min_ratios)
  zip_x = sorted(zip_x, key=lambda p: p[0])
  label, max_ratios, avg_ratios, min_ratios = zip(*zip_x)
  print(label, max_ratios, avg_ratios, min_ratios)

  if dependent=='level':
    ax = plt.figure(fig_count).gca()
    fig_count = fig_count + 1
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  #plt.plot(label, max_ratios, 'r--', label, avg_ratios, 'bs', label, min_ratios, 'g^')
  plt.plot(label, max_ratios, 'ro', label, avg_ratios, 'bs', label, min_ratios, 'g^')
  #plt.plot(label, max_ratios, 'r--', label, avg_ratios, 'b--', label, min_ratios, 'g--')
  if dependent=='node':
    plt.xlabel('Number of vertices', fontsize=20)
  elif dependent=='level':
    plt.xlabel('Number of levels', fontsize=20)
  elif dependent=='terminal':
    plt.xlabel('Terminal selection method', fontsize=20)
  elif dependent=='number_of_terminals':
    plt.xlabel('Number of terminals', fontsize=20)
  plt.ylabel('Ratio', fontsize=20)
  #plt.ylim(1,max_label)
  #plt.ylim(.94,1.8)
  #plt.ylim(.94,1.9)
  plt.ylim(y_min, y_max)
  plt.legend(['max', 'avg', 'min'], loc='upper right', fontsize=16)
  plt.tick_params(axis='x', labelsize=16)
  plt.tick_params(axis='y', labelsize=16)
  #plt.show()
  plt.savefig(path_to_plots_directory+file_name, bbox_inches='tight')
  plt.close()

'''
line_plot("SUB_NVR_ER.png", "log_folder", "log_folder_subset", "node", y_min, y_max, "ER")
line_plot("SUB_LVR_ER.png", "log_folder", "log_folder_subset", "level", y_min, y_max, "ER")
line_plot("SUB_TVR_ER.png", "log_folder", "log_folder_subset", "terminal", y_min, y_max, "ER")
line_plot("SUB_NTVR_ER.png", "log_folder", "log_folder_subset", "number_of_terminals", y_min, y_max, "ER")
'''
'''
line_plot("SUB_NVR_WS.png", "log_folder", "log_folder_subset", "node", y_min, y_max, "WS")
line_plot("SUB_LVR_WS.png", "log_folder", "log_folder_subset", "level", y_min, y_max, "WS")
line_plot("SUB_TVR_WS.png", "log_folder", "log_folder_subset", "terminal", y_min, y_max, "WS")
line_plot("SUB_NTVR_WS.png", "log_folder", "log_folder_subset", "number_of_terminals", y_min, y_max, "WS")
'''
'''
line_plot("SUB_NVR_BA.png", "log_folder", "log_folder_subset", "node", y_min, y_max, "BA")
line_plot("SUB_LVR_BA.png", "log_folder", "log_folder_subset", "level", y_min, y_max, "BA")
line_plot("SUB_TVR_BA.png", "log_folder", "log_folder_subset", "terminal", y_min, y_max, "BA")
line_plot("SUB_NTVR_BA.png", "log_folder", "log_folder_subset", "number_of_terminals", y_min, y_max, "BA")
'''
'''
line_plot("SUB_NVR_GE.png", "log_folder", "log_folder_subset", "node", y_min, y_max, "GE")
line_plot("SUB_LVR_GE.png", "log_folder", "log_folder_subset", "level", y_min, y_max, "GE")
line_plot("SUB_TVR_GE.png", "log_folder", "log_folder_subset", "terminal", y_min, y_max, "GE")
line_plot("SUB_NTVR_GE.png", "log_folder", "log_folder_subset", "number_of_terminals", y_min, y_max, "GE")
'''
'''
line_plot("PAIR_2W_NVR_ER.png", "log_folder_local", "log_folder_pairwise2W", "node", y_min, y_max, "ER")
line_plot("PAIR_2W_LVR_ER.png", "log_folder_local", "log_folder_pairwise2W", "level", y_min, y_max, "ER")
line_plot("PAIR_2W_TVR_ER.png", "log_folder_local", "log_folder_pairwise2W", "terminal", y_min, y_max, "ER")
line_plot("PAIR_2W_NTVR_ER.png", "log_folder_local", "log_folder_pairwise2W", "number_of_terminals", y_min, y_max, "ER")
'''
'''
line_plot("PAIR_2W_NVR_WS.png", "log_folder_local", "log_folder_pairwise2W", "node", y_min, y_max, "WS")
line_plot("PAIR_2W_LVR_WS.png", "log_folder_local", "log_folder_pairwise2W", "level", y_min, y_max, "WS")
line_plot("PAIR_2W_TVR_WS.png", "log_folder_local", "log_folder_pairwise2W", "terminal", y_min, y_max, "WS")
line_plot("PAIR_2W_NTVR_WS.png", "log_folder_local", "log_folder_pairwise2W", "number_of_terminals", y_min, y_max, "WS")
'''
'''
line_plot("PAIR_2W_NVR_BA.png", "log_folder_local", "log_folder_pairwise2W", "node", y_min, y_max, "BA")
line_plot("PAIR_2W_LVR_BA.png", "log_folder_local", "log_folder_pairwise2W", "level", y_min, y_max, "BA")
line_plot("PAIR_2W_TVR_BA.png", "log_folder_local", "log_folder_pairwise2W", "terminal", y_min, y_max, "BA")
line_plot("PAIR_2W_NTVR_BA.png", "log_folder_local", "log_folder_pairwise2W", "number_of_terminals", y_min, y_max, "BA")
'''
'''
line_plot("PAIR_2W_NVR_GE.png", "log_folder_local", "log_folder_pairwise2W", "node", y_min, y_max, "GE")
line_plot("PAIR_2W_LVR_GE.png", "log_folder_local", "log_folder_pairwise2W", "level", y_min, y_max, "GE")
line_plot("PAIR_2W_TVR_GE.png", "log_folder_local", "log_folder_pairwise2W", "terminal", y_min, y_max, "GE")
line_plot("PAIR_2W_NTVR_GE.png", "log_folder_local", "log_folder_pairwise2W", "number_of_terminals", y_min, y_max, "GE")
'''
'''
line_plot("PAIR_4W_NVR_ER.png", "log_folder_local4W", "log_folder_pairwise4W", "node", y_min, y_max, "ER")
line_plot("PAIR_4W_LVR_ER.png", "log_folder_local4W", "log_folder_pairwise4W", "level", y_min, y_max, "ER")
line_plot("PAIR_4W_TVR_ER.png", "log_folder_local4W", "log_folder_pairwise4W", "terminal", y_min, y_max, "ER")
line_plot("PAIR_4W_NTVR_ER.png", "log_folder_local4W", "log_folder_pairwise4W", "number_of_terminals", y_min, y_max, "ER")
'''
'''
line_plot("PAIR_4W_NVR_WS.png", "log_folder_local4W", "log_folder_pairwise4W", "node", y_min, y_max, "WS")
line_plot("PAIR_4W_LVR_WS.png", "log_folder_local4W", "log_folder_pairwise4W", "level", y_min, y_max, "WS")
line_plot("PAIR_4W_TVR_WS.png", "log_folder_local4W", "log_folder_pairwise4W", "terminal", y_min, y_max, "WS")
line_plot("PAIR_4W_NTVR_WS.png", "log_folder_local4W", "log_folder_pairwise4W", "number_of_terminals", y_min, y_max, "WS")
'''
'''
line_plot("PAIR_4W_NVR_BA.png", "log_folder_local4W", "log_folder_pairwise4W", "node", y_min, y_max, "BA")
line_plot("PAIR_4W_LVR_BA.png", "log_folder_local4W", "log_folder_pairwise4W", "level", y_min, y_max, "BA")
line_plot("PAIR_4W_TVR_BA.png", "log_folder_local4W", "log_folder_pairwise4W", "terminal", y_min, y_max, "BA")
line_plot("PAIR_4W_NTVR_BA.png", "log_folder_local4W", "log_folder_pairwise4W", "number_of_terminals", y_min, y_max, "BA")
'''
'''
line_plot("PAIR_4W_NVR_GE.png", "log_folder_local4W", "log_folder_pairwise4W", "node", y_min, y_max, "GE")
line_plot("PAIR_4W_LVR_GE.png", "log_folder_local4W", "log_folder_pairwise4W", "level", y_min, y_max, "GE")
line_plot("PAIR_4W_TVR_GE.png", "log_folder_local4W", "log_folder_pairwise4W", "terminal", y_min, y_max, "GE")
line_plot("PAIR_4W_NTVR_GE.png", "log_folder_local4W", "log_folder_pairwise4W", "number_of_terminals", y_min, y_max, "GE")
'''
'''
line_plot("PAIR_6W_NVR_ER.png", "log_folder_global6W", "log_folder_pairwise6W", "node", y_min, y_max, "ER")
line_plot("PAIR_6W_LVR_ER.png", "log_folder_global6W", "log_folder_pairwise6W", "level", y_min, y_max, "ER")
line_plot("PAIR_6W_TVR_ER.png", "log_folder_global6W", "log_folder_pairwise6W", "terminal", y_min, y_max, "ER")
line_plot("PAIR_6W_NTVR_ER.png", "log_folder_global6W", "log_folder_pairwise6W", "number_of_terminals", y_min, y_max, "ER")
'''
'''
line_plot("PAIR_6W_NVR_WS.png", "log_folder_global6W", "log_folder_pairwise6W", "node", y_min, y_max, "WS")
line_plot("PAIR_6W_LVR_WS.png", "log_folder_global6W", "log_folder_pairwise6W", "level", y_min, y_max, "WS")
line_plot("PAIR_6W_TVR_WS.png", "log_folder_global6W", "log_folder_pairwise6W", "terminal", y_min, y_max, "WS")
line_plot("PAIR_6W_NTVR_WS.png", "log_folder_global6W", "log_folder_pairwise6W", "number_of_terminals", y_min, y_max, "WS")
'''
'''
line_plot("PAIR_6W_NVR_BA.png", "log_folder_global6W", "log_folder_pairwise6W", "node", y_min, y_max, "BA")
line_plot("PAIR_6W_LVR_BA.png", "log_folder_global6W", "log_folder_pairwise6W", "level", y_min, y_max, "BA")
line_plot("PAIR_6W_TVR_BA.png", "log_folder_global6W", "log_folder_pairwise6W", "terminal", y_min, y_max, "BA")
line_plot("PAIR_6W_NTVR_BA.png", "log_folder_global6W", "log_folder_pairwise6W", "number_of_terminals", y_min, y_max, "BA")
'''
'''
line_plot("PAIR_6W_NVR_GE.png", "log_folder_global6W", "log_folder_pairwise6W", "node", y_min, y_max, "GE")
line_plot("PAIR_6W_LVR_GE.png", "log_folder_global6W", "log_folder_pairwise6W", "level", y_min, y_max, "GE")
line_plot("PAIR_6W_TVR_GE.png", "log_folder_global6W", "log_folder_pairwise6W", "terminal", y_min, y_max, "GE")
line_plot("PAIR_6W_NTVR_GE.png", "log_folder_global6W", "log_folder_pairwise6W", "number_of_terminals", y_min, y_max, "GE")
'''

def parse_output_file_for_time(log_folder, specific_node_size, graph_type):
  global folder_name
  ids = read_id_file(folder_name+"/id_to_file.csv", graph_type)
  node = []
  level = []
  terminal = []
  number_of_terminals = []
  time = []
  #print(ids)
  for graph in ids.keys():
    filename = folder_name + "/" + log_folder + "/output_" + ids[graph] + ".dat"
    #print(filename)
    f = open(filename, 'r')
    l = str(f.read())
    if l=='':continue
    arr = l.split(";")
    if float(arr[2])>14400:continue
    #l = f.readline()
    #arr = l.split(";")
    info = arr[1].split("_") #graph_ER_30_2_L_10_0
    if not specific_node_size == None:
      #if int(info[5])!=specific_node_size:
      if int(info[5]) not in specific_node_size:
        continue
    node.append(int(info[5]))
    level.append(int(info[3]))
    size = int(info[5])
    levels = int(info[3])
    if info[4]=='L':
      terminal.append(0)
      if size<10:
        steiner_nodes = size-2
      else:
        steiner_nodes = int(size*(levels)/(levels+1))
      number_of_terminals.append(steiner_nodes)
    elif info[4]=='E':
      terminal.append(1)
      if size<10:
        steiner_nodes = size-2
      else:
        steiner_nodes = int(math.ceil(size/2.0))
      number_of_terminals.append(steiner_nodes)
    time.append(float(arr[3]))
    f.close()
  return node, level, terminal, number_of_terminals, time

def box_plot_single(experiment_folder, log_folder, file_name, dependent, y_min, y_max, graph_type, specific_node_size=None):
  global fig_count, my_inf
  path_to_plots_directory = experiment_folder + 'plots/'
  node, level, terminal, number_of_terminals, obj = parse_output_file_for_time(log_folder, specific_node_size, graph_type)
  #print(obj[:10], qos_obj[:10])
  #print(obj2[:10], krus_obj[:10])
  time_dict = OrderedDict()
  if dependent=='node':
    dependent_var = node
  elif dependent=='level':
    dependent_var = level
  elif dependent=='terminal':
    dependent_var = terminal
  elif dependent=='number_of_terminals':
    dependent_var = number_of_terminals
  len_obj = len(obj)
  for i in range(len_obj):
    if obj[i]!=-1:
      if dependent_var[i] not in time_dict.keys():
        time_dict[dependent_var[i]] = []
      time_dict[dependent_var[i]].append(obj[i])
  #print("qos_dict", qos_dict)
  size = len(time_dict.keys())
  label_ind = 0
  labels = []
  sorted_keys = []
  for k in time_dict.keys():
    sorted_keys.append(k)
  sorted_keys.sort()
  for k in sorted_keys:
    if dependent=="terminal":
      if k==0:
        labels.append("Linear")
        labels.append('')
      else:
        labels.append("Exponential")
        labels.append('')
      continue
    if (dependent=='node' or dependent=='number_of_terminals') and ((k%2)==1):
      labels.append('')
      labels.append('')
      continue
    labels.append(k)
    labels.append('')
  data = []
  gaps = 1
  i = 0
  for k in sorted_keys:
    data.append(sorted(time_dict[k]))
    if i<size-1:
      for g in range(gaps):
        # some space
        data.append([])
    i = i + 1

  plt.figure(fig_count)
  fig_count = fig_count + 1
  #bp = plt.boxplot(data, 0, '', whis=1000, patch_artist=True)
  medianprops = dict(color='k')
  bp = plt.boxplot(data, 0, '', whis=1000, patch_artist=True, medianprops=medianprops)
  # labels computation is complex, it has add 2 for gaps, negate 2 for boundary condition
  #tmp = range(1,size*(len(text)+gaps)+1-2)
  tmp = range(1,len(labels)+1)
  tmp2 = labels
  plt.xticks(tmp, tmp2)
  if dependent=='node':
    plt.xlabel('Number of vertices', fontsize=20)
  elif dependent=='level':
    plt.xlabel('Number of priorities', fontsize=20)
  elif dependent=='terminal':
    plt.xlabel('Terminal selection method', fontsize=20)
  elif dependent=='number_of_terminals':
    plt.xlabel('Number of terminals', fontsize=20)
  plt.ylabel('Time (seconds)', fontsize=20)
  #plt.ylim(y_min, y_max)
  plt.tick_params(axis='x', labelsize=16)
  plt.tick_params(axis='y', labelsize=16)
  plt.show()
  plt.savefig(path_to_plots_directory+file_name, bbox_inches='tight')
  plt.close()

'''
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_subset",  "SUB_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_LVT_box_ER.png", "level", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder", "NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder",  "TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder", "LVT_box_ER.png", "level", y_min, y_max, "ER")
'''
'''
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_LVT_box_WS.png", "level", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder", "NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder", "TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder", "LVT_box_WS.png", "level", y_min, y_max, "WS")
'''
'''
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_subset",  "SUB_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_LVT_box_BA.png", "level", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder", "NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder",  "TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder", "LVT_box_BA.png", "level", y_min, y_max, "BA")
'''
'''
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_subset", "SUB_LVT_box_GE.png", "level", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder", "NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder", "TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder", "LVT_box_GE.png", "level", y_min, y_max, "GE")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_pairwise2W",  "PAIR_2W_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_LVT_box_ER.png", "level", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_local",  "LOC_2W_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_LVT_box_ER.png", "level", y_min, y_max, "ER")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_pairwise2W",  "PAIR_2W_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_LVT_box_WS.png", "level", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_local",  "LOC_2W_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_LVT_box_WS.png", "level", y_min, y_max, "WS")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_pairwise2W",  "PAIR_2W_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_LVT_box_BA.png", "level", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_local",  "LOC_2W_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_LVT_box_BA.png", "level", y_min, y_max, "BA")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_pairwise2W",  "PAIR_2W_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_pairwise2W", "PAIR_2W_LVT_box_GE.png", "level", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_local",  "LOC_2W_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_local", "LOC_2W_LVT_box_GE.png", "level", y_min, y_max, "GE")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_LVT_box_ER.png", "level", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_LVT_box_ER.png", "level", y_min, y_max, "ER")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_LVT_box_WS.png", "level", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_LVT_box_WS.png", "level", y_min, y_max, "WS")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_LVT_box_BA.png", "level", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_LVT_box_BA.png", "level", y_min, y_max, "BA")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_pairwise4W", "PAIR_4W_LVT_box_GE.png", "level", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_local4W", "LOC_4W_LVT_box_GE.png", "level", y_min, y_max, "GE")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_LVT_box_ER.png", "level", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_LVT_box_ER.png", "level", y_min, y_max, "ER")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_LVT_box_WS.png", "level", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_LVT_box_WS.png", "level", y_min, y_max, "WS")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_LVT_box_BA.png", "level", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_LVT_box_BA.png", "level", y_min, y_max, "BA")
'''
'''
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_pairwise6W", "PAIR_6W_LVT_box_GE.png", "level", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
box_plot_single(folder_name + '/', "log_folder_global6W", "GLOB_6W_LVT_box_GE.png", "level", y_min, y_max, "GE")
'''

def box_plot(experiment_folder, log_folder_exact1, log_folder_exact2, log_folder_qos, log_folder_krus, text, file_name, dependent, y_min, y_max, graph_type, specific_node_size=None):
  global fig_count, my_inf
  path_to_plots_directory = experiment_folder + 'plots/'
  node, level, terminal, number_of_terminals, obj, qos_obj = parse_output_file(log_folder_exact1, log_folder_qos, specific_node_size, graph_type)
  node2, level2, terminal2, number_of_terminals2, obj2, krus_obj = parse_output_file(log_folder_exact2, log_folder_krus, specific_node_size, graph_type)
  #print(obj[:10], qos_obj[:10])
  #print(obj2[:10], krus_obj[:10])
  qos_dict = OrderedDict()
  krus_dict = OrderedDict()
  if dependent=='node':
    dependent_var = node
  elif dependent=='level':
    dependent_var = level
  elif dependent=='terminal':
    dependent_var = terminal
  elif dependent=='number_of_terminals':
    dependent_var = number_of_terminals
  #len_obj = len(obj)
  #for i in range(len_obj):
  for i in range(len(obj)):
    if obj[i]!=-1 and qos_obj[i]!=-1:
      rat = qos_obj[i]/obj[i]
      if dependent_var[i] not in qos_dict.keys():
        qos_dict[dependent_var[i]] = []
      qos_dict[dependent_var[i]].append(rat)
  #print("qos_dict", qos_dict)
  for i in range(len(obj2)):
  # for glob vs glob GE the above line does not work, try the below one
  #for i in range(min(len(obj), len(obj2))):
    if obj2[i]!=-1 and krus_obj[i]!=-1:
      rat = krus_obj[i]/obj2[i]
      if dependent_var[i] not in krus_dict.keys():
        krus_dict[dependent_var[i]] = []
      krus_dict[dependent_var[i]].append(rat)
  #print(krus_dict)
  size = len(qos_dict.keys())
  label_ind = 0
  labels = []
  sorted_keys = []
  for k in qos_dict.keys():
    sorted_keys.append(k)
  sorted_keys.sort()
  for k in sorted_keys:
    if dependent=="terminal":
      if k==0:
        labels.append("Linear")
      else:
        labels.append("Exponential")
      continue
    if (dependent=='node' or dependent=='number_of_terminals') and ((k%2)==1):
      labels.append('')
      continue
    labels.append(k)
  data = []
  gaps = 1
  i = 0
  for k in sorted_keys:
    data.append(sorted(qos_dict[k]))
    data.append(sorted(krus_dict[k]))
    if i<size-1:
      for g in range(gaps):
        # some space
        data.append([])
    i = i + 1

  plt.figure(fig_count)
  fig_count = fig_count + 1
  color = ['red', 'blue']
  #text = ['Global', 'Local']
  bp = plt.boxplot(data, 0, '', whis=1000, patch_artist=True)
  i = 0
  for box in bp['boxes']:
    # change outline color
    # check whether it is a gap, if gap no need to color
    c_i = i%(len(text)+gaps)
    if c_i<len(text):
      box.set(color=color[c_i], linewidth=2)
    i = i + 1
  handles = []
  for i in range(len(text)):
    patch = mpatches.Patch(color=color[i], label=text[i])
    handles.append(patch)
  plt.legend(handles=handles)
  label_i = 1
  # labels computation is complex, it has add 2 for gaps, negate 2 for boundary condition
  #tmp = range(1,size*(len(text)+gaps)+1-2)
  tmp = range(1,size*(len(text)+gaps)+1-1)
  tmp2 = []
  for i in range(len(tmp)):
    # Add 2 with len(text) because we want two gaps between groups of boxes
    if i%(len(text)+gaps)==label_i:
      tmp2.append(labels[label_ind])
      label_ind = label_ind + 1
    else:
      tmp2.append('')
  plt.xticks(tmp, tmp2)
  if dependent=='node':
    plt.xlabel('Number of vertices', fontsize=20)
  elif dependent=='level':
    plt.xlabel('Number of levels', fontsize=20)
  elif dependent=='terminal':
    plt.xlabel('Terminal selection method', fontsize=20)
  elif dependent=='number_of_terminals':
    plt.xlabel('Number of terminals', fontsize=20)
  plt.ylabel('Ratio', fontsize=20)
  plt.ylim(y_min, y_max)
  plt.tick_params(axis='x', labelsize=16)
  plt.tick_params(axis='y', labelsize=16)
  plt.show()
  plt.savefig(path_to_plots_directory+file_name, bbox_inches='tight')
  plt.close()

'''
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_NVR_box.png", "node", y_min, y_max, "ER")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_LVR_box.png", "level", y_min, y_max, "ER")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_TVR_box.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_NVR_box_WS.png", "node", y_min, y_max, "WS")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_LVR_box_WS.png", "level", y_min, y_max, "WS")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_TVR_box_WS.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_NVR_box_BA.png", "node", y_min, y_max, "BA")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_LVR_box_BA.png", "level", y_min, y_max, "BA")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_TVR_box_BA.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_NVR_box_GE.png", "node", y_min, y_max, "GE")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_LVR_box_GE.png", "level", y_min, y_max, "GE")
box_plot(folder_name + '/', "log_folder", "log_folder_local", "log_folder_subset", "log_folder_pairwise2W", ['Global', 'Local'], "GLOB_LOC_TVR_box_GE.png", "terminal", y_min, y_max, "GE")
'''
'''
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_NVR_box.png", "node", y_min, y_max, "ER")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_LVR_box.png", "level", y_min, y_max, "ER")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_TVR_box.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_NVR_box_WS.png", "node", y_min, y_max, "WS")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_LVR_box_WS.png", "level", y_min, y_max, "WS")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_TVR_box_WS.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_NVR_box_BA.png", "node", y_min, y_max, "BA")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_LVR_box_BA.png", "level", y_min, y_max, "BA")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_TVR_box_BA.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_NVR_box_GE.png", "node", y_min, y_max, "GE")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_LVR_box_GE.png", "level", y_min, y_max, "GE")
box_plot(folder_name + '/', "log_folder_local", "log_folder_local4W", "log_folder_pairwise2W", "log_folder_pairwise4W", ['2W(s, t)', '4W(s, t)'], "LOC_LOC_TVR_box_GE.png", "terminal", y_min, y_max, "GE")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_NVR_box.png", "node", y_min, y_max, "ER")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_LVR_box.png", "level", y_min, y_max, "ER")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_TVR_box.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_NVR_box_WS.png", "node", y_min, y_max, "WS")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_LVR_box_WS.png", "level", y_min, y_max, "WS")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_TVR_box_WS.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_NVR_box_BA.png", "node", y_min, y_max, "BA")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_LVR_box_BA.png", "level", y_min, y_max, "BA")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_TVR_box_BA.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_NVR_box_GE.png", "node", y_min, y_max, "GE")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_LVR_box_GE.png", "level", y_min, y_max, "GE")
box_plot(folder_name + '/', "log_folder", "log_folder_global6W", "log_folder_subset", "log_folder_pairwise6W", ['2W', '6W'], "GLOB_GLOB_TVR_box_GE.png", "terminal", y_min, y_max, "GE")
'''

def box_plot_all(experiment_folder, log_folder_exact1, log_folder_exact2, log_folder_exact3, log_folder_exact4, text, file_name, dependent, y_min, y_max, graph_type, specific_node_size=None):
  global fig_count, my_inf
  path_to_plots_directory = experiment_folder + 'plots/'
  node_global2W, level_global2W, terminal_global2W, number_of_terminals_global2W, qos_obj = parse_output_file_for_time(log_folder_exact1, specific_node_size, graph_type)
  node_local2W, level_local2W, terminal_local2W, number_of_terminals_local2W, krus_obj = parse_output_file_for_time(log_folder_exact2, specific_node_size, graph_type)
  node_local4W, level_local4W, terminal_local4W, number_of_terminals_local4W, pair4W_obj = parse_output_file_for_time(log_folder_exact3, specific_node_size, graph_type)
  node_global6W, level_global6W, terminal_global6W, number_of_terminals_global6W, pair6W_obj = parse_output_file_for_time(log_folder_exact4, specific_node_size, graph_type)
  #print(obj[:10], qos_obj[:10])
  #print(obj2[:10], krus_obj[:10])
  qos_dict = OrderedDict()
  krus_dict = OrderedDict()
  pair4W_dict = OrderedDict()
  pair6W_dict = OrderedDict()
  if dependent=='node':
    dependent_var_global2W = node_global2W
    dependent_var_local2W = node_local2W
    dependent_var_local4W = node_local4W
    dependent_var_global6W = node_global6W
  elif dependent=='level':
    dependent_var_global2W = level_global2W
    dependent_var_local2W = level_local2W
    dependent_var_local4W = level_local4W
    dependent_var_global6W = level_global6W
  elif dependent=='terminal':
    dependent_var_global2W = terminal_global2W
    dependent_var_local2W = terminal_local2W
    dependent_var_local4W = terminal_local4W
    dependent_var_global6W = terminal_global6W
  elif dependent=='number_of_terminals':
    dependent_var = number_of_terminals
  #len_obj = len(obj)
  #for i in range(len_obj):
  for i in range(len(qos_obj)):
    if qos_obj[i]!=-1:
      rat = qos_obj[i]
      if dependent_var_global2W[i] not in qos_dict.keys():
        qos_dict[dependent_var_global2W[i]] = []
      qos_dict[dependent_var_global2W[i]].append(rat)
  #print("qos_dict", qos_dict)
  for i in range(len(krus_obj)):
  # for glob vs glob GE the above line does not work, try the below one
  #for i in range(min(len(obj), len(obj2))):
    if krus_obj[i]!=-1:
      rat = krus_obj[i]
      if dependent_var_local2W[i] not in krus_dict.keys():
        krus_dict[dependent_var_local2W[i]] = []
      krus_dict[dependent_var_local2W[i]].append(rat)
  #print(krus_dict)
  for i in range(len(pair4W_obj)):
    if pair4W_obj[i]!=-1:
      rat = pair4W_obj[i]
      if dependent_var_local4W[i] not in pair4W_dict.keys():
        pair4W_dict[dependent_var_local4W[i]] = []
      pair4W_dict[dependent_var_local4W[i]].append(rat)
  for i in range(len(pair6W_obj)):
    if pair6W_obj[i]!=-1:
      rat = pair6W_obj[i]
      if dependent_var_global6W[i] not in pair6W_dict.keys():
        pair6W_dict[dependent_var_global6W[i]] = []
      pair6W_dict[dependent_var_global6W[i]].append(rat)
  size = len(qos_dict.keys())
  label_ind = 0
  labels = []
  sorted_keys = []
  for k in qos_dict.keys():
    sorted_keys.append(k)
  sorted_keys.sort()
  for k in sorted_keys:
    if dependent=="terminal":
      if k==0:
        labels.append("Linear")
      else:
        labels.append("Exponential")
      continue
    if (dependent=='node' or dependent=='number_of_terminals') and ((k%4)==1):
      labels.append('')
      continue
    labels.append(k)
  data = []
  gaps = 1
  i = 0
  for k in sorted_keys:
    data.append(sorted(qos_dict[k]))
    data.append(sorted(krus_dict[k]))
    data.append(sorted(pair4W_dict[k]))
    data.append(sorted(pair6W_dict[k]))
    if i<size-1:
      for g in range(gaps):
        # some space
        data.append([])
    i = i + 1

  plt.figure(fig_count)
  fig_count = fig_count + 1
  color = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
  #text = ['Global', 'Local']
  bp = plt.boxplot(data, 0, '', whis=1000, patch_artist=True)
  i = 0
  for box in bp['boxes']:
    # change outline color
    # check whether it is a gap, if gap no need to color
    c_i = i%(len(text)+gaps)
    if c_i<len(text):
      box.set(color=color[c_i], linewidth=2)
    i = i + 1
  handles = []
  for i in range(len(text)):
    patch = mpatches.Patch(color=color[i], label=text[i])
    handles.append(patch)
  plt.legend(handles=handles)
  label_i = 1
  # labels computation is complex, it has add 2 for gaps, negate 2 for boundary condition
  #tmp = range(1,size*(len(text)+gaps)+1-2)
  tmp = range(1,size*(len(text)+gaps)+1-1)
  tmp2 = []
  for i in range(len(tmp)):
    # Add 2 with len(text) because we want two gaps between groups of boxes
    if i%(len(text)+gaps)==label_i:
      tmp2.append(labels[label_ind])
      label_ind = label_ind + 1
    else:
      tmp2.append('')
  plt.xticks(tmp, tmp2)
  if dependent=='node':
    plt.xlabel('Number of vertices', fontsize=20)
  elif dependent=='level':
    plt.xlabel('Number of levels', fontsize=20)
  elif dependent=='terminal':
    plt.xlabel('Terminal selection method', fontsize=20)
  elif dependent=='number_of_terminals':
    plt.xlabel('Number of terminals', fontsize=20)
  plt.ylabel('Time (seconds)', fontsize=20)
  #plt.ylim(y_min, y_max)
  plt.tick_params(axis='x', labelsize=16)
  plt.tick_params(axis='y', labelsize=16)
  plt.show()
  plt.savefig(path_to_plots_directory+file_name, bbox_inches='tight')
  plt.close()

'''
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_ER.png", "level", y_min, y_max, "ER")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
'''
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_ER_14.png", "level", y_min, y_max, "ER", [14, 16])
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_ER_14.png", "terminal", y_min, y_max, "ER", [14, 16])
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_ER_24.png", "level", y_min, y_max, "ER", [24, 26])
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_ER_24.png", "terminal", y_min, y_max, "ER", [24, 26])
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_ER_34.png", "level", y_min, y_max, "ER", [34, 36])
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_ER_34.png", "terminal", y_min, y_max, "ER", [34, 36])
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_ER.png", "node", y_min, y_max, "ER")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_ER.png", "level", y_min, y_max, "ER")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_ER.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_WS.png", "level", y_min, y_max, "WS")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_WS.png", "node", y_min, y_max, "WS")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_WS.png", "level", y_min, y_max, "WS")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_WS.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_BA.png", "level", y_min, y_max, "BA")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_BA.png", "node", y_min, y_max, "BA")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_BA.png", "level", y_min, y_max, "BA")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_BA.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_LVT_box_GE.png", "level", y_min, y_max, "GE")
box_plot_all(folder_name + '/', "log_folder", "log_folder_local", "log_folder_local4W", "log_folder_global6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "ALL_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
'''
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_GE.png", "node", y_min, y_max, "GE")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_GE.png", "level", y_min, y_max, "GE")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_GE.png", "terminal", y_min, y_max, "GE")
'''

'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_WS_large.png", "node", y_min, y_max, "WS")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_WS_large.png", "level", y_min, y_max, "WS")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_WS_large.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_ER_large.png", "node", y_min, y_max, "ER")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_ER_large.png", "level", y_min, y_max, "ER")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_ER_large.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_BA_large.png", "node", y_min, y_max, "BA")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_BA_large.png", "level", y_min, y_max, "BA")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_BA_large.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVT_box_GE_large.png", "node", y_min, y_max, "GE")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVT_box_GE_large.png", "level", y_min, y_max, "GE")
box_plot_all(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVT_box_GE_large.png", "terminal", y_min, y_max, "GE")
'''

def box_plot_apprx(experiment_folder, log_folder_subset, log_folder_pairwise2W, log_folder_pairwise4W, log_folder_pairwise6W, text, file_name, dependent, y_min, y_max, graph_type, specific_node_size=None):
  global fig_count, my_inf
  path_to_plots_directory = experiment_folder + 'plots/'
  node_global2W, level_global2W, terminal_global2W, number_of_terminals_global2W, qos_obj = parse_output_file_apprx(log_folder_subset, specific_node_size, graph_type)
  node_local2W, level_local2W, terminal_local2W, number_of_terminals_local2W, krus_obj = parse_output_file_apprx(log_folder_pairwise2W, specific_node_size, graph_type)
  node_local4W, level_local4W, terminal_local4W, number_of_terminals_local4W, pair4W_obj = parse_output_file_apprx(log_folder_pairwise4W, specific_node_size, graph_type)
  node_global6W, level_global6W, terminal_global6W, number_of_terminals_global6W, pair6W_obj = parse_output_file_apprx(log_folder_pairwise6W, specific_node_size, graph_type)
  #print(obj[:10], qos_obj[:10])
  #print(obj2[:10], krus_obj[:10])
  qos_dict = OrderedDict()
  krus_dict = OrderedDict()
  pair4W_dict = OrderedDict()
  pair6W_dict = OrderedDict()
  if dependent=='node':
    dependent_var_global2W = node_global2W
    dependent_var_local2W = node_local2W
    dependent_var_local4W = node_local4W
    dependent_var_global6W = node_global6W
  elif dependent=='level':
    dependent_var_global2W = level_global2W
    dependent_var_local2W = level_local2W
    dependent_var_local4W = level_local4W
    dependent_var_global6W = level_global6W
  elif dependent=='terminal':
    dependent_var_global2W = terminal_global2W
    dependent_var_local2W = terminal_local2W
    dependent_var_local4W = terminal_local4W
    dependent_var_global6W = terminal_global6W
  elif dependent=='number_of_terminals':
    dependent_var = number_of_terminals
  #len_obj = len(obj)
  #for i in range(len_obj):
  for i in range(len(qos_obj)):
    if qos_obj[i]!=-1:
      if qos_obj[i]==my_inf:continue
      rat = qos_obj[i]/min(qos_obj[i], krus_obj[i], pair4W_obj[i], pair6W_obj[i])
      if dependent_var_global2W[i] not in qos_dict.keys():
        qos_dict[dependent_var_global2W[i]] = []
      qos_dict[dependent_var_global2W[i]].append(rat)
  #print("qos_dict", qos_dict)
  for i in range(len(krus_obj)):
  # for glob vs glob GE the above line does not work, try the below one
  #for i in range(min(len(obj), len(obj2))):
    if krus_obj[i]!=-1:
      if krus_obj[i]==my_inf:continue
      rat = krus_obj[i]/min(qos_obj[i], krus_obj[i], pair4W_obj[i], pair6W_obj[i])
      if dependent_var_local2W[i] not in krus_dict.keys():
        krus_dict[dependent_var_local2W[i]] = []
      krus_dict[dependent_var_local2W[i]].append(rat)
  #print(krus_dict)
  for i in range(len(pair4W_obj)):
    if pair4W_obj[i]!=-1:
      if pair4W_obj[i]==my_inf:continue
      rat = pair4W_obj[i]/min(qos_obj[i], krus_obj[i], pair4W_obj[i], pair6W_obj[i])
      if dependent_var_local4W[i] not in pair4W_dict.keys():
        pair4W_dict[dependent_var_local4W[i]] = []
      pair4W_dict[dependent_var_local4W[i]].append(rat)
  for i in range(len(pair6W_obj)):
    if pair6W_obj[i]!=-1:
      if pair6W_obj[i]==my_inf:continue
      rat = pair6W_obj[i]/min(qos_obj[i], krus_obj[i], pair4W_obj[i], pair6W_obj[i])
      if dependent_var_global6W[i] not in pair6W_dict.keys():
        pair6W_dict[dependent_var_global6W[i]] = []
      pair6W_dict[dependent_var_global6W[i]].append(rat)
  size = len(qos_dict.keys())
  label_ind = 0
  labels = []
  sorted_keys = []
  for k in qos_dict.keys():
    sorted_keys.append(k)
  sorted_keys.sort()
  for k in sorted_keys:
    if dependent=="terminal":
      if k==0:
        labels.append("Linear")
      else:
        labels.append("Exponential")
      continue
    if (dependent=='node' or dependent=='number_of_terminals') and ((k%4)==1):
      labels.append('')
      continue
    labels.append(k)
  data = []
  gaps = 1
  i = 0
  for k in sorted_keys:
    data.append(sorted(qos_dict[k]))
    data.append(sorted(krus_dict[k]))
    data.append(sorted(pair4W_dict[k]))
    data.append(sorted(pair6W_dict[k]))
    if i<size-1:
      for g in range(gaps):
        # some space
        data.append([])
    i = i + 1
  plt.figure(fig_count)
  fig_count = fig_count + 1
  color = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
  #text = ['Global', 'Local']
  bp = plt.boxplot(data, 0, '', whis=1000, patch_artist=True)
  i = 0
  for box in bp['boxes']:
    # change outline color
    # check whether it is a gap, if gap no need to color
    c_i = i%(len(text)+gaps)
    if c_i<len(text):
      box.set(color=color[c_i], linewidth=2)
    i = i + 1
  handles = []
  for i in range(len(text)):
    patch = mpatches.Patch(color=color[i], label=text[i])
    handles.append(patch)
  plt.legend(handles=handles)
  label_i = 1
  # labels computation is complex, it has add 2 for gaps, negate 2 for boundary condition
  #tmp = range(1,size*(len(text)+gaps)+1-2)
  tmp = range(1,size*(len(text)+gaps)+1-1)
  tmp2 = []
  for i in range(len(tmp)):
    # Add 2 with len(text) because we want two gaps between groups of boxes
    if i%(len(text)+gaps)==label_i:
      tmp2.append(labels[label_ind])
      label_ind = label_ind + 1
    else:
      tmp2.append('')
  plt.xticks(tmp, tmp2)
  if dependent=='node':
    plt.xlabel('Number of vertices', fontsize=20)
  elif dependent=='level':
    plt.xlabel('Number of levels', fontsize=20)
  elif dependent=='terminal':
    plt.xlabel('Terminal selection method', fontsize=20)
  elif dependent=='number_of_terminals':
    plt.xlabel('Number of terminals', fontsize=20)
  plt.ylabel('Ratio', fontsize=20)
  #plt.ylim(y_min, y_max)
  plt.tick_params(axis='x', labelsize=16)
  plt.tick_params(axis='y', labelsize=16)
  plt.show()
  plt.savefig(path_to_plots_directory+file_name, bbox_inches='tight')
  plt.close()

'''
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVR_box_WS_large.png", "node", y_min, y_max, "WS")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVR_box_WS_large.png", "level", y_min, y_max, "WS")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVR_box_WS_large.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVR_box_ER_large.png", "node", y_min, y_max, "ER")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVR_box_ER_large.png", "level", y_min, y_max, "ER")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVR_box_ER_large.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVR_box_BA_large.png", "node", y_min, y_max, "BA")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVR_box_BA_large.png", "level", y_min, y_max, "BA")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVR_box_BA_large.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_NVR_box_GE_large.png", "node", y_min, y_max, "GE")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_LVR_box_GE_large.png", "level", y_min, y_max, "GE")
box_plot_apprx(folder_name + '/', "log_folder_subset", "log_folder_pairwise2W", "log_folder_pairwise4W", "log_folder_pairwise6W", ['2W', '2W(s,t)', '4W(s,t)', '6W'], "APRX_ALL_TVR_box_GE_large.png", "terminal", y_min, y_max, "GE")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_NVR_box_WS_d.png", "node", y_min, y_max, "WS")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_LVR_box_WS_d.png", "level", y_min, y_max, "WS")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_TVR_box_WS_d.png", "terminal", y_min, y_max, "WS")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_NVR_box_ER_d.png", "node", y_min, y_max, "ER")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_LVR_box_ER_d.png", "level", y_min, y_max, "ER")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_TVR_box_ER_d.png", "terminal", y_min, y_max, "ER")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_NVR_box_BA_d.png", "node", y_min, y_max, "BA")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_LVR_box_BA_d.png", "level", y_min, y_max, "BA")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_TVR_box_BA_d.png", "terminal", y_min, y_max, "BA")
'''
'''
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_NVR_box_GE_d.png", "node", y_min, y_max, "GE")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_LVR_box_GE_d.png", "level", y_min, y_max, "GE")
box_plot_apprx(folder_name + '/', "log_folder_pairwise2W", "log_folder_pairwise2W_d2", "log_folder_pairwise2W_d4", "log_folder_pairwise2W_d8", ['d', 'd/2', 'd/4', 'd/8'], "APRX_ALL_TVR_box_GE_d.png", "terminal", y_min, y_max, "GE")
'''


