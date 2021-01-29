from subprocess import call
import math

def generate():
  #root_folder = "experiment_non_uniform_ER"
  #root_folder = "experiment_non_uniform_ER2"
  #root_folder = "experiment_non_uniform_ER3"
  #root_folder = "experiment_non_uniform_ER_test"
  #root_folder = "experiment_ER4"
  #root_folder = "experiment_WS"
  #root_folder = "experiment_WS2"
  #root_folder = "experiment_BA2"
  #root_folder = "exp_ER_wgt_subset"
  #root_folder = "exp_ER_wgt_subset_2"
  #root_folder = "exp_ER_wgt_subset_3"
  #root_folder = "exp_ER_wgt_subset_4"
  #root_folder = "exp_GE_wgt_subset_5"
  #root_folder = "exp_all_wgt_subset_6"
  root_folder = "exp_all_wgt_large"
  #root_folder = "experiment_2"
  #root_folder = "experiment_4"
  #call(["rm", "-rf", "Graph_generator/"+experiment_name])
  call(["mkdir", root_folder])
  name_of_graph_class = ['WS','ER','BA','GE']
  name_of_graph_class_code = ['0','1','2','3']
  #name_of_graph_class = ['ER']
  #name_of_graph_class = ['WS']
  #name_of_graph_class = ['BA']
  #name_of_graph_class = ['GE']
  #name_of_graph_class_code = ['1']
  #name_of_graph_class_code = ['0']
  #name_of_graph_class_code = ['2']
  #name_of_graph_class_code = ['3']
  #number_of_nodes_progression = [100, 100, 100]
  #number_of_nodes_progression = [80, 80, 80]
  #number_of_nodes_progression = [40, 40, 40]
  number_of_nodes_progression = [500 for i in range(10)]
  #number_of_nodes_progression = [100, 100, 100, 100, 100, 100]
  #number_of_nodes_progression = [300]
  #number_of_nodes_progression = [100]
  #number_of_nodes_progression = [30, 30]
  #number_of_levels = [1, 2, 3]
  number_of_levels = [i for i in range(1, 11)]
  #number_of_levels = [1]
  #number_of_levels = [2, 3, 4, 5, 6, 7]
  #number_of_levels = [2, 3]
  #number_of_nodes_progression = [15, 15, 15]
  #number_of_levels = [1, 2, 3]
  node_distribution_in_levels = ['L','E']
  node_distribution_in_levels_code = ['0','1']
  #node_distribution_in_levels = ['L']
  #node_distribution_in_levels_code = ['0']
  param1 = ['6', '.25', '5', '1.62']
  param2 = ['.2', '0', '0', '0']
  #param1 = ['.25']
  #param2 = ['0']
  #param1 = ['6']
  #param2 = ['.2']
  #param1 = ['5']
  #param2 = ['0']
  #initial_nodes = 10
  initial_nodes = 50
  #node_increment = 5
  #node_increment = 2
  node_increment = 50
  #initial_nodes = 20
  #node_increment = 20
  number_of_graphs_with_same_setting = 5
  #number_of_graphs_with_same_setting = 10
  curr_id = 1
  f1 = open(root_folder + '/id_to_file.csv', 'w')
  f2 = open(root_folder + '/id_to_file_subset.csv', 'w')
  f3 = open(root_folder + '/id_to_file_local.csv', 'w')
  f4 = open(root_folder + '/id_to_file_pairwise2W.csv', 'w')
  f5 = open(root_folder + '/id_to_file_local4W.csv', 'w')
  f6 = open(root_folder + '/id_to_file_pairwise4W.csv', 'w')
  f7 = open(root_folder + '/id_to_file_global6W.csv', 'w')
  f8 = open(root_folder + '/id_to_file_pairwise6W.csv', 'w')
  f9 = open(root_folder + '/id_to_file_pairwise2W_d2.csv', 'w')
  f10 = open(root_folder + '/id_to_file_pairwise2W_d4.csv', 'w')
  f11 = open(root_folder + '/id_to_file_pairwise2W_d8.csv', 'w')
  #f2 = open(root_folder + '/id_to_file_qos.csv', 'w')
  #f3 = open(root_folder + '/id_to_file_cmp.csv', 'w')
  #f4 = open(root_folder + '/id_to_file_CMP_exact.csv', 'w')
  #f5 = open(root_folder + '/id_to_file_kruskal.csv', 'w')
  for cl in range(len(name_of_graph_class)):
   for l in range(len(number_of_levels)):
    for nd in range(len(node_distribution_in_levels)):
      common_part_of_name = name_of_graph_class[cl]+'_'+str(number_of_nodes_progression[l])+'_'+str(number_of_levels[l])+'_'+node_distribution_in_levels[nd]
      for p in range(initial_nodes, number_of_nodes_progression[l]+1, node_increment):
       for i in range(number_of_graphs_with_same_setting):
        if name_of_graph_class[cl] == 'ER':
         param1[cl] = str((1+1)*math.log(p)/p)
        elif name_of_graph_class[cl] == 'GE':
         param1[cl] = str(math.sqrt((1+1)*math.log(p)/(math.pi*p)))
        #call(["python3", "graph_non_uniform_generator.py", str(number_of_levels[l]), str(p), root_folder + "/graph_" + common_part_of_name + '_' + str(p) + '_' + str(i), name_of_graph_class_code[cl], param1[cl], param2[cl], node_distribution_in_levels_code[nd]])
        # Uncomment the following, not above one
        #call(["python3", "graph_generator_v2.py", str(number_of_levels[l]), str(p), root_folder + "/graph_" + common_part_of_name + '_' + str(p) + '_' + str(i), name_of_graph_class_code[cl], param1[cl], param2[cl], node_distribution_in_levels_code[nd]])
        #f1.write(str(curr_id) + ';' + 'mlst_exact.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_exact.csv' + ';\n')
        f1.write(str(curr_id) + ';' + 'additive_spanner_exact_algorithm.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '2' + ';\n')
        #f2.write(str(curr_id) + ';' + 'mlst_charikar.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_qos.csv' + ';\n')
        #f2.write(str(curr_id) + ';' + 'mlst_qos.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_qos.csv' + ';\n')
        #f2.write(str(curr_id) + ';' + 'subsetwise.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_subset.csv' + ';\n')
        f2.write(str(curr_id) + ';' + 'subsetwise_2W.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_subset.csv' + ';\n')
        f3.write(str(curr_id) + ';' + 'additive_spanner_exact_algorithm_local.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '2' + ';\n')
        f4.write(str(curr_id) + ';' + 'pairwise.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '2' + ';\n')
        f5.write(str(curr_id) + ';' + 'additive_spanner_exact_algorithm_local.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '4' + ';\n')
        f6.write(str(curr_id) + ';' + 'pairwise.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '4' + ';\n')
        f7.write(str(curr_id) + ';' + 'additive_spanner_exact_algorithm.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '6' + ';\n')
        f8.write(str(curr_id) + ';' + 'pairwise.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '6' + ';\n')
        f9.write(str(curr_id) + ';' + 'pairwise_d.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '2' + ';' + '2' + ';\n')
        f10.write(str(curr_id) + ';' + 'pairwise_d.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '2' + ';' + '4' + ';\n')
        f11.write(str(curr_id) + ';' + 'pairwise_d.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + '2' + ';' + '8' + ';\n')
        #f3.write(str(curr_id) + ';' + 'mlst_cmp.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_cmp.csv' + ';\n')
        #f5.write(str(curr_id) + ';' + 'mlst_kruskal_non_uniform2.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_kruskal.csv' + ';\n')
        #f5.write(str(curr_id) + ';' + 'mlst_kruskal_non_uniform_update_cost.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_kruskal.csv' + ';\n')
        #f5.write(str(curr_id) + ';' + 'mlst_cmp.py' + ';' + root_folder + ';' + "graph_" + common_part_of_name + '_' + str(p) + '_' + str(i) + ';' + 'output_kruskal.csv' + ';\n')
        curr_id = curr_id + 1
  f1.close()
  f2.close()
  f3.close()
  f4.close()
  f5.close()
  f6.close()
  f7.close()
  f8.close()
  f9.close()
  f10.close()
  f11.close()

generate()

