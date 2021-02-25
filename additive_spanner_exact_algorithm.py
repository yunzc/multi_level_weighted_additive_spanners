import sys
sys.path.append('/cm/shared/uaapps/cplex/12.7.1/cplex/python/3.5/x86-64_linux/')
import cplex
from cplex.exceptions import CplexError
import time
import math
from input_functions import *
from collections import defaultdict
import cspp
from bellman_ford import *

#if len(sys.argv)<5:
#	print('Usage:python3 spanner_exact_algorithm.py file_name_pattern number_of_files score_file_name.js read_subset_from_file/generate_subset')
#	quit()

def all_pairs_from_subset(s):
        all_pairs = []
        for i in range(len(s)):
                for j in range(i+1, len(s)):
                        p = []
                        p.append(s[i])
                        p.append(s[j])
                        all_pairs.append(p)
        return all_pairs

def path_weighted_distance(G, pth):
        result = 0
        for i in range(len(pth)-1):
                result = result + G.get_edge_data(pth[i], pth[i+1])['weight']
        return result

def path_is_not_in_the_list(list_of_paths, pth):
        for pth2 in list_of_paths:
                if len(pth2)!=len(pth):continue
                mismatch = False
                for i in range(len(pth)):
                        if pth2[i]!=pth[i]:
                                mismatch = True
                if not mismatch:
                        return False
        return True


def verify_spanner(G_S, G, all_pairs, additive_stretch):
	for i in range(len(all_pairs)):
		if nx.dijkstra_path_length(G_S,all_pairs[i][0], all_pairs[i][1]) > nx.dijkstra_path_length(G,all_pairs[i][0], all_pairs[i][1]) + additive_stretch:
			print('Shortest path in G')
			print(nx.dijkstra_path(G,all_pairs[i][0], all_pairs[i][1]))
			print('Shortest path in G_S')
			print(nx.dijkstra_path(G_S,all_pairs[i][0], all_pairs[i][1]))
			return False
	return True

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

def find_paths_between_two_vertices(list_of_paths, u, v):
        result = []
        for i in range(len(list_of_paths)):
                if ((list_of_paths[i][0]==u) and (list_of_paths[i][len(list_of_paths[i])-1]==v)) or ((list_of_paths[i][0]==v) and (list_of_paths[i][len(list_of_paths[i])-1]==u)):
                        result.append(i)
        return result

def find_paths_that_contains_two_vertices(list_of_paths, u, v):
	result = []
	for i in range(len(list_of_paths)):
		contains, ui, vi = path_contains_pair(list_of_paths[i], u, v)
		if contains:
			result.append(i)
	return result

def find_paths_that_contains_edge(list_of_paths, path_indices, eu, ev):
        result = []
        for i in path_indices:
                for j in range(len(list_of_paths[i])-1):
                        if ((list_of_paths[i][j]==eu) and (list_of_paths[i][j+1]==ev)) or ((list_of_paths[i][j]==ev) and (list_of_paths[i][j+1]==eu)):
                                result.append(i)
                                break
        return result

def get_existing_path(pths, u, v, G_S, G, additive_stretch):
	for i in range(len(pths)):
		pth = pths[i]
		u_i = -1
		v_i = -1
		for j in range(len(pth)):
			if pth[j] == u:
				u_i = j
			elif pth[j] == v:
				v_i = j
		if (u_i!=-1) and (v_i!=-1):
			if u_i > v_i:
				tmp = u_i
				u_i = v_i
				v_i = tmp
			sum_x = 0
			#existing_pth = [pth[u_i]]
			for j in range(u_i, v_i):
				sum_x = sum_x + G_S.get_edge_data(pth[j], pth[j+1])['weight']
				#existing_pth.append(pth[j+1])
			if sum_x <= nx.dijkstra_path_length(G, u, v) + additive_stretch:
				return i
	return -1

import networkx as nx

def get_path_weight(G, pth, ui, vi):
	tmp = 0
	for j in range(ui, vi):
		tmp = tmp + G.get_edge_data(pth[j], pth[j+1])['weight']
	return tmp

def path_contains_pair(pth, u, v):
	u_i = -1
	v_i = -1
	for j in range(len(pth)):
		if pth[j] == u:
			u_i = j
		elif pth[j] == v:
			v_i = j
	if (u_i!=-1) and (v_i!=-1):
		if u_i > v_i:
			tmp = u_i
			u_i = v_i
			v_i = tmp
		return True, u_i, v_i
	return False, u_i, v_i

def get_related_paths(G, all_pairs, init_path, check_stretch, param):
	# Instead of computing from the beginning, we should just add path
	# Because, there are exponential paths

	pair_to_paths = []
	for i in range(len(all_pairs)):
		pair_to_paths.append([])
	for i in range(len(all_pairs)):
		for j in range(len(init_path)):
			contains, ui, vi = path_contains_pair(init_path[j], all_pairs[i][0], all_pairs[i][1])
			if contains:
				#if all_pairs[i]==[0,1]:
				#	print('path:')
				#	print(init_path[j])
				#	print(str(get_path_weight(G, init_path[j], ui, vi)))
				#	print(nx.shortest_path(G, source=all_pairs[i][0], target=all_pairs[i][1]))
				#	print(nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight'))
				sp = nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight')
				if check_stretch(get_path_weight(G, init_path[j], ui, vi), sp, param):
					pair_to_paths[i].append(j)
	return pair_to_paths

def get_related_paths_exact(G, all_pairs, init_path, check_stretch, param):
        # Instead of computing from the beginning, we should just add path
        # Because, there are exponential paths

        pair_to_paths = []
        for i in range(len(all_pairs)):
                pair_to_paths.append([])
        for i in range(len(all_pairs)):
                for j in range(len(init_path)):
                        if (init_path[j][0]==all_pairs[i][0] and init_path[j][len(init_path[j])-1]==all_pairs[i][1]) or (init_path[j][0]==all_pairs[i][1] and init_path[j][len(init_path[j])-1]==all_pairs[i][0]):
                                #if all_pairs[i]==[0,1]:
                                #        print('path:')
                                #        print(init_path[j])
                                #        print(str(get_path_weight(G, init_path[j], ui, vi)))
                                #        print(nx.shortest_path(G, source=all_pairs[i][0], target=all_pairs[i][1]))
                                #        print(nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight'))
                                sp = nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight')
                                if check_stretch(get_path_weight(G, init_path[j], 0, len(init_path[j])-1), sp, param):
                                        pair_to_paths[i].append(j)
        return pair_to_paths

def primal_program(G, all_pairs_arr, init_path_arr, ilp_or_lp, check_stretch, stretch_param):
	global log_file_name
	m = len(G.edges())
	init_prim_obj = list()
	init_prim_sense = ""
	init_prim_rownames = list()
	init_prim_total_rows = 0
	init_prim_colnames = list()
	init_prim_total_columns = 0
	init_prim_ub = list()
	init_prim_lb = list()
	get_column = dict()
	my_ctype = list()

	rows = []
	cols = []
	vals = []

	init_prim_constraint_values = []

	for l in range(len(all_pairs_arr)):
		all_pairs = all_pairs_arr[l]
		init_path = init_path_arr[l]
		#What are the variables in initial primal lp?
        	#One veriable for each edge, xe, and one variable for each path, yp
        	#The objective function is simple, for xe's, the weight of e is the coefficient, for yp's, coefficient is zero
		for (u,v,d) in G.edges(data='weight'):
			init_prim_obj.append(d)
			var_name = 'x_'+str(u)+'_'+str(v)+'_'+str(l)
			init_prim_colnames.append(var_name)
			get_column[var_name] = init_prim_total_columns
			init_prim_total_columns = init_prim_total_columns + 1
		for i in range(len(init_path)):
			init_prim_obj.append(0)
			var_name = 'y_'+str(init_path[i][0])+'_'+str(init_path[i][len(init_path[i])-1])+'_'+str(i)+'_'+str(l)
			#print('var_name:',var_name)
			init_prim_colnames.append(var_name)
			get_column[var_name] = init_prim_total_columns
			init_prim_total_columns = init_prim_total_columns + 1



		#pair_to_paths = get_related_paths(G, all_pairs, init_path, check_stretch, stretch_param)
		pair_to_paths = get_related_paths_exact(G, all_pairs, init_path, check_stretch, stretch_param)
		#print('pair_to_paths:',pair_to_paths)

		#init_prim_constraint_coefficients = []
		#init_prim_constraint_values = []
		#there are four kinds of columns, last two are just bounds
		#We now consider the first constraints
		#Number of such constraints is equal to number of pairs times number of edges
		for i in range(len(all_pairs)):
			edge_ind = 0
			for (u,v,d) in G.edges(data='weight'):
				#find the paths from ith pair, here we get only one path
				#path_indices = find_paths_between_two_vertices(init_path, all_pairs[i][0], all_pairs[i][1])
				path_indices = pair_to_paths[i]
				path_indices = find_paths_that_contains_edge(init_path, path_indices, u, v)
				if (len(path_indices)==0):
					edge_ind = edge_ind + 1
					continue
				init_prim_rownames.append('pi_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1])+'_e_'+str(u)+'_'+str(v)+'_'+str(l))
				#init_prim_constraint_coefficients.append([])
				init_prim_constraint_values.append(0)
				init_prim_sense = init_prim_sense + 'G'
				j=0
				#for j in range(m):
				for (u,v,d) in G.edges(data='weight'):
					if j==edge_ind:
						#init_prim_constraint_coefficients[init_prim_total_rows].append(1)
						rows.append(init_prim_total_rows)
						cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(l)])
						vals.append(1)
					#else:
					#	init_prim_constraint_coefficients[init_prim_total_rows].append(0)
					j = j+1
				for j in range(len(init_path)):
					if j in path_indices:
						#init_prim_constraint_coefficients[init_prim_total_rows].append(-1)
						rows.append(init_prim_total_rows)
						cols.append(get_column['y_'+str(init_path[j][0])+'_'+str(init_path[j][len(init_path[j])-1])+'_'+str(j)+'_'+str(l)])
						vals.append(-1)
					#else:
					#	init_prim_constraint_coefficients[init_prim_total_rows].append(0)
				init_prim_total_rows = init_prim_total_rows + 1
				edge_ind = edge_ind + 1

		#We now consider second type of constraints
		for i in range(len(all_pairs)):
			#path_indices = find_paths_that_contains_two_vertices(init_path, all_pairs[i][0], all_pairs[i][1])
			init_prim_rownames.append('sigma_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1])+'_'+str(l))
			#init_prim_constraint_coefficients.append([])
			init_prim_constraint_values.append(1)
			init_prim_sense = init_prim_sense + 'G'
			#init_prim_sense = init_prim_sense + 'E'
			#for j in range(m):
			#	init_prim_constraint_coefficients[init_prim_total_rows].append(0)
			#if 'sigma_0_1_0'=='sigma_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1])+'_'+str(l):
			#	print(all_pairs[i], pair_to_paths[i], init_prim_total_rows, get_column['y_'+str(init_path[j][0])+'_'+str(init_path[j][len(init_path[j])-1])+'_'+str(j)+'_'+str(l)])
			for j in range(len(init_path)):
				if j in pair_to_paths[i]:
					#if 'sigma_4_5_1'=='sigma_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1])+'_'+str(l):
					#	print('*******************')
					#init_prim_constraint_coefficients[init_prim_total_rows].append(1)
					rows.append(init_prim_total_rows)
					cols.append(get_column['y_'+str(init_path[j][0])+'_'+str(init_path[j][len(init_path[j])-1])+'_'+str(j)+'_'+str(l)])
					vals.append(1)
				#else:
				#	init_prim_constraint_coefficients[init_prim_total_rows].append(0)
			init_prim_total_rows = init_prim_total_rows + 1

		if l>0:
			for (u,v,d) in G.edges(data='weight'):
                        	init_prim_constraint_values.append(0)
                        	init_prim_rownames.append("r_"+str(init_prim_total_rows))
                        	init_prim_sense = init_prim_sense + "L"
                        	rows.append(init_prim_total_rows)
                        	cols.append(get_column['x_'+str(u)+'_'+str(v)+"_"+str(l-1)])
                        	vals.append(1)
                        	rows.append(init_prim_total_rows)
                        	cols.append(get_column['x_'+str(u)+'_'+str(v)+"_"+str(l)])
                        	vals.append(-1)
                        	init_prim_total_rows = init_prim_total_rows + 1


	#for i in range(len(init_prim_constraint_coefficients)):
	#	for j in range(len(init_prim_constraint_coefficients[i])):
	#		if init_prim_constraint_coefficients[i][j]!=0:
	#			rows.append(i)
	#			cols.append(j)
	#			vals.append(init_prim_constraint_coefficients[i][j])

	#the upper and lower bounds are also simple
	for i in range(len(init_prim_obj)):
		#init_prim_ub.append(cplex.infinity)
		init_prim_ub.append(1.0)
		init_prim_lb.append(0.0)

	for i in range(len(init_prim_obj)):
		if ilp_or_lp=='lp':
			my_ctype.append(cplex.Cplex().variables.type.continuous)
		else:
			my_ctype.append(cplex.Cplex().variables.type.integer)

	prob = cplex.Cplex()
	prob.objective.set_sense(prob.objective.sense.minimize)
	prob.linear_constraints.add(rhs = init_prim_constraint_values, senses = init_prim_sense, names = init_prim_rownames)
	print(len(init_prim_obj), len(init_prim_ub), len(init_prim_lb), len(my_ctype), len(init_prim_colnames))
	if ilp_or_lp=='lp':
		prob.variables.add(obj = init_prim_obj, ub = init_prim_ub, lb = init_prim_lb, names = init_prim_colnames)
	else:
		prob.variables.add(obj = init_prim_obj, ub = init_prim_ub, lb = init_prim_lb, types=my_ctype, names = init_prim_colnames)

	name_indices = [i for i in range(len(init_prim_obj))]
	names = prob.variables.get_names(name_indices)

	prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
	prob.write(log_file_name.split('/')[0]+ "/" + "model_" + log_file_name.split('/')[1][10:len(log_file_name.split('/')[1])-4] + ".lp")
	prob.solve()
	#print('Is dual feasible:')
	#print(prob.solution.is_dual_feasible())

	# print coefficient matrix
	#print('Coefficient matrix:')
	#for i in range(len(init_prim_rownames)):
	#	name_pairs = []
	#	for j in range(len(init_prim_colnames)):
	#		name_pairs.append((init_prim_rownames[i], init_prim_colnames[j]))
	#	print(prob.linear_constraints.get_coefficients(name_pairs))

	print("Solution value  = ", prob.solution.get_objective_value())
	numcols = prob.variables.get_num()
	x = prob.solution.get_values()
	if ilp_or_lp=='ilp':
		for j in range(numcols):
			print("Column %s:  Value = %10f" % (names[j], x[j]))
	#return prob.solution.get_objective_value()
	dual_vals = None
	if ilp_or_lp=='lp':
		dual_vals = prob.solution.get_dual_values()
	#print('init_prim_rownames:', init_prim_rownames)
	#print('len(init_prim_rownames)', len(init_prim_rownames))
	#print('dual_vals:', dual_vals)
	#print('len(dual_vals):', len(dual_vals))
	return init_prim_rownames, dual_vals, prob.solution.get_objective_value(), names, x


def find_subset_wise_spanner(input_file, subset_size, subset_argument):
	#input_file = sys.argv[1]
	#graph_property = 'weighted'
	graph_property = 'unweighted'
	file = open(input_file,"r")
	print("File name: "+input_file)
	m = int(file.readline())
	edge_list = list()
	for i in range(m):
	        t_arr1 = []
	        t_arr2 = file.readline().split()
	        t_arr1.append(int(t_arr2[0]))
	        t_arr1.append(int(t_arr2[1]))
	        if graph_property == 'weighted':
	                t_arr1.append(float(t_arr2[2]))
	        elif graph_property == 'unweighted':
	                t_arr1.append(1)
	        edge_list.append(t_arr1)

	n = max(max(u, v) for [u, v, w] in edge_list) # Get size of matrix
	matrix = [[0] * n for i in range(n)]

	for [u, v, w] in edge_list:
	        matrix[u-1][v-1] = matrix[v-1][u-1] = w

	#What are the pairs?
	#Lets assume that we have a subset, 1/5 of the vertices
	subset = []
	#subset_size = int(sys.argv[2])
	if subset_argument == 'generate_subset':
		for i in range(subset_size):
			subset.append(i)
	elif subset_argument == 'read_subset_from_file':
		subset = [(int(i)-1) for i in file.readline().split()]
	#Now find the pairs

	all_pairs = all_pairs_from_subset(subset)
	# some silly changes to check a test case
	if len(all_pairs)>2:
		tmp = all_pairs[1]
		all_pairs[1] = all_pairs[2]
		all_pairs[2] = tmp

	#initial primal
	#take one path for every pair
	G=nx.Graph()
	for i in range(n):
		for j in range(n):
			if matrix[i][j]!=0:
				G.add_weighted_edges_from([(i, j, matrix[i][j])])

	print('edges')
	for (u,v,d) in G.edges(data='weight'):
		print('(%d, %d, %.3f)'%(u,v,d))

	my_infinity = 100000
	shortest_distance = []
	for i in range(n):
		shortest_distance.append([])
		for j in range(n):
			shortest_distance[i].append(my_infinity)

	P = []
	for i in range(n):
		P.append([])
		for j in range(n):
			P[i].append([])

	additive_stretch = 2

	# Add one shortest path for all pair
	#init_path = []
	#for i in range(len(all_pairs)):
	#	init_path.append(nx.dijkstra_path(G,all_pairs[i][0], all_pairs[i][1]))
	#	tmp = nx.dijkstra_path_length(G,all_pairs[i][0], all_pairs[i][1])
	#	shortest_distance[all_pairs[i][0]][all_pairs[i][1]] = tmp
	#	shortest_distance[all_pairs[i][1]][all_pairs[i][0]] = tmp

	# Adding one shortest path for all pair is not efficient, hence first check whether already an enogh short path exist or not?
	# initially we have an empty graph
	G2 = nx.Graph()
	init_path = []
	# for every pair do
	for i in range(len(all_pairs)):
	#	check whether there is an enogh short path for that pair
	#	if there is no such path add the shortest path of the original graph in the current graph
		if (all_pairs[i][0] not in G2.nodes() or all_pairs[i][1] not in G2.nodes()) or (nx.dijkstra_path_length(G2,all_pairs[i][0], all_pairs[i][1]) > nx.dijkstra_path_length(G,all_pairs[i][0], all_pairs[i][1]) + additive_stretch):
			pth = nx.dijkstra_path(G,all_pairs[i][0], all_pairs[i][1])
			init_path.append(pth)
			for j in range(len(pth)-1):
				G2.add_weighted_edges_from([(pth[j], pth[j+1], G.get_edge_data(pth[j], pth[j+1])['weight'])])

	# the following lines added all shortest path in G2, but now we don't need this, so comment out
	#G2 = nx.Graph()
	#edges_in_path = []
	#for pth in init_path:
	#	for i in range(len(pth)-1):
	#		edges_in_path.append((pth[i], pth[i+1], G.get_edge_data(pth[i], pth[i+1])['weight']))
	#G2.add_weighted_edges_from(edges_in_path)


	# Current guess is, this part is fundamentally wrong
	# I am checking all paths, which is not efficient
	# And the overall model is wrong
	# Suppose there are three vertices u, v and w
	# Suppose there is a path from u to v to w
	# this model allows to take two path variable p_u_v and p_u_w
	# and it allows to set p_u_w=1 but p_u_v=0, which is completely inconsistent
	#init_path = []
	#for i in range(len(all_pairs)):
	#	for pth in nx.all_simple_paths(G2, all_pairs[i][0], all_pairs[i][1]):
	#		if path_weighted_distance(G2, pth) <= shortest_distance[pth[0]][pth[len(pth)-1]] + additive_stretch:
	#	        	init_path.append(pth)

	# Take a 2d list, for every pair it contains the list of paths that CONTAINS that pair, and provides an enough short path
	# Now we consider another formulation
	# We put a path in the list, first path may be the shortest path connecting the first pair vertices
	# Now we consider the next pair, we may start a loop here that takes all pairs one by one
	# If there is a path that CONTAINS these two vertices
	#	Add this path to the list of paths of that pair
	# Else
	#	Add the shortest path between that pair
	#	Update the list of short paths for every pair
	init_path = []
	pair_to_paths = []
	for i in range(len(all_pairs)):
		pair_to_paths.append([])
	for i in range(len(all_pairs)):
		pth_i = get_existing_path(init_path, all_pairs[i][0], all_pairs[i][1], G2, G, additive_stretch)
		if pth_i!=-1:
			pair_to_paths[i].append(pth_i)
		else:
			pth = nx.dijkstra_path(G2,all_pairs[i][0], all_pairs[i][1])
			print('New path computed')
			print(pth)
			for j in range(len(all_pairs)):
				pth_i = get_existing_path([pth], all_pairs[j][0], all_pairs[j][1], G2, G, additive_stretch)
				if pth_i!=-1:
					pair_to_paths[j].append(len(init_path))
			init_path.append(pth)

	print('initial paths:')
	print(init_path)

	init_prim_obj = list()
	init_prim_sense = ""
	init_prim_rownames = list()
	init_prim_total_rows = 0
	init_prim_colnames = list()
	init_prim_total_columns = 0
	init_prim_ub = list()
	init_prim_lb = list()

	#What are the variables in initial primal lp?
	#One veriable for each edge, xe, and one variable for each path, yp
	#The objective function is simple, for xe's, the weight of e is the coefficient, for yp's, coefficient is zero
	for (u,v,d) in G.edges(data='weight'):
		init_prim_obj.append(d)
		init_prim_colnames.append('x_'+str(u)+'_'+str(v)+'_'+str(init_prim_total_columns))
		init_prim_total_columns = init_prim_total_columns + 1
	for i in range(len(init_path)):
		init_prim_obj.append(0)
		init_prim_colnames.append('y_'+str(init_path[i][0])+'_'+str(init_path[i][len(init_path[i])-1])+'_'+str(i))
		init_prim_total_columns = init_prim_total_columns + 1

	#the upper and lower bounds are also simple
	for i in range(len(init_prim_obj)):
		#init_prim_ub.append(cplex.infinity)
		init_prim_ub.append(1.0)
		init_prim_lb.append(0.0)


	constraint_coefficients = []
	constraint_values = []
	#there are four kinds of columns, last two are just bounds
	#We now consider the first constraints
	#Number of such constraints is equal to number of pairs times number of edges
	for i in range(len(all_pairs)):
		edge_ind = 0
		for (u,v,d) in G.edges(data='weight'):
			#find the paths from ith pair, here we get only one path
			#path_indices = find_paths_between_two_vertices(init_path, all_pairs[i][0], all_pairs[i][1])
			path_indices = pair_to_paths[i]
			path_indices = find_paths_that_contains_edge(init_path, path_indices, u, v)
			#if (len(path_indices)==0):continue
			init_prim_rownames.append('pi_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1])+'_e_'+str(u)+'_'+str(v))
			constraint_coefficients.append([])
			constraint_values.append(0)
			init_prim_sense = init_prim_sense + 'G'
			for j in range(m):
				if j==edge_ind:
					constraint_coefficients[init_prim_total_rows].append(1)
				else:
					constraint_coefficients[init_prim_total_rows].append(0)
			for j in range(len(init_path)):
				if j in path_indices:
	                                constraint_coefficients[init_prim_total_rows].append(-1)
				else:
					constraint_coefficients[init_prim_total_rows].append(0)
			init_prim_total_rows = init_prim_total_rows + 1
			edge_ind = edge_ind + 1

	#We now consider second type of constraints
	for i in range(len(all_pairs)):
		path_indices = find_paths_between_two_vertices(init_path, all_pairs[i][0], all_pairs[i][1])
		init_prim_rownames.append('sigma_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1]))
		constraint_coefficients.append([])
		constraint_values.append(1)
		init_prim_sense = init_prim_sense + 'G'
		for j in range(m):
			constraint_coefficients[init_prim_total_rows].append(0)
		for j in range(len(init_path)):
			if j in path_indices:
				constraint_coefficients[init_prim_total_rows].append(1)
			else:
				constraint_coefficients[init_prim_total_rows].append(0)
		init_prim_total_rows = init_prim_total_rows + 1

	print('obj')
	print(init_prim_obj)
	print(init_prim_colnames)
	print(init_prim_rownames)
	print('ub')
	print(init_prim_ub)
	print('lb')
	print(init_prim_lb)
	print('constraints')
	print(constraint_coefficients)
	print(constraint_values)
	print(init_prim_sense)

	prim_obj = init_prim_obj
	prim_sense = init_prim_sense
	prim_rownames = init_prim_rownames
	prim_total_rows = init_prim_total_rows
	prim_colnames = init_prim_colnames
	prim_total_columns = init_prim_total_columns
	prim_ub = init_prim_ub
	prim_lb = init_prim_lb
	prim_constraint_coefficients = constraint_coefficients
	prim_constraint_values = constraint_values

	#while dual is not feasible
	while True:
		dual_obj = constraint_values
		dual_rownames = prim_colnames
		dual_total_rows = prim_total_columns
		dual_colnames = prim_rownames
		dual_total_columns = prim_total_rows
		dual_sense = ''
		for i in range(dual_total_rows):
			dual_sense = dual_sense + 'L'
		dual_ub = [cplex.infinity]*dual_total_columns 
		dual_lb = [0.0]*dual_total_columns
		dual_constraint_coefficients = []
		for i in range(len(prim_constraint_coefficients[0])):
			dual_constraint_coefficients.append([])
			for j in range(len(prim_constraint_coefficients)):
				dual_constraint_coefficients[i].append(prim_constraint_coefficients[j][i])
		dual_constraint_values = prim_obj

		#print('Dual:')
		#print('obj')
		#print(dual_obj)
		#print(dual_colnames)
		#print(dual_rownames)
		#print('ub')
		#print(dual_ub)
		#print('lb')
		#print(dual_lb)
		#print('constraints')
		#print(dual_constraint_coefficients)
		#print(dual_constraint_values)
		#print(dual_sense)

		rows = []
		cols = []
		vals = []
		#for i in range(len(dual_constraint_coefficients)):
		#	for j in range(len(dual_constraint_coefficients[i])):
		#		if dual_constraint_coefficients[i][j]!=0:
		#			rows.append(i)
		#			cols.append(j)
		#			vals.append(dual_constraint_coefficients[i][j])

		for i in range(len(prim_constraint_coefficients)):
			for j in range(len(prim_constraint_coefficients[i])):
				if prim_constraint_coefficients[i][j]!=0:
					rows.append(i)
					cols.append(j)
					vals.append(prim_constraint_coefficients[i][j])

		try:
			#prob = cplex.Cplex()
			#prob.objective.set_sense(prob.objective.sense.maximize)
			#prob.linear_constraints.add(rhs = dual_constraint_values, senses = dual_sense, names = dual_rownames)
			#prob.variables.add(obj = dual_obj, ub = dual_ub, lb = dual_lb, names = dual_colnames)

			#name_indices = [i for i in range(len(dual_obj))]
			#names = prob.variables.get_names(name_indices)


			#prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
			#prob.write("model.lp")
			#prob.solve()
			#print('Is dual feasible:')
			#print(prob.solution.is_primal_feasible())
			#numcols = prob.variables.get_num()
			#x = prob.solution.get_values()
			#for j in range(numcols):
			#	print("Column %s:  Value = %10f" % (names[j], x[j]))
			#print(prob.solution.get_objective_value())

			prob = cplex.Cplex()
			prob.objective.set_sense(prob.objective.sense.minimize)
			prob.linear_constraints.add(rhs = prim_constraint_values, senses = prim_sense, names = prim_rownames)
			prob.variables.add(obj = prim_obj, ub = prim_ub, lb = prim_lb, names = prim_colnames)

			name_indices = [i for i in range(len(prim_obj))]
			names = prob.variables.get_names(name_indices)

			prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
			prob.write("model.lp")
			prob.solve()
			print('Is dual feasible:')
			print(prob.solution.is_dual_feasible())

			#dual_values = prob.solution.get_dual_values()
			#print('Dual values:')
			#print(dual_values)

			if prob.solution.is_dual_feasible()==False:
				# Remove some constraints from the dual, at some point the solution may become feasible
				ignore_last_constraints_counter = 0
				while True:
					#build the rows, columns and values arrays
					rows = []
					cols = []
					vals = []
					for i in range(len(dual_constraint_coefficients) - ignore_last_constraints_counter):
						for j in range(len(dual_constraint_coefficients[i])):
							if dual_constraint_coefficients[i][j]!=0:
								rows.append(i)
								cols.append(j)
								vals.append(dual_constraint_coefficients[i][j])

					dual_sense = ''
					for i in range(len(dual_constraint_coefficients) - ignore_last_constraints_counter):
						dual_sense = dual_sense + 'L'

					prob = cplex.Cplex()
					prob.objective.set_sense(prob.objective.sense.maximize)
					prob.linear_constraints.add(rhs = dual_constraint_values[:len(dual_constraint_coefficients) - ignore_last_constraints_counter], senses = dual_sense, names = dual_rownames[:len(dual_constraint_coefficients) - ignore_last_constraints_counter])
					prob.variables.add(obj = dual_obj, ub = dual_ub, lb = dual_lb, names = dual_colnames)

					name_indices = [i for i in range(len(dual_obj))]
					names = prob.variables.get_names(name_indices)

					prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
					prob.write("model.lp")
					prob.solve()
					if prob.solution.is_primal_feasible():
						print('Constraints ignored')
						print(str(ignore_last_constraints_counter))
						print('Dual values for dual:')
						print(prob.solution.get_dual_values())
						break
					#numcols = prob.variables.get_num()
					#x = prob.solution.get_values()
					#for j in range(numcols):
					#       print("Column %s:  Value = %10f" % (names[j], x[j]))
					#print(prob.solution.get_objective_value())

					ignore_last_constraints_counter = ignore_last_constraints_counter + 1

			dual_values = prob.solution.get_dual_values()
			print('Dual values:')
			print(dual_values)
			numcols = prob.variables.get_num()
			x = prob.solution.get_values()
			for j in range(numcols):
				print("Column %s:  Value = %10f" % (names[j], x[j]))

			# check whether the dual values are feasible
			#for i in range(len(dual_constraint_coefficients)):
			#	total_sum = 0
			#	for j in range(len(dual_constraint_coefficients[i])):
			#		total_sum = total_sum + dual_constraint_coefficients[i][j] * dual_values[j]
			#	if total_sum > dual_constraint_values[i]:
			#		print(dual_rownames[i]+' is violated')
			#		quit()

			#the dual is feasible, hence just solve the primal
			rows = []
			cols = []
			vals = []
			for i in range(len(prim_constraint_coefficients)):
				for j in range(len(prim_constraint_coefficients[i])):
					if prim_constraint_coefficients[i][j]!=0:
						rows.append(i)
						cols.append(j)
						vals.append(prim_constraint_coefficients[i][j])


			#names = []
			#x = []
			#try:
			#	prob = cplex.Cplex()
			#	prob.objective.set_sense(prob.objective.sense.minimize)
			#	prob.linear_constraints.add(rhs = prim_constraint_values, senses = prim_sense, names = prim_rownames)
			#	prob.variables.add(obj = prim_obj, ub = prim_ub, lb = prim_lb, names = prim_colnames)

			#	name_indices = [i for i in range(len(prim_obj))]
			#	names = prob.variables.get_names(name_indices)

			#	prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
			#	prob.write("model.lp")
			#	prob.solve()
			#	numcols = prob.variables.get_num()
			#	x = prob.solution.get_values()
			#	#for j in range(numcols):
			#	#	print("Column %s:  Value = %10f" % (names[j], x[j]))
			#	print(prob.solution.get_objective_value())
			#	G_S = nx.Graph()
			#	i_x = 0
			#	spanner_edges = []
			#	for (u,v,d) in G.edges(data='weight'):
			#		if x[i_x] > 0.0000001:
			#			spanner_edges.append((u,v,d))
			#		i_x = i_x + 1
			#	G_S.add_weighted_edges_from(spanner_edges)
			#	#if verify_spanner(G_S, G, all_pairs, additive_stretch)==False:
			#	#	print(G_S.edges())
			#	#	print('is not a valid spanner of')
			#	#	print(G.edges())
			#	#	quit()
			#	#return len(G_S.edges())
			#except CplexError as exc:
			#	print(exc)

                        #Compute both the ILP and LP and verify whether both are same
			my_ctype = ""
			prim_ub = []
			prim_lb = []
			for i in range(len(prim_obj)):
				prim_ub.append(1)
				prim_lb.append(0)
				my_ctype = my_ctype + "I"

			ilp_names = []
			ilp_x = []
			try:
				prob = cplex.Cplex()
				prob.objective.set_sense(prob.objective.sense.minimize)
				prob.linear_constraints.add(rhs = prim_constraint_values, senses = prim_sense, names = prim_rownames)
				prob.variables.add(obj = prim_obj, ub = prim_ub, lb = prim_lb, types=my_ctype, names = prim_colnames)

				name_indices = [i for i in range(len(prim_obj))]
				ilp_names = prob.variables.get_names(name_indices)

				prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
				prob.write("model.lp")
				prob.solve()
				numcols = prob.variables.get_num()
				ilp_x = prob.solution.get_values()
				#for j in range(numcols):
				#	print("Column %s:  Value = %10f" % (ilp_names[j], ilp_x[j]))
				print(prob.solution.get_objective_value())
				G_S = nx.Graph()
				i_x = 0
				spanner_edges = []
				for (u,v,d) in G.edges(data='weight'):
					if ilp_x[i_x] == 1:
						spanner_edges.append((u,v,d))
			#			if x[i_x] <= 0.0000001:
			#				print('ILP and LP are not same')
			#				#quit()
			#		else:
			#			if x[i_x] > 0.0000001:
			#				print('ILP and LP are not same')
			#				#quit()
					i_x = i_x + 1
				G_S.add_weighted_edges_from(spanner_edges)
				if verify_spanner(G_S, G, all_pairs, additive_stretch)==False:
					print(G_S.edges())
					print('is not a valid spanner of')
					print(G.edges())
					quit()
				return len(G_S.edges())
			except CplexError as exc:
				print(exc)

		except CplexError as exc:
			print(exc)
		print('Need more iteration')
		return -1
	#	constraint shortest path
	#	modify primal

#score_file = sys.argv[3]
#file = open(score_file,"w")
#file.write("var scores = [");
#file.close()

#number_of_files = int(sys.argv[2])

##subset_size = 10
#subset_size = 2
#for i in range(number_of_files):
#	score = find_subset_wise_spanner(sys.argv[1]+str(i+1)+'.txt', subset_size, sys.argv[4])
#	file = open(score_file,"a")
#	file.write(str(score))
#	file.close()
#	subset_size = subset_size + 1
#	if i<(number_of_files-1):
#                file = open(score_file,"a")
#                file.write(",")
#                file.close(#)

#file = open(score_file,"a")
#file.write("];\n")
#file.close()

#G = nx.path_graph(5)
#for i,j in G.edges():
#	G.add_weighted_edges_from([(i, j, 1)])
#print(G.edges())

def multiplicative_check(subgraph_distance, actual_distance, multiplicative_stretch):
	if subgraph_distance <= multiplicative_stretch*actual_distance:
		return True
	return False

def naive_ilp(G, subset_arr, stretch_param):
	all_pairs_arr = []
	for subset in subset_arr:
		all_pairs_arr.append( all_pairs_from_subset(subset) )
	all_paths_arr = []
	#all_possible_pairs = all_pairs_from_subset([i for i in range(len(G.nodes()))])
	for j in range(len(all_pairs_arr)):
		all_paths = []
		all_pairs = all_pairs_arr[j]
		for i in range(len(all_pairs)):
			all_paths.extend(nx.all_simple_paths(G, all_pairs[i][0], all_pairs[i][1]))
		all_paths_arr.append(all_paths)
	primal_program(G, all_pairs_arr, all_paths_arr, 'ilp', multiplicative_check, stretch_param)

def brute_force_ilp(G, subset_arr, checker, param):
  #print(G.edges())
  #all_pairs = all_pairs_from_subset(subset)
  l = len(subset_arr)
  all_pairs_arr = []
  for subset in subset_arr:
    all_pairs_arr.append( all_pairs_from_subset(subset) )
  print('all_pairs_arr:',all_pairs_arr)
  min_cost = 100000000000
  edg_list = []
  for e in G.edges():
    edg_list.append(e)
  m = len(edg_list)
  min_MLGS = []
  for i in range(int(math.pow(2,l*m))):
    t = []
    q = i
    for j in range(l*m):
      t = [q%2]+t
      q = math.floor(q/2)
    infeasible = False
    for j in range(l*m):
      if j-m>=0:
        if t[j]==1 and t[j-m]==0:
          infeasible = True
          break
    if infeasible:
      continue
    MLG_S = []
    s = 0
    for k in range(l):
      G_S = nx.Graph()
      for j in range(k*m,(k+1)*m):
        if t[j]==1:
          u,v = edg_list[j%m]
          G_S.add_weighted_edges_from([(u, v, G.get_edge_data(u, v)['weight'])])
      #print(G_S.edges())
      if verify_spanner_with_checker(G_S, G, all_pairs_arr[l-k-1], checker, param):
        for e in G_S.edges():
          s = s + G_S.get_edge_data(e[0], e[1])['weight']
        MLG_S.append(G_S)
        if k==l-1:
          if s<min_cost:
            print(t)
            min_cost = s
            min_MLGS = MLG_S
      else:
        break
  for GS in min_MLGS:
    print(GS.edges())
  return min_cost

#naive_ilp(G, [[0, 1, 2],[0,1,2]], 2)
#print(brute_force_ilp(G, [[0, 1, 2], [0, 1, 2]], multiplicative_check, 2))

def check_ilp_working_or_not():
  global log_file_name
  #folder_name = 'erdos_renyi_sm2'
  folder_name = 'erdos_renyi_one_level_2'
  #graph = ['graph_1']
  #graph = ['graph_20_1', 'graph_40_1', 'graph_80_1']
  graph = ['graph_20_1']
  #for i in range(1,2):
  for i in range(len(graph)):
    #filename = folder_name + '/' + 'graph_'+str(i+1)+'.txt'
    filename = folder_name + '/' + graph[i] + '.txt'
    print(filename)
    G, subset_arr = build_networkx_graph(filename)
    #subset = [i for i in range(int(len(G.nodes())/2))]
    #log_file_name = folder_name + '/' + 'print_log_' + 'graph_'+str(i+1) + '_stretch_' + '2' + '.txt'
    log_file_name = folder_name + '/' + 'print_log_' + graph[i] + '_stretch_' + '2' + '.txt'
    ew = []
    for e in G.edges():
      ew.append([e[0], e[1], G.get_edge_data(e[0], e[1])['weight']])
    print(ew)
    print('subset array:',subset_arr)
    print('cost of naive_ilp:',naive_ilp(G, subset_arr, 2))
    #print('cost of brute force:',brute_force_ilp(G, subset_arr, multiplicative_check, 2))
    print('********************************************')

#check_ilp_working_or_not()

def greedy_spanner(G, r):
  G_S = nx.Graph()
  for a, b, data in sorted(G.edges(data=True), key=lambda x: x[2]['weight']):
    #print('{a} {b} {w}'.format(a=a, b=b, w=data['weight']))
    if not (a in G_S.nodes() and b in G_S.nodes() and nx.has_path(G_S, a, b)):
      G_S.add_weighted_edges_from([(a, b, G.get_edge_data(a, b)['weight'])])
    else:
      sp = nx.shortest_path_length(G_S, a, b, 'weight')
      if r*data['weight'] < sp:
        G_S.add_weighted_edges_from([(a, b, G.get_edge_data(a, b)['weight'])])
  return G_S

def check_greedy(checker, param):
  #folder_name = 'erdos_renyi_sm2'
  folder_name = 'erdos_renyi_one_level'
  graph = ['graph_20_1', 'graph_40_1', 'graph_80_1']
  for i in range(1):
    filename = folder_name + '/' + graph[i]+'.txt'
    print(filename)
    G, subset_arr = build_networkx_graph(filename)
    print('G:', G.edges())
    G_S = greedy_spanner(G, param)
    print('G_S:', G_S.edges())
    ver_arr = []
    for v in G.nodes():
      ver_arr.append(v)
    if verify_spanner_with_checker(G_S, G, all_pairs_from_subset(ver_arr), checker, param):
      s = 0
      for e in G_S.edges():
        s = s + G_S.get_edge_data(e[0], e[1])['weight']
      print('Sum of weights of spanner edges:', s)
    else:
      print('Greedy spanner is not working!')

#check_greedy(multiplicative_check, 2)

def print_edges(G):
    edg_arr = []
    for (u,v,d) in G.edges(data='weight'):
        tmp = []
        tmp.append(u)
        tmp.append(v)
        tmp.append(d)
        edg_arr.append(tmp)
    print(edg_arr)

def prune(G, G_main, subset, checker, param):
  G_S = nx.Graph()
  G_S.add_weighted_edges_from(G.edges(data='weight'))
  for e in G.edges():
    G_S.remove_edge(e[0], e[1])
    if not verify_spanner_with_checker(G_S, G_main, all_pairs_from_subset(subset), checker, param):
      G_S.add_edge(e[0], e[1], weight=G.get_edge_data(e[0], e[1])['weight'])
    #else:
    #  print('Pruned:', e[0], e[1])
  return G_S

def bot_up(G, subset_arr, checker, param):
  G_S = greedy_spanner(G, param)
  l = len(subset_arr)
  MLG_S = []
  for i in range(l):
    k = l-i-1
    #print(subset_arr[k])
    G_S = prune(G_S, G, subset_arr[k], checker, param)
    MLG_S.append(G_S)
  return MLG_S

def exact_graph(G, subset, checker, param):
  print('Exact:')
  all_pairs = all_pairs_from_subset(subset)
  edges = [ [0, 18], [0, 7], [1, 3], [1, 4], [2, 10], [3, 5], [3, 8], [3, 10], [4, 19], [4, 13], [5, 19], [6, 13], [7, 19], [8, 18], [9, 10], [9, 13], [13, 18] ]
  edges = [ [0, 18], [0, 7], [1, 3], [1, 4], [1, 7], [2, 4], [2, 10], [2, 11], [3, 5], [3, 7], [3, 8], [3, 10], [4, 18], [4, 13], [6, 11], [6, 13], [7, 11], [8, 18], [9, 10], [9, 13], [13, 18] ]
  G_exact = nx.Graph()
  for e in edges:
    G_exact.add_edge(e[0], e[1], weight=G.get_edge_data(e[0], e[1])['weight'])
  if not verify_spanner_with_checker(G_exact, G, all_pairs_from_subset(subset), checker, param):
    print('Exact spanner is wrong!')
  s =0
  for e in G_exact.edges():
    s = s + G_exact.get_edge_data(e[0], e[1])['weight']
    print(G_exact.get_edge_data(e[0], e[1])['weight'])
  print('exact value befor pruning:', s)
  G_exact = prune(G_exact, G, subset, checker, param)
  #for p in all_pairs:
  #  print('pair:', p)
  #  print('sp:', nx.shortest_path_length(G, p[0], p[1], 'weight'))
  #  print('sp in spanner:', nx.shortest_path_length(G_exact, p[0], p[1], 'weight'))
  s =0
  for e in G_exact.edges():
    s = s + G_exact.get_edge_data(e[0], e[1])['weight']
    print(G_exact.get_edge_data(e[0], e[1])['weight'])
  print('exact value:', s)

write_output = False

def check_bot_up(checker):
  global write_output
  #folder_name = 'erdos_renyi_sm2'
  #folder_name = 'erdos_renyi_one_level'
  #folder_name = 'erdos_renyi_one_level_2'
  folder_name = 'erdos_renyi_multi_level'
  #stretch_arr = [2, 4, 8]
  #stretch_arr = [4]
  stretch_arr = [2]
  #graph = ['graph_1', 'graph_2', 'graph_3']
  #graph = ['graph_2']
  #graph = ['graph_20_1', 'graph_40_1', 'graph_80_1']
  #graph = ['graph_20_1']
  #graph = ['graph_40_l_2_1']
  #graph = ['graph_100_l_2_1']
  graph = ['graph_100_l_3_1']
  #graph = ['graph_40_l_3_1']
  #graph = ['graph_60_l_2_1']
  #graph = ['graph_20_1_red_wgt']
  if write_output:
    output_file = open(folder_name + '/' + 'bottom_up_output.txt', 'w')
    output_file.write('filename;stretch;objective;\n')
    output_file.close()
  for i in range(len(graph)):
    filename = folder_name + '/' + graph[i] + '.txt'
    #print(filename)
    G, subset_arr = build_networkx_graph(filename)
    #print('G:', G.edges())
    for param in stretch_arr:
      if write_output:
        output_file = open(folder_name + '/' + 'bottom_up_output.txt', 'a')
        output_file.write(filename+';')
        output_file.close()
        #print('Stretch factor: ', param)
        output_file = open(folder_name + '/' + 'bottom_up_output.txt', 'a')
        output_file.write(str(param)+';')
        output_file.close()
      start_bot_up = time.time()
      MLGS = bot_up(G, subset_arr, checker, param)
      l = len(MLGS)
      for k in range(len(subset_arr)):
        subset = subset_arr[l-k-1]
        all_pairs = all_pairs_from_subset(subset)
        #for p in all_pairs:
        #  print('pair:', p)
        #  print('sp:', nx.shortest_path_length(G, p[0], p[1], 'weight'))
        #  print('sp in spanner:', nx.shortest_path_length(MLGS[k], p[0], p[1], 'weight'))
      #exact_graph(G, subset, checker, param)
      print('Time to compute bot up: ', (time.time() - start_bot_up))
      s = 0
      for k in range(len(MLGS)):
        G_S = MLGS[k]
        #print_edges(G_S)
        if not verify_spanner_with_checker(G_S, G, all_pairs_from_subset(subset_arr[l-k-1]), checker, param):
          print('Wrong!!!')
        #print(G_S.edges())
        for e in G_S.edges():
          s = s + G_S.get_edge_data(e[0], e[1])['weight']
          #print(G_S.get_edge_data(e[0], e[1])['weight'])
      print('Objective value: ', s)
      if write_output:
        output_file = open(folder_name + '/' + 'bottom_up_output.txt', 'a')
        output_file.write(str(s)+';\n')
        output_file.close()

#check_bot_up(multiplicative_check)

def top_down(G, subset_arr, checker, param):
  MLG_S = []
  l = len(subset_arr)
  # Traverse the layers top to down
  for i in range(l):
    # copy all the edges from upper level
    G_S = nx.Graph()
    if i>0:
      G_S.add_weighted_edges_from(MLG_S[i-1].edges(data='weight'))
    # cumpute the abstract graph based on the subset
    G_abs = nx.Graph()
    all_pairs = all_pairs_from_subset(subset_arr[i])
    edges_abs=[]
    for p in all_pairs:
      edges_abs.append((p[0], p[1], nx.shortest_path_length(G, p[0], p[1], 'weight')))
    G_abs.add_weighted_edges_from(edges_abs)
    # compute greedy spanner on this graph
    G_abs = greedy_spanner(G_abs, param)
    # add the edges from the greedy spanner
    for u, v in G_abs.edges():
      pth = nx.dijkstra_path(G, u, v)
      for j in range(1, len(pth)):
        G_S.add_edge(pth[j-1], pth[j], weight=G[pth[j-1]][pth[j]]['weight'])
    G_S = prune(G_S, G, subset_arr[i], checker, param)
    MLG_S.append(G_S)
  return MLG_S

def top_down_exact(G, subset_arr, checker, param):
  MLG_S = []
  l = len(subset_arr)
  # Traverse the layers top to down
  for i in range(l):
    # copy all the edges from upper level
    G_S = nx.Graph()
    if i>0:
      G_S.add_weighted_edges_from(MLG_S[i-1].edges(data='weight'))
    # cumpute the abstract graph based on the subset
    #G_abs = nx.Graph()
    #all_pairs = all_pairs_from_subset(subset_arr[i])
    #edges_abs=[]
    #for p in all_pairs:
    #  edges_abs.append((p[0], p[1], nx.shortest_path_length(G, p[0], p[1], 'weight')))
    #G_abs.add_weighted_edges_from(edges_abs)
    # compute greedy spanner on this graph
    #G_abs = greedy_spanner(G_abs, param)
    # add the edges from the greedy spanner
    #for u, v in G_abs.edges():
    #  pth = nx.dijkstra_path(G, u, v)
    #  for j in range(1, len(pth)):
    #    G_S.add_edge(pth[j-1], pth[j], weight=G[pth[j-1]][pth[j]]['weight'])
    #G_S = prune(G_S, G, subset_arr[i], checker, param)
    names, x = flow_ilp(G, [subset_arr[i]], checker, param)
    edges = []
    for j in range(len(names)):
      parts = names[j].split('_')
      if len(parts)==4:
        if (abs(x[j]-1)<.001):
          u = int(parts[1])
          v = int(parts[2])
          edges.append((u, v, G[u][v]['weight']))
    G_S.add_weighted_edges_from(edges)
    MLG_S.append(G_S)
  return MLG_S


def check_top_down(checker):
  global log_file_name
  #folder_name = 'erdos_renyi_sm2'
  #folder_name = 'erdos_renyi_one_level'
  #folder_name = 'erdos_renyi_one_level_2'
  folder_name = 'erdos_renyi_multi_level'
  #stretch_arr = [2, 4, 8]
  #stretch_arr = [4]
  stretch_arr = [2]
  #graph = ['graph_1', 'graph_2', 'graph_3']
  #graph = ['graph_3']
  #graph = ['graph_100_l_2_1']
  graph = ['graph_100_l_3_1']
  #graph = ['graph_20_1', 'graph_40_1', 'graph_80_1']
  #graph = ['graph_20_1']
  #graph = ['graph_40_l_2_1']
  #graph = ['graph_40_l_3_1']
  #graph = ['graph_20_1_small']
  if write_output:
    output_file = open(folder_name + '/' + 'top_down_output.txt', 'w')
    output_file.write('filename;stretch;objective;\n')
    output_file.close()
  for i in range(len(graph)):
    filename = folder_name + '/' + graph[i] + '.txt'
    print(filename)
    G, subset_arr = build_networkx_graph(filename)
    print_edges(G)
    #print(subset_arr)
    for param in stretch_arr:
      log_file_name = folder_name + '/' + 'print_log_' + graph[i] + '_stretch_' + str(param) + '.txt'
      if write_output:
        output_file = open(folder_name + '/' + 'top_down_output.txt', 'a')
        output_file.write(filename+';')
        output_file.close()
        #print('Stretch factor: ', param)
        output_file = open(folder_name + '/' + 'top_down_output.txt', 'a')
        output_file.write(str(param)+';')
        output_file.close()
      start_top_down = time.time()
      MLGS = top_down(G, subset_arr, checker, param)
      #MLGS = top_down_exact(G, subset_arr, checker, param)
      l = len(MLGS)
      for k in range(len(subset_arr)):
        subset = subset_arr[l-k-1]
        all_pairs = all_pairs_from_subset(subset)
        #for p in all_pairs:
        #  print('pair:', p)
        #  print('sp:', nx.shortest_path_length(G, p[0], p[1], 'weight'))
        #  print('sp in spanner:', nx.shortest_path_length(MLGS[k], p[0], p[1], 'weight'))
      #exact_graph(G, subset, checker, param)
      #print('Time to compute top down: ', (time.time() - start_top_down))
      s = 0
      for k in range(len(MLGS)):
        G_S = MLGS[k]
        print_edges(G_S)
        if not verify_spanner_with_checker(G_S, G, all_pairs_from_subset(subset_arr[k]), checker, param):
          print('Wrong!!!')
        #print(G_S.edges())
        for e in G_S.edges():
          s = s + G_S.get_edge_data(e[0], e[1])['weight']
          #print(G_S.get_edge_data(e[0], e[1])['weight'])
      print('Objective value: ', s)
      if write_output:
        output_file = open(folder_name + '/' + 'top_down_output.txt', 'a')
        output_file.write(str(s)+';\n')
        output_file.close()

#check_top_down(multiplicative_check)

# returns true if the paths are same
def path_comparison(p, sub_pth):
  if len(p)!=len(sub_pth):
    return False
  matched = True
  for j in range(len(p)):
    if not p[j] == sub_pth[j]:
      matched = False
  if not matched:
    for j in range(len(p)):
      t = len(p)-j-1
      if not p[t] == sub_pth[j]:
        matched = False
  return matched

def add_and_update_paths(G, all_paths, all_pairs, pth, pair_marker, checker, param):
  print(all_paths)
  pair_to_paths = get_related_paths(G, all_pairs, [pth], checker, param)
  for i in range(len(all_pairs)):
    if pair_to_paths[i]:
      contains, ui, vi = path_contains_pair(pth, all_pairs[i][0], all_pairs[i][1])
      #print('contains, ui, vi:', contains, ui, vi)
      pair_marker[i] = True
      # Do the SUBPATH already exist?
      sub_pth = pth[ui:vi+1]
      path_already_exist = False
      for p in all_paths:
        matched = path_comparison(p, sub_pth)
        if matched:
          path_already_exist = True
          break
      if not path_already_exist:
        # add if it maintains spanner property
        G_p = nx.Graph()
        for pi in range(len(sub_pth)-1):
          G_p.add_weighted_edges_from([(sub_pth[pi], sub_pth[pi+1], G.get_edge_data(sub_pth[pi], sub_pth[pi+1])['weight'])])
        if verify_spanner_with_checker(G_p, G, [all_pairs[i]], checker, param):
          #print(all_pairs[i], ' is covered')
          all_paths.append(sub_pth)
  print(all_paths)


def column_generation(G, subset_arr, checker, param):
  my_inf = 1000000
  print('Graph:', end=' ')
  print_edges(G)
  print(subset_arr)

  start_time_initialization = time.time()
  # Adding path

  # initialize
  all_pairs_arr = []
  pair_has_a_path_arr = []
  init_path_arr = []
  for subset in subset_arr:
    all_pairs = all_pairs_from_subset(subset)
    all_pairs_arr.append( all_pairs )
    pair_has_a_path_arr.append([False]*len(all_pairs))
    init_path_arr.append([])

  for all_pairs_i in range(len(all_pairs_arr)):
    init_path = []
    all_pairs = all_pairs_arr[all_pairs_i]
    all_pair_has_a_path = False
    while not all_pair_has_a_path:
      # find a pair that does not has a path
      curr_pair = -1
      for pair_i in range(len(all_pairs)):
        if not pair_has_a_path_arr[all_pairs_i][pair_i]:
          curr_pair = pair_i
          break
      #print('current pair: ',all_pairs_arr[all_pairs_i][curr_pair])
      # Add a path that adds this path while updating the datastucture properly
      pth = nx.dijkstra_path(G,all_pairs_arr[all_pairs_i][curr_pair][0], all_pairs_arr[all_pairs_i][curr_pair][1])
      #print(pth)
      add_and_update_paths(G, init_path_arr[all_pairs_i], all_pairs_arr[all_pairs_i], pth, pair_has_a_path_arr[all_pairs_i], checker, param)
      # check whether all pair has a path
      all_pair_has_a_path = True
      for pair_i in range(len(all_pairs)):
        if not pair_has_a_path_arr[all_pairs_i][pair_i]:
          all_pair_has_a_path = False
          break
    init_path_arr.append(init_path)
  print_log('Time for initialization: ' + str(time.time() - start_time_initialization) + '\n')

  while True:
    # run primal on the initialized paths
    start_time_lp = time.time()
    rownames, dual_vals, obj_value, colnames, prim_vals = primal_program(G, all_pairs_arr, init_path_arr, 'lp', checker, param)
    #for i in range(len(rownames)):
    #  print(rownames[i], ':', dual_vals[i])
    print_log('objective value: ' + str(obj_value) + '\n')
    for i in range(len(rownames)):
      print_log(rownames[i] + ":" + "{:2.20f}".format(dual_vals[i]) + '\n')
    #print_log('dual_vals: ' + str(dual_vals) + '\n')
    print_log('Time for lp: ' + str(time.time() - start_time_lp) + '\n')
    #print('rownames:', rownames)
    #print('len(rownames):', len(rownames))
    #print('dual_vals:', dual_vals)
    #print('len(dual_vals):', len(dual_vals))
    #edge_to_weight = defaultdict(lambda: my_inf)
    start_time_add_path = time.time()
    total_time_cspp = 0
    edge_to_weight = defaultdict(int)
    path_to_weight = defaultdict(int)
    for i in range(len(rownames)):
      name = rownames[i]
      name = name.split('_')
      if name[0]=='pi':
        edge_to_weight[(int(name[1]), int(name[2]), int(name[4]), int(name[5]), int(name[6]))] = dual_vals[i]
        edge_to_weight[(int(name[1]), int(name[2]), int(name[5]), int(name[4]), int(name[6]))] = dual_vals[i]
      elif name[0]=='sigma':
        path_to_weight[(int(name[1]), int(name[2]), int(name[3]))] = dual_vals[i]
        path_to_weight[(int(name[2]), int(name[1]), int(name[3]))] = dual_vals[i]
    #print('edge_to_weight:', edge_to_weight)
    #print('path_to_weight:', path_to_weight)

    min_pth = None
    min_pth_cost = my_inf
    min_pth_l = -1
    min_pth_pair_i = -1
    for l in range(len(all_pairs_arr)):
      #print('init_path_arr[l]:', init_path_arr[l])
      all_pairs = all_pairs_arr[l]
      for pair_i in range(len(all_pairs)):
        #graphFile = 'tmp_graph.txt'
        pair = all_pairs[pair_i]
        #print('pair:', pair)
        G_cspp=nx.Graph()
        for (u,v,d) in G.edges(data='weight'):
          weight = d
          cost = edge_to_weight[(pair[0], pair[1], u, v, l)]
          #if cost==0:
          #  cost=-0.00001
          G_cspp.add_edge(u,v,weight=weight,cost=cost)
          #print('u, v, w, c:', u, v, weight, cost)
        # ***************** for additve spanner change the following line *********************
        MAX_WEIGHT = param * nx.shortest_path_length(G, pair[0], pair[1], 'weight')
        MAX_COST = path_to_weight[(pair[0], pair[1], l)]

        start_time_cspp = time.time()
        #(returnedPath, cost, weight) = cspp.main(G_cspp, MAX_WEIGHT, MAX_COST, pair[0], pair[1])
        (returnedPath, cost, weight) = bellman_ford(G_cspp, MAX_WEIGHT, MAX_COST, pair[0], pair[1])
        #print('returnedPath, cost, weight:', returnedPath, cost, weight)
        total_time_cspp += time.time() - start_time_cspp
        if cost != -1 and weight != -1:
          if min_pth_cost > cost:
            diff_path = True
            pth = nx.dijkstra_path(returnedPath, pair[0], pair[1])
            for p in init_path_arr[l]:
              matched = path_comparison(p, pth)
              if matched:
                diff_path = False
                break
            if diff_path:
              min_pth_cost = cost
              min_pth = pth
              min_pth_l = l
              min_pth_pair_i = pair_i
              print_log('min_pth:' + str(min_pth) + '\n')
              print_log('cost:' + str(cost) + '\n')
    if min_pth_cost != my_inf:
      print('min_pth:', min_pth)
      #print_log('min_pth:' + str(min_pth) + '\n')
      #for all_pairs_i in range(len(all_pairs_arr)):
      #  add_and_update_paths(G, init_path_arr[all_pairs_i], all_pairs_arr[all_pairs_i], min_pth, pair_has_a_path_arr[all_pairs_i], checker, param)
      init_path_arr[min_pth_l].append(min_pth)
      #pair_has_a_path_arr[min_pth_l][min_pth_pair_i] = True
    else:
      break
    print_log('Time for cspp: ' + str(total_time_cspp) + '\n')
    print_log('Time for add path: ' + str(time.time() - start_time_add_path) + '\n')
  start_time_ilp = time.time()
  rownames, dual_vals, obj_value, colnames, prim_vals = primal_program(G, all_pairs_arr, init_path_arr, 'ilp', checker, param)
  print_log('objective value: ' + str(obj_value) + '\n')
  print_log('Time for ilp: ' + str(time.time() - start_time_ilp) + '\n')

log_file_name = ''
def print_log(s):
 global log_file_name
 log_file = open(log_file_name, 'a')
 log_file.write(s+'\n')
 log_file.close()

def check_column_generation(folder_name, file_name_without_ext, stretch_factor):
  global log_file_name
  #folder_name = 'erdos_renyi_sm2'
  #folder_name = 'erdos_renyi_mid'
  #for i in range(0,1):
  log_file_name = folder_name + '/' + 'print_log_' + file_name_without_ext + '_stretch_' + str(stretch_factor) + '.txt'
  log_file = open(log_file_name, 'w')
  log_file.close()
  filename = folder_name + '/' + file_name_without_ext +'.txt'
  G, subset_arr = build_networkx_graph(filename)
  #subset_arr = [subset_arr[1]]
  column_generation(G, subset_arr, multiplicative_check, stretch_factor)
  #print('cost of brute force:',brute_force_ilp(G, subset_arr, multiplicative_check, stretch_factor))
  #print_log('cost of brute force:',brute_force_ilp(G, subset_arr, multiplicative_check, stretch_factor))
  #print('cost of naive_ilp:',naive_ilp(G, subset_arr, stretch_factor))

#check_column_generation(sys.argv[1], sys.argv[2], float(sys.argv[3]))

def flow_ilp(G, subset_arr, checker, param):
  my_inf = 1000000
  print('Graph:', end=' ')
  print_edges(G)
  print(subset_arr)

  # initialize
  all_pairs_arr = []
  for subset in subset_arr:
    all_pairs = all_pairs_from_subset(subset)
    all_pairs_arr.append( all_pairs )

  obj = list()
  sense = ""
  rownames = list()
  total_rows = 0
  colnames = list()
  total_columns = 0
  ub = list()
  lb = list()
  get_column = dict()
  ctype = list()

  rows = []
  cols = []
  vals = []

  constraint_values = []

  for l in range(len(all_pairs_arr)):
    all_pairs = all_pairs_arr[l]
    #What are the variables in initial primal lp?
    #One veriable for each edge, xe, and one variable for each path, x(i,j)u,v
    #The objective function is simple, for xe's, the weight of e is the coefficient, for x(i,j)u,v's, coefficient is zero
    for (u,v,d) in G.edges(data='weight'):
      obj.append(d)
      var_name1 = 'x_'+str(u)+'_'+str(v)+'_'+str(l)
      var_name2 = 'x_'+str(v)+'_'+str(u)+'_'+str(l)
      colnames.append(var_name1)
      get_column[var_name1] = total_columns
      get_column[var_name2] = total_columns
      total_columns = total_columns + 1

    # x(i,j)u,v
    for i in range(len(all_pairs)):
      u = all_pairs[i][0]
      v = all_pairs[i][1]
      for (i,j,d) in G.edges(data='weight'):
        obj.append(0)
        var_name = 'x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)+'_'+str(l)
        colnames.append(var_name)
        get_column[var_name] = total_columns
        total_columns = total_columns + 1
        obj.append(0)
        var_name = 'x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)+'_'+str(l)
        colnames.append(var_name)
        get_column[var_name] = total_columns
        total_columns = total_columns + 1

    # constraint 7 -> stretch constraint
    for i in range(len(all_pairs)):
      u = all_pairs[i][0]
      v = all_pairs[i][1]
      rownames.append('stretchiness_p_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1])+'_'+str(l))
      constraint_values.append(param*nx.shortest_path_length(G, u, v, 'weight'))
      sense = sense + 'L'
      for (i,j,d) in G.edges(data='weight'):
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)+'_'+str(l)])
        vals.append(d)
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)+'_'+str(l)])
        vals.append(d)
      total_rows = total_rows + 1

    print_log('total_rows:' + str(total_rows))

    # constraint 8 -> flow constraint
    for p in range(len(all_pairs)):
      u = all_pairs[p][0]
      v = all_pairs[p][1]
      for i in range(len(G.nodes())):
        rownames.append('flow_p_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(l))
        if i==u:
          constraint_values.append(1)
        elif i==v:
          constraint_values.append(-1)
        else:
          constraint_values.append(0)
        sense = sense + 'E'
        for j in G[i].keys():
          rows.append(total_rows)
          cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)+'_'+str(l)])
          vals.append(1)
          rows.append(total_rows)
          cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)+'_'+str(l)])
          vals.append(-1)
        total_rows = total_rows + 1

    print_log('total_rows:' + str(total_rows))

    # constraint 9 ->  constraint
    for p in range(len(all_pairs)):
      u = all_pairs[p][0]
      v = all_pairs[p][1]
      for i in range(len(G.nodes())):
        rownames.append('cons_9_p_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(l))
        constraint_values.append(1)
        sense = sense + 'L'
        for j in G[i].keys():
          rows.append(total_rows)
          cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)+'_'+str(l)])
          vals.append(1)
        total_rows = total_rows + 1

    print_log('total_rows:' + str(total_rows))

    # constraint 10 ->  constraint
    for p in range(len(all_pairs)):
      u = all_pairs[p][0]
      v = all_pairs[p][1]
      for (i,j,d) in G.edges(data='weight'):
        rownames.append('cons_10_p_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)+'_'+str(l))
        constraint_values.append(0)
        sense = sense + 'L'
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)+'_'+str(l)])
        vals.append(1)
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)+'_'+str(l)])
        vals.append(1)
        rows.append(total_rows)
        cols.append(get_column['x_'+str(i)+'_'+str(j)+'_'+str(l)])
        vals.append(-1)
        total_rows = total_rows + 1

    # multi-level constraint
    if l>0:
      for (u,v,d) in G.edges(data='weight'):
        constraint_values.append(0)
        rownames.append("r_"+str(total_rows))
        sense = sense + "L"
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+"_"+str(l-1)])
        vals.append(1)
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+"_"+str(l)])
        vals.append(-1)
        total_rows = total_rows + 1

    print_log('total_rows:' + str(total_rows))

  for i in range(len(obj)):
    #init_prim_ub.append(cplex.infinity)
    ub.append(1.0)
    lb.append(0.0)

    ctype.append(cplex.Cplex().variables.type.integer)

  print_log('total_rows:' + str(total_rows))

  prob = cplex.Cplex()
  prob.objective.set_sense(prob.objective.sense.minimize)
  prob.linear_constraints.add(rhs = constraint_values, senses = sense, names = rownames)
  prob.variables.add(obj = obj, ub = ub, lb = lb, types=ctype, names = colnames)

  name_indices = [i for i in range(len(obj))]
  names = prob.variables.get_names(name_indices)

  prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
  prob.write(log_file_name.split('/')[0]+ "/" + "model_" + log_file_name.split('/')[1][10:len(log_file_name.split('/')[1])-4] + ".lp")
  prob.solve()
  print("Solution value  = ", prob.solution.get_objective_value())
  print_log("Solution value  = " + str(prob.solution.get_objective_value()))
  numcols = prob.variables.get_num()
  x = prob.solution.get_values()
  for j in range(numcols):
    print("Column %s:  Value = %10f" % (names[j], x[j]))
  return names, x

def test_flow_ilp(folder_name, file_name_without_ext, stretch_factor):
  global log_file_name
  #log_file_name = './print_log_graph_triangle.txt'
  #log_file_name = './print_log_graph_6.txt'
  #log_file_name = './print_log_graph_multi_path.txt'
  #G = nx.Graph()
  #G.add_weighted_edges_from([(0, 1, 5), (1, 2, 6), (2, 0, 23)])
  #G.add_weighted_edges_from([(0, 1, 5), (1, 2, 6), (2, 0, 23), (2, 3, 3), (3, 4, 10), (4, 5, 7), (3, 5, 16), (0, 5, 16)])
  #G.add_weighted_edges_from([(0, 1, 5), (1, 2, 6), (2, 0, 23), (2, 3, 3), (3, 4, 10), (4, 5, 7), (3, 5, 16), (0, 5, 14)])
  #G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (3, 5, 2)])
  #flow_ilp(G, [[0, 1, 2]], multiplicative_check, 2)
  #flow_ilp(G, [[0, 1, 4, 5]], multiplicative_check, 2)
  log_file_name = folder_name + '/' + 'print_log_' + file_name_without_ext + '_stretch_' + str(stretch_factor) + '.txt'
  log_file = open(log_file_name, 'w')
  log_file.close()
  filename = folder_name + '/' + file_name_without_ext +'.txt'
  G, subset_arr = build_networkx_graph(filename)
  #subset_arr = [subset_arr[1]]
  flow_ilp(G, subset_arr, multiplicative_check, stretch_factor)

#test_flow_ilp(sys.argv[1], sys.argv[2], int(sys.argv[3]))

def find_top_level(v, subset_arr):
    for l in range(len(subset_arr)):
        if v in subset_arr[l]:
            return len(subset_arr)-l
    return 0

def single_flow_ilp(G, subset_arr, checker, param):
  my_inf = 1000000
  #print('Graph:', end=' ')
  #print_edges(G)
  #print(subset_arr)

  # initialize
  all_pairs_arr = []
  for subset in subset_arr:
    all_pairs = all_pairs_from_subset(subset)
    all_pairs_arr.append( all_pairs )

  obj = list()
  sense = ""
  rownames = list()
  total_rows = 0
  colnames = list()
  total_columns = 0
  ub = list()
  lb = list()
  get_column = dict()
  ctype = list()

  rows = []
  cols = []
  vals = []

  constraint_values = []

  dis = dict(nx.all_pairs_bellman_ford_path_length(G))

  W_max = max([d for (i,j,d) in G.edges(data='weight')])

  '''
  while True:
    graph_changed = False
    for (i,j,d) in G.edges(data='weight'):
      if dis[i][j]<d:
        G.remove_edge(i, j)
        dis = dict(nx.all_pairs_bellman_ford_path_length(G))
        graph_changed = True
        break
    if not graph_changed:
      break
  '''

  all_pairs = all_pairs_arr[len(all_pairs_arr)-1]
  #What are the variables in initial primal lp?
  #One veriable for each edge, xe, and one variable for each path, x(i,j)u,v
  #The objective function is simple, for xe's, the weight of e is the coefficient, for x(i,j)u,v's, coefficient is zero
  for (u,v,d) in G.edges(data='weight'):
    #obj.append(d)
    # Here we care about number of edges
    obj.append(1)
    var_name1 = 'x_'+str(u)+'_'+str(v)
    var_name2 = 'x_'+str(v)+'_'+str(u)
    colnames.append(var_name1)
    get_column[var_name1] = total_columns
    get_column[var_name2] = total_columns
    total_columns = total_columns + 1

  # x(i,j)u,v
  for i in range(len(all_pairs)):
    u = all_pairs[i][0]
    v = all_pairs[i][1]
    for (i,j,d) in G.edges(data='weight'):
      obj.append(0)
      var_name = 'x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)
      colnames.append(var_name)
      get_column[var_name] = total_columns
      total_columns = total_columns + 1
      obj.append(0)
      var_name = 'x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)
      colnames.append(var_name)
      get_column[var_name] = total_columns
      total_columns = total_columns + 1

  # constraint 7 -> stretch constraint
  for i in range(len(all_pairs)):
    u = all_pairs[i][0]
    v = all_pairs[i][1]
    rownames.append('stretchiness_p_'+str(all_pairs[i][0])+'_'+str(all_pairs[i][1]))
    #constraint_values.append(param*nx.shortest_path_length(G, u, v, 'weight'))
    constraint_values.append(param*W_max + dis[u][v])
    sense = sense + 'L'
    for (i,j,d) in G.edges(data='weight'):
      rows.append(total_rows)
      cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)])
      vals.append(d)
      rows.append(total_rows)
      cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)])
      vals.append(d)
    total_rows = total_rows + 1

  #print_log('total_rows:' + str(total_rows))

  # constraint 8 -> flow constraint
  for p in range(len(all_pairs)):
    u = all_pairs[p][0]
    v = all_pairs[p][1]
    for i in range(len(G.nodes())):
      rownames.append('flow_p_'+str(u)+'_'+str(v)+'_'+str(i))
      if i==u:
        constraint_values.append(1)
      elif i==v:
        constraint_values.append(-1)
      else:
        constraint_values.append(0)
      sense = sense + 'E'
      for j in G[i].keys():
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)])
        vals.append(1)
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)])
        vals.append(-1)
      total_rows = total_rows + 1

  #print_log('total_rows:' + str(total_rows))

  # constraint 9 ->  constraint
  for p in range(len(all_pairs)):
    u = all_pairs[p][0]
    v = all_pairs[p][1]
    for i in range(len(G.nodes())):
      rownames.append('cons_9_p_'+str(u)+'_'+str(v)+'_'+str(i))
      constraint_values.append(1)
      sense = sense + 'L'
      for j in G[i].keys():
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)])
        vals.append(1)
      total_rows = total_rows + 1

  #print_log('total_rows:' + str(total_rows))

  # constraint 10 ->  constraint
  for p in range(len(all_pairs)):
    u = all_pairs[p][0]
    v = all_pairs[p][1]
    for (i,j,d) in G.edges(data='weight'):
      rownames.append('cons_10_p_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j))
      constraint_values.append(0)
      sense = sense + 'L'
      rows.append(total_rows)
      cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)])
      vals.append(min(find_top_level(u, subset_arr), find_top_level(v, subset_arr)))
      rows.append(total_rows)
      cols.append(get_column['x_'+str(i)+'_'+str(j)])
      vals.append(-1)
      total_rows = total_rows + 1

  # constraint 11 ->  constraint
  for p in range(len(all_pairs)):
    u = all_pairs[p][0]
    v = all_pairs[p][1]
    for (i,j,d) in G.edges(data='weight'):
      rownames.append('cons_11_p_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i))
      constraint_values.append(0)
      sense = sense + 'L'
      rows.append(total_rows)
      cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(j)+'_'+str(i)])
      vals.append(min(find_top_level(u, subset_arr), find_top_level(v, subset_arr)))
      rows.append(total_rows)
      cols.append(get_column['x_'+str(i)+'_'+str(j)])
      vals.append(-1)
      total_rows = total_rows + 1

  # constraint 12 ->  shortest_path_test
  for p in range(len(all_pairs)):
    u = all_pairs[p][0]
    v = all_pairs[p][1]
    for (i,j,d) in G.edges(data='weight'):
      if dis[u][i] + d + dis[j][v] > (param*W_max + dis[u][v]):
        rownames.append('reduce_p_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j))
        constraint_values.append(0)
        sense = sense + 'E'
        rows.append(total_rows)
        cols.append(get_column['x_'+str(u)+'_'+str(v)+'_'+str(i)+'_'+str(j)])
        vals.append(1)
        total_rows = total_rows + 1

  #print_log('total_rows:' + str(total_rows))

  for i in range(len(obj)):
    #init_prim_ub.append(cplex.infinity)
    if len(colnames[i].split('_'))==3:
      ub.append(len(subset_arr))
      ctype.append(cplex.Cplex().variables.type.continuous)
    else:
      ub.append(1.0)
      ctype.append(cplex.Cplex().variables.type.integer)
    lb.append(0.0)

  #print_log('total_rows:' + str(total_rows))

  prob = cplex.Cplex()
  prob.set_log_stream(None)
  prob.set_error_stream(None)
  prob.set_warning_stream(None)
  prob.set_results_stream(None)
  prob.objective.set_sense(prob.objective.sense.minimize)
  prob.linear_constraints.add(rhs = constraint_values, senses = sense, names = rownames)
  prob.variables.add(obj = obj, ub = ub, lb = lb, types=ctype, names = colnames)

  name_indices = [i for i in range(len(obj))]
  names = prob.variables.get_names(name_indices)

  prob.linear_constraints.set_coefficients(zip(rows, cols, vals))
  #prob.write(log_file_name.split('/')[0]+ "/" + "model_" + log_file_name.split('/')[1][10:len(log_file_name.split('/')[1])-4] + ".lp")
  prob.solve()
  #print("Solution value  = ", prob.solution.get_objective_value())
  #print_log("Solution value  = " + str(prob.solution.get_objective_value()))
  numcols = prob.variables.get_num()
  x = prob.solution.get_values()
  #for j in range(numcols):
  #  print("Column %s:  Value = %10f" % (names[j], x[j]))
  #return names, x
  return prob.solution.get_objective_value()

def test_single_flow_ilp(folder_name, file_name_without_ext, stretch_factor):
  global log_file_name
  #log_file_name = './print_log_graph_triangle.txt'
  #log_file_name = './print_log_graph_6.txt'
  #log_file_name = './print_log_graph_multi_path.txt'
  #G = nx.Graph()
  #G.add_weighted_edges_from([(0, 1, 5), (1, 2, 6), (2, 0, 23)])
  #G.add_weighted_edges_from([(0, 1, 5), (1, 2, 6), (2, 0, 23), (2, 3, 3), (3, 4, 10), (4, 5, 7), (3, 5, 16), (0, 5, 16)])
  #G.add_weighted_edges_from([(0, 1, 5), (1, 2, 6), (2, 0, 23), (2, 3, 3), (3, 4, 10), (4, 5, 7), (3, 5, 16), (0, 5, 14)])
  #G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (3, 5, 2)])
  #flow_ilp(G, [[0, 1, 2]], multiplicative_check, 2)
  #flow_ilp(G, [[0, 1, 4, 5]], multiplicative_check, 2)
  #log_file_name = folder_name + '/' + 'print_log_' + file_name_without_ext + '_stretch_' + str(stretch_factor) + '.txt'
  #log_file = open(log_file_name, 'w')
  #log_file.close()
  filename = folder_name + '/' + file_name_without_ext +'.txt'
  G, subset_arr = build_networkx_graph(filename)
  #subset_arr = [subset_arr[1]]
  start_time = time.time()
  solution_value = single_flow_ilp(G, subset_arr, multiplicative_check, stretch_factor)
  total_time = time.time() - start_time
  print(folder_name + ';' + file_name_without_ext + ';' + str(solution_value) + ';' + str(total_time) + ';\n')


def read_model(model_file):
        prob = cplex.Cplex()
        prob.read(model_file)
        name_indices = [i for i in range(len(prob.objective.get_linear()))]
        names = prob.variables.get_names(name_indices)
        prob.solve()
        #print('Is dual feasible:')
        #print(prob.solution.is_dual_feasible())

        print("Solution value  = ", prob.solution.get_objective_value())
        numcols = prob.variables.get_num()
        x = prob.solution.get_values()
        for j in range(numcols):
                print("Column %s:  Value = %2.20f" % (names[j], x[j]))

#read_model('erdos_renyi_one_level_2/model_graph_20_1_stretch_2.lp')

if __name__ == '__main__':
  #check_top_down(multiplicative_check)
  test_single_flow_ilp(sys.argv[1], sys.argv[2], float(sys.argv[3]))
  #test_flow_ilp(sys.argv[1], sys.argv[2], float(sys.argv[3]))
  #check_bot_up(multiplicative_check)


