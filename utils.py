#!/usr/bin/python3


import sys
import networkx as nx
import matplotlib.pyplot as plt
import random as rnd
import itertools as it
import pathfinding as pf


'''
GRAPHML EDGE ATTRIBUTES
attr.name="htlc_maximum_msat" attr.type="long"
attr.name="htlc_minimim_msat" attr.type="long"
attr.name="fee_proportional_millionths" attr.type="long"
attr.name="fee_base_msat" attr.type="long"
attr.name="features" attr.type="string"
attr.name="timestamp" attr.type="long"
attr.name="destination" attr.type="string"
attr.name="source" attr.type="string"
attr.name="scid" attr.type="string"

GRAPHML NODE ATTRIBUTES
attr.name="in_degree" attr.type="long"
attr.name="out_degree" attr.type="long"
attr.name="addresses" attr.type="string"
attr.name="alias" attr.type="string"
attr.name="rgb_color" attr.type="string"
attr.name="features" attr.type="string"
attr.name="timestamp" attr.type="long"
attr.name="id" attr.type="string"
'''

'''
SAMPLE NETWORKX EDGE ATTRIBUTES
(
'0342dd8568081ae1bdd852c0d9440dd22e4bbc432391975e6a1e1f2688e3ca6fc1',
'0242a4ae0c5bef18048fbecf995094b74bfb0f7391418d71ed394784373f41e4f3',
	{
	'scid': '677836x2386x1/1',
	'source': '0342dd8568081ae1bdd852c0d9440dd22e4bbc432391975e6a1e1f2688e3ca6fc1',
	'destination': '0242a4ae0c5bef18048fbecf995094b74bfb0f7391418d71ed394784373f41e4f3',
	'timestamp': 1630848990,
	'fee_base_msat': 1000,
	'fee_proportional_millionths': 1,
	'htlc_minimim_msat': 1000,
	'htlc_maximum_msat': 83517000,
	'cltv_expiry_delta': 40
	}
)

SAMPLE NETWORKX NODE ATTRIBUTES
(
'039a64895e50e2fb4381c908308fe155355ea3332faff5589c8946e1b92f9da7f4',
	{
	'id': '039a64895e50e2fb4381c908308fe155355ea3332faff5589c8946e1b92f9da7f4',
	'timestamp': 1596491959,
	'features': '8000000002aaa2',
	'rgb_color': '039a64',
	'alias': 'PEEVEDDEITY',
	'out_degree': 0,
	'in_degree': 1
	}
)
'''


def get_edge_base_fee(e):
	if isinstance(e, dict):
		return e["fee_base_msat"]
	return e[2]["fee_base_msat"]


def get_edge_prop_fee(e):
	if isinstance(e, dict):
		return e["fee_proportional_millionths"]
	return e[2]["fee_proportional_millionths"]


def get_edge_htlc_max(e):
	if isinstance(e, dict):
		return e["htlc_maximum_msat"]
	return e[2]["htlc_maximum_msat"]


def get_edge_delay(e):
	if isinstance(e, dict):
		return e["cltv_expiry_delta"]
	return e[2]["cltv_expiry_delta"]


def get_edge_delay_average(G):
	edge_dict = nx.get_edge_attributes(G, name="cltv_expiry_delta")
	total = sum(edge_dict.values())
	num_edges = len(edge_dict)
	return total / num_edges


def set_edge_htlc_budget(e, b):
	e[2]["htlc_budget"] = b


def get_edge_htlc_budget(e):
	if isinstance(e, dict):
		return e["htlc_budget"]
	return e[2]["htlc_budget"]


# number of nodes in this path = len(path)
# number of edges in this path = len(path) - 1
def get_edges(G, path):
	edges = []
	for i in range(len(path) - 1):
		edges.append(G[path[i]][path[i + 1]])
	return edges


def get_node_id(n):
	# return n[1]["id"]
	if isinstance(n, dict):
		return n["id"]
	return n[0]


def get_node_centrality(G, node_id):
	return G.nodes[node_id]["node_centrality"]


def get_node_connectivity(G, node_id):
	return G.nodes[node_id]["connectivity"]


def get_node_inbound_funds_msat(G, node_id):
	return G.nodes[node_id]["inbound_funds"]


def get_node_inbound_path_lenght(G, node_id):
	return G.nodes[node_id]["inbound_path_lenght"]


def get_node_in_degree(n):
	if isinstance(n, dict):
		return n["in_degree"]
	return n[1]["in_degree"]


def get_node_out_degree(n):
	if isinstance(n, dict):
		return n["out_degree"]
	return n[1]["out_degree"]


def get_node_betweenness(G, node_id):
	return G.nodes[node_id]["betweenness"]


def get_node_betweenness_sum(G):
	s = 0
	for node_id in G:
		s += get_node_betweenness(G, node_id)

	return s


def get_spent_base_fees(G, spath, amount):
	s = 0
	edges = get_edges(G, spath)
	for e in edges:
		base_fee = get_edge_base_fee(e)
		# prop_fee = get_edge_prop_fee(e) / (1000 * 1000)
		# s += base_fee + amount * prop_fee
		s += base_fee

	return int(s)


# This function returns an iterator over all pairs of nodes (instead of a list of
# 112 million pairs of strings which takes 21 GB of memory).
def get_node_pairs(G, scale=0.001, share_from=0, share_to=0.1):
	if scale <= 0 or share_from < 0 or share_from >= 1 or share_to <= 0 or share_to > 1 or share_from >= share_to:
		print(f"[ERROR] Faulty input for scale or share")
		return []

	nodes = list(G.nodes(data=True))
	node_pairs = []

	if scale == 1.0 and share_from == 0 and share_to == 1:
		for src_node in nodes:
			src_node_id = get_node_id(src_node)
			i = nodes.index(src_node)
			for tgt_node in nodes[i + 1:]:
				tgt_node_id = get_node_id(tgt_node)
				node_pairs.append([src_node_id, tgt_node_id])
		return node_pairs
		# return it.combinations(nodes, 2)  # This excludes swapped endpoint duplicates (if [0, 1] is in, [1, 0] is not )

	num_nodes = len(nodes)
	print(f"scale: {scale} ({int(num_nodes * scale)} nodes), share: from {share_from} to {share_to}")
	print(f"num_nodes: {int(num_nodes)}")

	rnd.seed()
	from_node = int(share_from * num_nodes)
	to_node = int(share_to * num_nodes)
	if to_node == num_nodes:
		to_node -= 1

	for node in nodes:
		pairs = []
		node_id1 = get_node_id(node)
		# print(f"\tprocessing node {node_id1}")

		for _ in range(int(num_nodes * scale)):
			if from_node == to_node:
				if from_node == 0:
					to_node += 1
				else:
					from_node -= 1

			# print(f"\t\tfrom {from_node} to {to_node}")
			i = rnd.randint(from_node, to_node)
			node_id2 = get_node_id(nodes[i])

			# Maybe sscratch the nx.has_path condition if it's too time expensive
			# while node_id2 == node_id1 or [node_id1, node_id2] in pairs or not nx.has_path(G, node_id1, node_id2):
			while node_id2 == node_id1 or [node_id1, node_id2] in pairs:
				print(f"[{node_id1}, {node_id2}] this tgt node is badly chosen, get a new one")
				i = rnd.randint(from_node, to_node)
				node_id2 = get_node_id(nodes[i])

			# print(f"\t\tprocessing pair [{node_id1}, {node_id2}]")
			pairs.append([node_id1, node_id2])
		node_pairs += pairs

	return node_pairs


def decr_htlc_budget(e, amount=1):
	if isinstance(e, dict):
		e["htlc_budget"] -= amount
	else:
		e[2]["htlc_budget"] -= amount


def set_node_centrality(G, pc):
	nx.set_node_attributes(G, pc, name="node_centrality")


def set_htlc_budet(G, b):
	nx.set_edge_attributes(G, b, name="htlc_budget")


# @param type: options are "graphml", "barabasi", "custom", "custom2"
def get_graph(graph_type):
	if graph_type == "graphml":
		# GRAPH: read graphml file from console and create an nx graph from it
		f_graphml = sys.argv[1]
		G = nx.read_graphml(f_graphml)

	elif graph_type == "barabasi":
		# GRAPH: Barabasi Albert Graph, similar to LN topology (?)
		G = nx.barabasi_albert_graph(50, 2)

	elif graph_type == "custom":
		# GRAPH: handmade path graph consisting out of six nodes connected by a set of edges that make
		# the graph an useful sandbox for testing shortest path searches
		G = nx.path_graph(0)
		for i in range(8):
			G.add_node(i, id=i)

		G.add_edge(0, 1, source=0, destination=1, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)
		G.add_edge(1, 2, source=1, destination=2, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)
		G.add_edge(2, 3, source=2, destination=3, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)

		G.add_edge(0, 4, source=0, destination=4, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)
		G.add_edge(4, 5, source=4, destination=5, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)
		G.add_edge(5, 3, source=5, destination=3, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)

		G.add_edge(0, 6, source=0, destination=6, fee_base_msat=1000, fee_proportional_millionths=1, weight=10)
		G.add_edge(6, 3, source=6, destination=3, fee_base_msat=10000, fee_proportional_millionths=1, weight=1)

		G.add_edge(3, 7, source=3, destination=7, fee_base_msat=1000, fee_proportional_millionths=1, weight=1)

	elif graph_type == "custom2":
		G = nx.path_graph(0)
		for i in range(5):
			G.add_node(i, id=i, node_centrality=0)

		G.add_edge(0, 3, source=0, destination=1, weight=99)
		G.add_edge(0, 2, source=0, destination=2, weight=1)
		G.add_edge(0, 3, source=0, destination=3, weight=99)
		G.add_edge(1, 4, source=1, destination=4, weight=99)
		G.add_edge(2, 4, source=2, destination=4, weight=99)
		G.add_edge(3, 4, source=3, destination=4, weight=99)

	elif graph_type == "custom3":
		G = nx.path_graph(0)
		for i in range(19):
			G.add_node(i, id=i)

		G.add_edge(0, 1, source=0, destination=1, fee_base_msat=1, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(0, 2, source=0, destination=2, fee_base_msat=2, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(0, 3, source=0, destination=3, fee_base_msat=3, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(0, 4, source=0, destination=4, fee_base_msat=4, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(4, 5, source=4, destination=5, fee_base_msat=5, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(5, 6, source=5, destination=6, fee_base_msat=6, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(5, 7, source=5, destination=7, fee_base_msat=7, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(5, 8, source=5, destination=8, fee_base_msat=8, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(0, 8, source=0, destination=8, fee_base_msat=8, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)

		G.add_edge(0, 9, source=0, destination=9, fee_base_msat=9, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)

		G.add_edge(9, 10, source=9, destination=10, fee_base_msat=10, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(9, 11, source=9, destination=11, fee_base_msat=11, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(9, 12, source=9, destination=12, fee_base_msat=12, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(9, 13, source=9, destination=13, fee_base_msat=13, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(12, 14, source=12, destination=14, fee_base_msat=14, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(13, 15, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)

		G.add_edge(9, 16, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(9, 17, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(9, 18, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(16, 15, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(17, 15, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(18, 15, source=13, destination=15, fee_base_msat=15, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)

	elif graph_type == "custom4":
		G = nx.path_graph(0)
		for i in range(2):
			G.add_node(i, id=i, node_centrality=0)

		G.add_edge(0, 1, source=0, destination=1, weight=1)

	elif graph_type == "custom5":
		G = nx.path_graph(0)
		for i in range(4):
			G.add_node(i, id=i, node_centrality=0)

		G.add_edge(0, 1, source=0, destination=1, fee_base_msat=11, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(1, 2, source=1, destination=2, fee_base_msat=22, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)
		G.add_edge(2, 3, source=2, destination=3, fee_base_msat=44, fee_proportional_millionths=1,
		           htlc_maximum_msat=1000 * 1000 * 1000, cltv_expiry_delta=144)

	else:
		print(f"Unknown graph type <{graph_type}>. Falling back onto graphml type.")
		f_graphml = sys.argv[1]
		G = nx.read_graphml(f_graphml)
	return G


def print_degrees(G, skip_zero_lines=False):
	deg = nx.degree_histogram(G)

	skip_next = False
	total = sum(deg)
	cumu = 0

	print("DEGR  #NODES".ljust(19, ' ') + "PERC \tCUMULATIVE PERC")
	print("===============================================")
	for i in range(len(deg)):
		if deg[i] == 0 and not skip_zero_lines:
			if not skip_next:
				print("..")
				skip_next = True
		elif deg[i] != 0:
			skip_next = False
			cumu += deg[i]
			print("{0:0>4}: {1}".format(i + 1, deg[i]).ljust(19, ' ') +
			      "{0:0>4}%\t{1}%".format(round(100 * deg[i] / total, 3), round(100 * cumu / total, 3)))

	# id = "0342dd8568081ae1bdd852c0d9440dd22e4bbc432391975e6a1e1f2688e3ca6fc1"
	# print(f"\nDegree of one particular node identified by id={id}:")
	# print(G.degree[id])


def print_shortest_paths_and_adj_nodes():
	G = get_graph("graphml")
	nodes = list(G.nodes(data=True))

	id_1000 = get_node_id(nodes[1000])
	id_2000 = get_node_id(nodes[2000])

	print(f"All shortest paths from id={id_1000} to id={id_2000}:")
	print([(p, len(p)) for p in nx.all_shortest_paths(G, source=id_1000, target=id_2000)])

	p = nx.shortest_path(G, source=id_1000, target=id_2000)
	print(f"One shortest path from id={id_1000} to id={id_2000}:\n{p}")

	adj = G.adj[id_1000]
	print(f"Adjacent nodes of node={id_1000}:\n{adj}")


def print_node_details(node):
	node_dict = node[1]
	for item in node_dict:
		print(f"\t{item}: {node_dict[item]}")
	print("")


def print_edge_details(e):
	edge_dict = e[2]
	for item in edge_dict:
		print(f"\t{item}: {edge_dict[item]}")
	print("")


def print_graph_details(G, n=True, e=True, m=5):
	if n:
		print(f"Number of nodes: {G.number_of_nodes()}")
	if e:
		print(f"Number of edges: {G.number_of_edges()}")

	if n:
		nodes = list(G.nodes(data=True))
		print("\nNode details:")
		for i in range(m):
			print(f"[node {i}]")
			print_node_details(nodes[i])

	if e:
		edges = list(G.edges(data=True))
		print("\nEdge details:")
		for i in range(m):
			print(f"[edge {i}]")
			print_edge_details(edges[i])


def print_edge_weights(G, n=3):
	edges_by_weight = sorted(G.edges(), key=lambda m: G.edges[m]["weight"], reverse=False)
	for i in range(n):
		edge_id = edges_by_weight[i]
		edge_w = G.edges[edge_id]["weight"]
		print(f"edge[{i}][\"weight\"]: {edge_w}")

	for j in range(n):
		edge_id = edges_by_weight[-j]
		edge_w = G.edges[edge_id]["weight"]
		print(f"edge[{-j}][\"weight\"]: {edge_w}")


# This function takes a list of nodes and prints the cumulative percentage of paths that go through it
def print_results_cumu(G, node_ids_by_node_centrality, top_n):
	num_paths = 0
	num_long_paths = 0

	for node_id in G:
		num_long_paths += get_node_centrality(G, node_id)

	for i in range(top_n):
		pc = get_node_centrality(G, node_ids_by_node_centrality[i])
		print(f"node[{i}][\"node_centrality\"]: {pc}")
		num_paths += pc
		perc = num_paths / num_long_paths
		print(f"{int(100 * perc)}% ({100 * perc}%) of paths going through top {i + 1} nodes")
	print("")


# Sample graph that is used to showcase networkX's visualization capabilities.
def visualize_graph_and_print_simple_paths(G):
	nodes = list(G.nodes(data=True))
	src = get_node_id(nodes[0])
	tgt = get_node_id(nodes[-1])
	print([p for p in nx.all_simple_paths(G, source=src, target=tgt, cutoff=3)])
	nx.draw(G, with_labels=True)
	# plt.hist([v for k, v in nx.degree(G)])
	plt.show()


# Sandbox to perform actions on the graph, vizualize it, and print some information
def sample_output(G):
	# Save all paths from src to tgt in a set and make aflattened list from set
	src, tgt = 0, 3
	unique_single_paths_set = set(tuple(p) for p in nx.all_simple_edge_paths(G, src, tgt))
	endpoint_list = [ep for sublist in list(unique_single_paths_set) for ep in sublist]
	print(f"Endpoint list:\n{endpoint_list}")

	print(f"\nAll paths from source={src} to target={tgt}:")
	for p in nx.all_simple_paths(G, source=src, target=tgt):
		print(p)

	rnd.seed()  # needed for random fuzz in CLN_weight()
	print(f"\nLowest weighted path from source={src} to target={tgt}:")
	p = nx.shortest_path(G, source=src, target=tgt, weight="weight")
	print(p)

	print(f"\nPath weight from source={src} to target={tgt}:")
	print(nx.dijkstra_path_length(G, src, tgt))

	print_graph_details(G, n=True, e=False, m=2)

	nx.draw_networkx(G, with_labels=True)
	plt.show()


def pgfplot_node_degree(G, aggr=2, cap=50):
	print("\n[DEBUG] Computing node degrees in pgfplot format...")
	deg = nx.degree_histogram(G)
	total = sum(deg)
	cumu = 0

	for i in range(len(deg)):
		cumu += deg[i]
		if i % aggr == 0:
			print(f"({i}, {round(100 * cumu / total, 3)})")
			cumu = 0
		if i > cap:
			cumu = sum(deg[i:])
			print(f"(more, {round(100 * cumu / total, 3)})\n")
			break


def pgfplot_edge_attr(G, attr_type, bucket_size, cap):
	edges = list(G.edges(data=True))
	total = len(edges)
	attrs = [0] * (2 + int(cap/bucket_size))

	for e in edges:
		if attr_type == "base_fee":
			attr = get_edge_base_fee(e)
		elif attr_type == "prop_fee":
			attr = get_edge_prop_fee(e)
		elif attr_type == "delay":
			attr = get_edge_delay(e)
		elif attr_type == "htlc_max":
			attr = get_edge_htlc_max(e)
		else:
			print(f"[ERROR] Channel attribute type <{attr_type}> unknown.")
			return -1
			
		if attr > (cap + bucket_size):
			attrs[-1] += 1
		else:
			attrs[round(attr/bucket_size)] += 1

	for i in range(len(attrs) - 1):
		print(f"({i * bucket_size}, {100 * round(attrs[i] / total, 3)})")
	print(f"(more, {100 * round(attrs[-1] / total, 3)})\n")


def pgfplot_base_fee(G, bucket_size=100, cap=1000):
	print("\n[DEBUG] Computing base fees in pgfplot format...")
	pgfplot_edge_attr(G, "base_fee", bucket_size, cap)


# TODO: Check is these bucket_size and cap values are reasonable
def pgfplot_prop_fee(G, bucket_size=1, cap=1000):
	print("\n[DEBUG] Computing proportional fees in pgfplot format...")
	pgfplot_edge_attr(G, "prop_fee", bucket_size, cap)


def pgfplot_max_payment_per_htlc(G, bucket_size=200 * 1000 * 1000, cap=5000 * 1000 * 1000):
	print("\n[DEBUG] Computing htlc_max in pgfplot format...")
	pgfplot_edge_attr(G, "htlc_max", bucket_size, cap)


def pgfplot_channel_delay(G, bucket_size=10, cap=150):
	print("\n[DEBUG] Computing channel delay in pgfplot format...")
	pgfplot_edge_attr(G, "delay", bucket_size, cap)


'''
# Adjust bucket_size and cap to reasonable values after inspecting the plot
# x axis: millisatoshis, y axis: percentage of all transactions
def pgfplot_avg_spent_fees(G, spent_fees, bucket_size=100 * 1000, cap=10*1000*1000):
	# mpp relevant
	# bar plot
	# base fees and prop fees to show the increased cost for multipathing payments
	print("[DEBUG] Computing spent fees in pgfplot format...")
	total = len(spent_fees)
	spent_fees_aggr = [0] * (2 + int(cap/bucket_size))

	for f in spent_fees:
		if f > (cap + bucket_size):
			spent_fees_aggr[-1] += 1
		else:
			spent_fees_aggr[int(f/bucket_size)] += 1

	for i in range(len(spent_fees_aggr) - 1):
		print(f"({i * bucket_size}, {round(spent_fees_aggr[i] / total, 3)})")
	print(f"(more, {round(spent_fees_aggr[-1] / total, 3)})\n\n")
'''


# x axis: lenght of shortest paths, y axis: percentage of all transactions
# Two bars for each x value (singelpathing/multipathing)
def pgfplot_shortest_paths_length_distribution(path_lengths, bucket_size=1, cap=7):
	print("\n[DEBUG] Computing transactions distributed by path length in pgfplot format...")
	total = len(path_lengths)
	# To compute average:
	# [i][0] holds a counter for i'th bucket that specifies spath length,
	# [i][1] holds the sum of the spent fees for this path length
	path_lengths_aggr = [0] * (2 + int(cap/bucket_size))

	for pl in path_lengths:
		if pl[0] > (cap + bucket_size):
			path_lengths_aggr[-1] += 1
		else:
			path_lengths_aggr[int(pl[0] / bucket_size)] += 1

	for i in range(len(path_lengths_aggr) - 1):
		print(f"({i * bucket_size}, {round(path_lengths_aggr[i] / total, 3)})")
	print(f"(more, {round(path_lengths_aggr[-1] / total, 3)})\n")
	# mpp relevant


# I can use this to measure price difference of singlepath transactions vs multipath transactions per path length
# x axis: lenght of shortest paths, y axis: millisatoshis
# Two bars for each x value (singelpathing/multipathing)
def pgfplot_shortest_paths_cost(spent_fees, bucket_size=1, cap=7):
	print("\n[DEBUG] Computing average fees spent per path length in pgfplot format...")
	# To compute average:
	# [i][0] holds a counter for i'th bucket that specifies spath length,
	# [i][1] holds the sum of the spent fees for this path length

	spent_fees_aggr = []
	for _ in range(2 + int(cap/bucket_size)):
		spent_fees_aggr.append([0, 0])
	# Apparently this initialization works differently (makes )
	#   spent_fees_aggr = [[0, 0]] * (2 + int(cap/bucket_size))

	for f in spent_fees:
		if f[0] > (cap + bucket_size):
			spent_fees_aggr[-1][0] += 1
			spent_fees_aggr[-1][1] += f[1]
		else:
			i = int(f[0]/bucket_size)
			spent_fees_aggr[i][0] += 1
			spent_fees_aggr[i][1] += f[1]

	print(f"spent_fees_aggr: {spent_fees_aggr}")
	avg_spent_fees_aggr = []

	for f in spent_fees_aggr:
		if f[0] == 0:
			avg_spent_fees_aggr.append(0)
		else:
			avg_spent_fees_aggr.append(int(f[1] / f[0]))


	for i in range(len(avg_spent_fees_aggr) - 1):
		print(f"({i * bucket_size}, {round(spent_fees_aggr[i][1], 3)})")
	print(f"(more, {round(spent_fees_aggr[-1][1], 3)})\n")
	# mpp relevant


def pgfplot_first_strongest_nodes(G, top_n=30):
	print("\n[DEBUG] Computing first n strongest nodes and cumulative percentage of paths that go through them "
	      "in pgfplot format...")
	node_ids_by_node_centrality = sorted(G.nodes(), key=lambda n: G.nodes[n]["node_centrality"], reverse=True)
	pc_cumu, total = 0, 0

	for node_id in G:
		total += get_node_centrality(G, node_id)

	for i in range(top_n):
		pc = get_node_centrality(G, node_ids_by_node_centrality[i])
		pc_cumu += pc
		perc = pc_cumu / total
		print(f"({i + 1}, {100 * round(pc_cumu / total, 3)})")
		# print(f"{int(100 * perc)}% ({100 * perc}%) of paths going through top {i + 1} nodes")
	print(f"(more, {100 * round((total - pc_cumu) / total, 3)})\n")
	# mpp relevant
	# smooth plot (https://pgfplots.net/smooth-plot/)


def pgfplot_first_strongest_nodes_over_time():
	print("TODO")
	# mpp relevant
	# smooth plot (https://pgfplots.net/smooth-plot/)


# x axis: number of unresponsive most central nodes, y axis: percentage of disconnected nodes
def pgfplot_disconnected_nodes(G, top_n):
	print("\n[DEBUG] Computing number of disconnected nodes cumulatively when top_n most central nodes stop being "
	      "responsive in pgfplot format...")
	node_ids_by_node_centrality = sorted(G.nodes(), key=lambda n: G.nodes[n]["node_centrality"], reverse=True)
	c_cumu, total = 0, 0

	for node_id in G:
		total += get_node_connectivity(G, node_id)

	for i in range(top_n):
		c = get_node_connectivity(G, node_ids_by_node_centrality[i])
		c_cumu += c
		print(f"({i + 1}, {100 * round(c_cumu / total, 3)})")
		# print(f"\tnode {node_ids_by_node_centrality[i]} - node_centrality: {get_node_centrality(G, i)}\t node_connectivity: {c}")
	# "more" data point only makes sense for bar plots, not for smooth plots
	# print(f"(more, {100 * round((total - c_cumu) / total, 3)})\n")
# mpp relevant
# smooth plot (https://pgfplots.net/smooth-plot/)


# x axis: number of unresponsive most central nodes, y axis: opportunity cost [millisatoshis * hours]
def pgfplot_blocked_funds(G, top_n, amount):
	print("\n[DEBUG] Computing the network-wide incurred normalized opportunity cost cumulatively when top_n most "
	      "central nodes stop being responsive in [millisatoshis * hours] in pgfplot format...")
	node_ids_by_node_centrality = sorted(G.nodes(), key=lambda n: G.nodes[n]["node_centrality"], reverse=True)
	cost_cumu, total = 0, 0
	avg_delay = get_edge_delay_average(G)

	for i in range(top_n):
		msat = get_node_inbound_funds_msat(G, node_ids_by_node_centrality[i])
		pl = get_node_inbound_path_lenght(G, node_ids_by_node_centrality[i])
		hrs = pl * avg_delay / 6  # total time = path length * avg delay per channel, block creation time = 1/6 hrs
		cost_cumu += msat * hrs / amount
		print(f"({i + 1}, {100 * round(cost_cumu, 3)})")
	# This "more" would take time to compute since we would need to compute the total cost of all nodes for that, right?
	# print(f"(more, {100 * round((total - cost_cumu), 3)})\n")


# x axis: ?, y axis: ?
def pgfplot_max_htlc_payment_failures():
	print("TODO")
	# mpp relevant


# x axis: ?, y axis: ?
def pgfplot_htlc_budget_payment_failures():
	print("TODO")
	# mpp relevant


