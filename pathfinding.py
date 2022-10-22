#!/usr/bin/python3

import sys
import networkx as nx
import time
import random as rnd
import json
from csiphash24 import siphash24 as sh24
import matplotlib.pyplot as plt
import utils as u
import route_hijacking as rh


def sandbox(G):
	# How many nodes have degree 1, 2, 3, etc. in a formatted table
	# u.print_degrees()

	# Print all shortest paths and neighbour nodes
	# u.print_shortest_paths_and_adj_nodes()

	# Print small sample of nodes and edges to explore their format
	# u.print_graph_details()

	# Sample graph that is used to showcase networkX's visualization capabilities.
	# u.visualize_graph_and_print_simple_paths()

	u.sample_output(G)


'''
1 BTC = 100.000.000 SAT = 100.000.000.000 MSAT
'''

HASH = 0xfedcba9876543210  # TODO: Make this random 64-bit values, produced by siphash24
AMOUNT = 42 * 1000 * 1000  # TODO: This dosn't need to be global
# AMOUNT = 1000 * 1000
COUNTER = 0
MAX_CONCURRENT_HTLCS = 30


def cln_weight(edge):
	risk_factor = 10  # 10 by CLN default
	fuzz = rnd.randint(0, 1000) / 10000 - 0.05  # +- 0.05 by default
	scale = 1 + fuzz * (2 * HASH / (2 ** 64 - 1) + 1)
	base_fee = u.get_edge_base_fee(edge)
	prop_fee = u.get_edge_prop_fee(edge) / (1000 * 1000)
	delay = u.get_edge_delay(edge)
	fee = scale * (base_fee + AMOUNT * prop_fee)
	weight = (AMOUNT + fee) * (delay * risk_factor) + 1
	return weight


# Change htlc_budget, weight, node_centrality, and other edge and node attributes to use CLN's defaults
def init_cln_defaults(G):
	# for e in G.edges(data=True):
		# e[2]["weight"] = cln_weight(e)

	nx.set_edge_attributes(G, values=MAX_CONCURRENT_HTLCS, name="htlc_budget")

	nx.set_node_attributes(G, values=0, name="node_centrality")

	nx.set_node_attributes(G, values=0, name="connectivity")

	nx.set_node_attributes(G, values=0, name="inbound_funds")

	nx.set_node_attributes(G, values=0, name="inbound_path_lenght")


# extrapolate node_centrality, connectivity, inbound_funds,
def extrapolate_mpp_results(G, scale):
	print("TODO extrapolate_mpp_results")


def activate_log():
	t = time.time()
	filename = str(t) + "_benchmark.txt"
	sys.stdout = open(filename, 'w')


def main():
	if len(sys.argv) != 2:
		print("Usage: ./pathfinding.py <GRAPHML FILE>")
		exit()

	# activate_log()

	print("[DEBUG] Loading graphml data and initializing CLN defaults...")
	t_before_graphml = time.time()
	G = u.get_graph("graphml")  # for a list of possible options check utils.py
	init_cln_defaults(G)
	t_before_all_pair = time.time()
	print(f"[DEBUG] Done loading graphml data and initializing CLN defaults "
	      f"({int(10 * (t_before_all_pair - t_before_graphml)) / 10}s)\n")
	print(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}\n")

	num_nodes = G.number_of_nodes()
	scale = 2 / num_nodes
	rh.mpp_colluding_attack(G, AMOUNT, scale)
	if scale < 1:
		extrapolate_mpp_results(G, scale)
	return 1

	u.pgfplot_node_degree(G)
	u.pgfplot_base_fee(G)
	u.pgfplot_prop_fee(G)
	u.pgfplot_max_payment_per_htlc(G)
	u.pgfplot_channel_delay(G)

	t_before_colluding_attack = time.time()
	
	print("[DEBUG] Computing all node pairs...")
	all_spaths_dict = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))
	t_before_singlepathing_colluding = time.time()
	print(f"[DEBUG] Done computing all node pairs "
	      f"({int(10 * (t_before_singlepathing_colluding - t_before_all_pair)) / 10}s)\n")

	print("[DEBUG] Executing colluding Route Hijacking attack for singlepathing transactions...")
	rh.colluding_attack(G, AMOUNT, all_spaths_dict, mpp=False)
	t_before_multipathing_colluding = time.time()
	print(f"[DEBUG] Done executing colluding Route Hijacking attack for singlepathing transactions "
	      f"({int(10 * (t_before_multipathing_colluding - t_before_colluding_attack)) / 10 / 60}min)\n")

	u.pgfplot_first_strongest_nodes(G, top_n=30)

	print("[DEBUG] Execute colluding Route Hijacking attack for multipathing transactions...")
	rh.colluding_attack(G, AMOUNT, all_spaths_dict, mpp=True)
	t_stop = time.time()
	print(f"[DEBUG] Done executing colluding Route Hijacking attack for multipathing transactions "
	      f"({int(10 * (t_stop - t_before_colluding_attack)) / 10 / 60}min)\n")

	u.pgfplot_first_strongest_nodes(G, top_n=30)

	'''
	print(all_spaths_dict)
	{0: {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4], 8: [0, 8], 9: [0, 9], 5: [0, 8, 5], 10: [0, 9, 10],
		11: [0, 9, 11], 12: [0, 9, 12], 13: [0, 9, 13], 6: [0, 8, 5, 6], 7: [0, 8, 5, 7], 15: [0, 9, 13, 15],
		14: [0, 9, 12, 14]},
	 1: {1: [1], 0: [1, 0], 2: [1, 0, 2], 3: [1, 0, 3], 4: [1, 0, 4], 8: [1, 0, 8], 9: [1, 0, 9], 5: [1, 0, 8, 5],
	    10: [1, 0, 9, 10], 11: [1, 0, 9, 11], 12: [1, 0, 9, 12], 13: [1, 0, 9, 13], 6: [1, 0, 8, 5, 6],
	    7: [1, 0, 8, 5, 7], 15: [1, 0, 9, 13, 15], 14: [1, 0, 9, 12, 14]},
	 ...
	}
	'''

	# nx.write_graphml(G, "1656633600_lngraph_node_centrality_incl_endpoints.graphml", infer_numeric_types=False, named_key_ids=True)

	'''
	SETTING UP BETEENNESS CENTRALITY IN NODES
	print(f"[DEBUG] Initializing betweenness centrality in nodes...")
	t_after_betw = time.time()
	bc = nx.betweenness_centrality(G, normalized=False, weight="weight")
	nx.set_node_attributes(G, bc, "betweenness")
	t_after_betw = time.time()
	print(f"[DEBUG] Done ({int(10 * (t_after_betw - t_before_betw)) / 10 / 60}min)")

	nx.write_graphml(G, "1656633600_lngraph_betw.graphml", infer_numeric_types=False, named_key_ids=True)

	nodes_by_betw = sorted(G.nodes(), key=lambda n: G.nodes[n]["betweenness"], reverse=True)
	rh.print_results_cumu(G, nodes_by_betw, top_n=30)
	'''


def test():
	if len(sys.argv) != 2:
		print("Usage: ./pathfinding.py <GRAPHML FILE>")
		exit()
	activate_log()
	# print("[DEBUG] Loading graphml data and initializing CLN defaults...")
	t_before_graphml = time.time()
	G = u.get_graph("graphml")  # for a list of possible options check utils.py

	init_cln_defaults(G)

	t_before_singlepathing_colluding = time.time()
	# print(f"[DEBUG] Done loading graphml data and initializing CLN defaults "
	#      f"({int(10 * (t_before_singlepathing_colluding - t_before_graphml)) / 10}s)\n")
	print(f"[DEBUG] Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}\n")


	u.pgfplot_node_degree(G)
	u.pgfplot_base_fee(G)
	u.pgfplot_prop_fee(G)
	u.pgfplot_max_payment_per_htlc(G)
	u.pgfplot_channel_delay(G)

	'''
	print("[DEBUG] Executing colluding Route Hijacking attack for singlepathing transactions...")
	scale = 1
	share_from = 0
	share_to = 1
	all_spaths_dict = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))
	rh.spp_colluding_attack(G, AMOUNT, scale, share_from, share_to, all_spaths_dict)
	# num_long_paths = rh.colluding_attack(G, AMOUNT, all_spaths_dict, mpp=False)
	t_before_multipathing_colluding = time.time()
	print(f"[DEBUG] Done executing colluding Route Hijacking attack for singlepathing transactions "
	      f"({int(10 * (t_before_multipathing_colluding - t_before_singlepathing_colluding)) / 10 / 60}min)\n")

	u.pgfplot_first_strongest_nodes(G, top_n=30)
	u.pgfplot_disconnected_nodes(G, top_n=30)
	u.pgfplot_blocked_funds(G, top_n=30, amount=AMOUNT)

	t_end = time.time()
	print(f"\n\n[DEBUG] Total run time: ({int(10 * (t_end - t_before_graphml)) / 10 / 60}min)\n")
	return 1
	'''
	
	num_nodes = G.number_of_nodes()
	scale = 2 / num_nodes
	share_from = 0.9
	share_to = 1
	node_pairs = u.get_node_pairs(G, scale, share_from, share_to)
	# print("[DEBUG] Execute colluding Route Hijacking attack for multipathing transactions...")
	t_before_colluding_attack = time.time()
	rh.mpp_colluding_attack(G, AMOUNT, scale, share_from, share_to)

	if scale < 1:
		extrapolate_mpp_results(G, scale)

	# rh.colluding_attack(G, AMOUNT, all_spaths_dict, mpp=True)
	t_stop = time.time()
	# print(f"[DEBUG] Done executing colluding Route Hijacking attack for multipathing transactions "
	#       f"({int(10 * (t_stop - t_before_colluding_attack)) / 10 / 60}min)\n")

	u.pgfplot_first_strongest_nodes(G, top_n=30)
	u.pgfplot_disconnected_nodes(G, top_n=30)
	u.pgfplot_blocked_funds(G, top_n=30, amount=AMOUNT)

	# u.visualize_graph_and_print_simple_paths(G)

	t_end = time.time()
	print(f"[DEBUG] Total run time: ({int(10 * (t_end - t_before_graphml)) / 10 / 60}min)\n")


if __name__ == "__main__":
	# main()
	test()
