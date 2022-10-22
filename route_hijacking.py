import networkx as nx
import itertools as it
import utils as u
import time
import pathfinding as pf

TOTAL_NUMBER_OF_PATHS = 0

'''
By probing the capacity from a well-connected vantage point in the network we found that the 80th percentile of
capacities is >= 9765 sats.

Rounding to 10.000 sats = 10.000.000 msats = 1e7 msats per part, there is a ~80% chance that the payment will go
through without requiring further splitting. The fuzzing is symmetric and uniformy distributed around this value,
so this should not change the success rate much. For the remaining 20% of payments we might require a split to make
the parts succeed, so we try only a limited number of times before we split adaptively.

Notice that these numbers are based on a worst case assumption that payments from any node to any other node are
equally likely, which isn't really the case, so this is likely a lower bound on the success rate.

As the network evolves these numbers are also likely to change.
'''
MPP_TARGET_SIZE_MSAT = 10 * 1000 * 1000

'''
PRESPLIT_MAX_SPLITS defines how many parts do we split into before we increase the bucket size. This is a tradeoff
between the number of payments whose parts are identical and the number of concurrent HTLCs. The larger this amount
the more HTLCs we may end up starting, but the more payments result in the same part sizes.*/
'''
PRESPLIT_MAX_SPLITS = 16

'''
Number of concurrent HTLCs one channel can handle in each direction. Should be between 1 and 483 (default 30).
reference for default MAX_CONCURRENT_HTLCS value:
https://github.com/ElementsProject/lightning/blob/8c9fa457babd8ac09009fb93fe7a1a6409aba911/lightningd/options.c#L781-L835
'''
MAX_CONCURRENT_HTLCS = 30

'''
If applied trivially this splitter may end up creating more splits than the sum of all channels can support,
i.e. each split results in an HTLC, and each channel has an upper limit on the number of HTLCs it'll allow us to add.
If the initial split would result in more than 1/3rd of the total available HTLCs we clamp the number of splits to
1/3rd. We don't use 3/3rds in order to retain flexibility in the adaptive splitter.
'''
PRESPLIT_MAX_HTLC_SHARE = 3


# reference for majority of the following multipathing code:
# https://github.com/ElementsProject/lightning/blob/1da9b30b9abd26e9861ae199c2754f3d9cf7249f/plugins/libplugin-pay.c#L3384-L3788


def payment_supports_mpp(amt):
	return True
	# return amt > MPP_TARGET_SIZE_MSAT


def payment_max_htlcs(path):
	return MAX_CONCURRENT_HTLCS


def presplit_cb(amount, max_num_parts=PRESPLIT_MAX_SPLITS):
	if not payment_supports_mpp(amount):
		print("Multipathing is not supported with this payment.")
		return -1
	parts = []
	target_amount = MPP_TARGET_SIZE_MSAT  # 10*1000*1000 millisatoshis = 10.000 Satoshis

	'''
	We aim for at most PRESPLIT_MAX_SPLITS parts, even for large values. To achieve this we take the base amount and
	multiply it by the number of targetted parts until the total amount divided by part amount gives us at most that
	number of parts.
	'''
	while (target_amount * max_num_parts) < amount:
		target_amount *= max_num_parts

	while amount > target_amount:
		parts.append(target_amount)
		amount -= target_amount
	parts.append(amount)

	return parts


def singlepath_transaction(G, spath, amount):
	incr_node_counters_on_path(G, spath, amount, spath.copy(), endpoints=False)


def sufficient_htlc_max_limit(G, path, amounts):
	success = True
	# There is no bigger value in the list of sub-payment amounts than the first so it is sufficient to compare
	# this value with the all HTLC max values
	amt = amounts[0]
	edges = u.get_edges(G, path)

	for e in edges:
		success &= amt < u.get_edge_htlc_max(e)

	return success


def check_htlc_budget(G, path, num_htlcs):
	edges = u.get_edges(G, path)
	for e in edges:
		available_num_htlcs = u.get_edge_htlc_budget(e) / PRESPLIT_MAX_HTLC_SHARE

		if num_htlcs > available_num_htlcs:
			return available_num_htlcs

	return 0


'''
@path is a list of node IDs that define a path. For singlepathing @path and @on_path_nodes is identical.
For multipathing @on_path_nodes contains all node IDs of nodes that are part of every payment part's path (excluding
endpoints). Connectivity counts the number of paths that connect source and target nodes in a multipathing transaction.
By this definition the connectivity of node n is the number of node pairs that were disconnected in case n would turn
unresponsive.  
'''
def incr_node_counters_on_path(G, path, amount, on_path_nodes, endpoints=False):
	# Disregard source and target nodes by default since they do not perform payment griefing

	# print(f"\tincr_node_counters on path {path}, on_path_nodes: {on_path_nodes}")
	if endpoints:
		pc = 1 / len(path)
		for node_id in path:
			G.nodes[node_id]["node_centrality"] += pc
			G.nodes[node_id]["inbound_funds"] += path.index(node_id) * amount
			G.nodes[node_id]["inbound_path_lenght"] += path.index(node_id)
			if node_id in on_path_nodes:
				G.nodes[node_id]["connectivity"] += 1
				on_path_nodes.remove(node_id)

	else:
		if len(path) > 2:
			pc = 1 / (len(path) - 2)
			for node_id in path[1:-1]:
				G.nodes[node_id]["node_centrality"] += pc
				G.nodes[node_id]["inbound_funds"] += path.index(node_id) * amount
				G.nodes[node_id]["inbound_path_lenght"] += path.index(node_id)
				# print(f"\t\tnode_id: {node_id}, on_path_nodes: {on_path_nodes}")
				if node_id in on_path_nodes:
					# print(f"\t\t\t connectivity of node_id: {node_id} incremented")
					G.nodes[node_id]["connectivity"] += 1
					on_path_nodes.remove(node_id)


def decr_htlc_budget_on_path(G, path, amount):
	edges = u.get_edges(G, path)

	for e in edges:
		u.decr_htlc_budget(e, amount=amount)


'''
Required to check if a mpp transaction is successful:
1. All on-path channels have a sufficient HTLC budget
	That means the number of HTLCs required for this mpp transaction is not higher than 1/PRESPLIT_MAX_SPLITS of
	the remaining limit of concurrent HTLCs in this channel (1/3 by CLN default)
2. The payment size limit of all on-path channel HTLCs is sufficiently high to support all partial payments
'''
def multipath_transaction(G, src, tgt, amount, spent_fees, max_num_parts=PRESPLIT_MAX_SPLITS):
	parts = presplit_cb(amount, max_num_parts)
	i = 0
	num_parts = len(parts)
	spaths = []
	nodes = list(G.nodes)
	num_nodes = len(nodes)
	print(f"mpp transaction between node pair [{src[:4]}, {tgt[:4]}]")
	if G.degree[src] > 0:
		while True:
			try:
				spaths_dict = nx.shortest_simple_paths(G, src, tgt, weight="weight", k=num_parts)
				for spath in spaths_dict:
					# print(f"\tsrc={src} to tgt={tgt}, path: {spath}")
					spaths.append(spath)
				# i += 1

				# if i >= num_parts:
				# 	break

				'''
				for path in spaths:
					htlcs = check_htlc_budget(G, path, num_htlcs=num_parts)
	
					if htlcs > 0:
						print("ERROR - Insufficient htlc budget")
						return False, htlcs
	
					if not sufficient_htlc_max_limit(G, path, parts):
						print("ERROR - Insufficient htlc max limit")
						return False, 0
				'''

				on_path_nodes = []
				# print(f"\tspaths: {spaths}")
				for path in spaths:
					# print(f"\t\tpath: {path}")
					if len(path) > 2:
						for node_id in path[1:-1]:
							if node_id not in on_path_nodes:
								on_path_nodes.append(node_id)
				# print(f"\t\ton_path_nodes after adding new node_ids: {on_path_nodes}")

				# Arriving at this line indicates transaction success
				for i in range(len(spaths)):
					path = spaths[i]
					part_amount = parts[i]

					fees = u.get_spent_base_fees(G, path, part_amount)
					# print(f"\t\tFees added to spent_fees (mpp) for path {path}: {fees}")
					spent_fees.append((len(path) - 1, fees))

					incr_node_counters_on_path(G, path, part_amount, on_path_nodes, endpoints=False)
					decr_htlc_budget_on_path(G, path, amount=num_parts)
				return True, 0
			except nx.NetworkXNoPath:
				# print(f"No path from {src[:4]} to {tgt[:4]}")
				i = nodes.index(tgt)
				tgt = nodes[(i + 1) % num_nodes]
				continue


def spp_colluding_attack(G, amount, scale, share_from, share_to, all_spaths_dict):
	spent_fees = []
	num_spath_dicts = len(all_spaths_dict)
	i = 0
	t_start = time.time()

	for _, spaths_dict in all_spaths_dict.items():
		t_start = time.time()

		for _, spath in spaths_dict.items():
			length = len(spath) - 1

			singlepath_transaction(G, spath, amount)
			fees = u.get_spent_base_fees(G, spath, amount)
			spent_fees.append((length, fees))

		t_stop = time.time()
		t_delta = t_stop - t_start
		i += 1
		print(f"[DEBUG] \t{i}/{num_spath_dicts} nodes processed. This node time: {int(t_delta / 60)} mins"
		      f"Total estimated time: {int(t_delta * num_spath_dicts / 60)} mins")

	# u.pgfplot_avg_spent_fees(G, spent_fees)
	u.pgfplot_shortest_paths_length_distribution(spent_fees)
	u.pgfplot_shortest_paths_cost(spent_fees)

	return 1


def mpp_colluding_attack(G, amount, scale, share_from, share_to):
	failed_payments = 0
	node_pairs = u.get_node_pairs(G, scale, share_from, share_to)
	# num_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
	num_pairs = len(node_pairs)
	print(f"number of pairs: {num_pairs}")
	spent_fees = []
	j = 0

	t_start = time.time()
	for pair in node_pairs:
		# src, tgt = u.get_node_id(pair[0]), u.get_node_id(pair[1])
		src, tgt = pair[0], pair[1]
		payment_successful, max_num_parts = multipath_transaction(G, src, tgt, amount, spent_fees)

		'''
		if not payment_successful:
			if max_num_parts > 0:
				# Transaction might be successful with fewer parts.
				# So try again, this time including the determined max number of parts
				payment_successful, max_num_parts = multipath_transaction(G, src, tgt, amount, spent_fees)
				if not payment_successful:
					# If the second pass fails, multipathing is apparently not possible for this transaction
					failed_payments += 1
			else:
				# Transaction failed due to insufficient HTLC payment size limit, no additional passes performed
				failed_payments += 1
		'''

		j += 1
		print(j)
		if j > int(num_pairs * 0.01):
			t_stop = time.time()
			t_delta = t_stop - t_start
			# print(f"mpp colluding attack - {j}/{num_pairs} pairs processed. "
			#       f"{t_delta} sec for this amount of pairs. Total estimate: {int(t_delta * 100 / 60)} min")
			j = num_pairs

	u.pgfplot_shortest_paths_length_distribution(spent_fees)
	u.pgfplot_shortest_paths_cost(spent_fees)


# This can take some time - about 10 minutes for 9k nodes in singlepathing mode
def colluding_attack(G, amount, all_spaths_dict, mpp=False):
	num_long_paths = 0
	spent_fees = []
	failed_payments = 0
	u.set_node_centrality(G, 0)
	u.set_htlc_budet(G, 0)

	'''
	dict structure:
	{src_n_id0: {tgt_n_id0: spath, tgt_n_id1: spath, ...},
	 src_n_id1: ...}
	Sample:
	{0: {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4], 8: [0, 8], 9: [0, 9], 5: [0, 8, 5], 10: [0, 9, 10]},
	 1: {...},
	 ...
	}
	'''
	num_spath_dicts = len(all_spaths_dict)
	i = 0

	for _, spaths_dict in all_spaths_dict.items():
		t_start = time.time()

		for _, spath in spaths_dict.items():
			length = len(spath) - 1

			if length > 0:
				num_long_paths += 1

				if mpp:
					src, tgt = spath[0], spath[-1]
					payment_successful, max_num_parts = multipath_transaction(G, src, tgt, amount, spent_fees)

					if not payment_successful:

						if max_num_parts > 0:
							# Transaction might be successful with fewer parts.
							# So try again, this time including the determined max number of parts
							payment_successful, max_num_parts = multipath_transaction(G, src, tgt, amount, spent_fees)

							if not payment_successful:
								# If the second pass fails, multipathing is apparently not possible for this transaction
								failed_payments += 1

						else:
							# Transaction failed due to insufficient HTLC payment size limit, no additional passes performed
							failed_payments += 1

				else:
					singlepath_transaction(G, spath, amount)
					fees = u.get_spent_base_fees(G, spath, amount)
					spent_fees.append((length, fees))

		t_stop = time.time()
		t_delta = t_stop - t_start
		i += 1
		print(f"[DEBUG] \t{i}/{num_spath_dicts} nodes processed. This node time: {int(t_delta / 60)} mins"
		      f"Total estimated time: {int(t_delta * num_spath_dicts / 60)} mins")
		# TODO: If htlc_maximum_msat(c) < amount for one c \in spath, incr. payment failure counter
		# TODO: If htlc_budget(c) < num_payments, incr. payment failure counter
		# TODO: Log all that properly for result section in the thesis!

	# u.pgfplot_avg_spent_fees(G, spent_fees)
	u.pgfplot_shortest_paths_length_distribution(spent_fees)
	u.pgfplot_shortest_paths_cost(spent_fees)


	'''
	For measurements of
		"amounts of disconnected nodes after top_n nodes turn unresponsive" (= pgfplot_first_strongest_nodes()?) and
		"amount of founds locked in HTLCs of top_n node's channels after top_n nodes turn unresponsive"
	remove top_n nodes and all corresponding channels from the graph and repeat the big nested for loop.
	
	'''

	'''
	for node_id, spaths_dict in all_spaths_dict.items():
		print(f"node_id: {node_id} (node_centrality: {rh.get_node_centrality(G, node_id)})")
		for _, spath in spaths_dict.items():
			print(f"\tshortest path: {spath}")
	'''

	total_node_centralitys = 0

	for node_id in G:
		total_node_centralitys += u.get_node_centrality(G, node_id)
	n = G.number_of_nodes()
	print(f"[DEBUG] \tNumber of node pairs for {n} nodes: {n * (n - 1)}")
	print(f"[DEBUG] \tNumber of node pairs that are connected by paths longer than 2: {num_long_paths}")
	print(f"[DEBUG] \tSum of all node centralities: {total_node_centralitys}")

	return num_long_paths

# rh.print_results_cumu(G, top_n=30)
