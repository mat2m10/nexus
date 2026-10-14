import numpy as np

from bisect import bisect_left, insort


def simulate_tree(k, n_per_deme, Ne, M, neighbors, rng):
    """
    Simulate a single coalescent tree on a k×k stepping-stone grid.
    Returns:
        nodes      : list of (time, left, right) for each internal node
                     left/right are indices into the full node list
        n_samples  : number of leaf nodes (first n_samples entries)

    Performance note: instead of scanning every lineage with ``np.where`` on
    each event, a sorted list of active lineage slots is kept per deme. The
    random draws are made with ``rng.choice(L)`` on the candidate count and
    mapped to the L-th smallest slot, which consumes the generator exactly like
    the former ``rng.choice(candidates)`` on the ascending candidate array, so
    output for a fixed seed is unchanged.
    """
    n_demes  = k * k
    n_tot    = n_demes * n_per_deme

    # Each lineage: which deme is it currently in?
    # Lineages 0..n_tot-1 are leaves; internal nodes appended as coalescences happen
    pop_list = np.repeat(np.arange(n_demes), n_per_deme)  # deme of each lineage
    k_list   = np.arange(n_tot, dtype=int)                 # node id of each lineage
    n        = np.array([n_per_deme] * n_demes, dtype=float)  # lineage count per deme

    # Active lineage slots per deme, kept sorted (same order as np.where gave)
    slots = [list(range(m * n_per_deme, (m + 1) * n_per_deme)) for m in range(n_demes)]
    n_nbrs = np.array([len(neighbors[m]) for m in range(n_demes)])

    # Node records: (time, daughter1, daughter2)
    # Leaves have time=0, daughters=-1
    node_times = np.zeros(2 * n_tot - 1)
    node_d1    = np.full(2 * n_tot - 1, -1, dtype=int)
    node_d2    = np.full(2 * n_tot - 1, -1, dtype=int)

    t            = 0.0
    current_node = n_tot       # next internal node id
    n_sum        = n_tot       # total active lineages

    mig_rate_per_lineage = M / (2 * Ne)

    while n_sum > 1:
        # ── Rates ──
        rate_co  = n * (n - 1) / (2 * Ne)          # coalescence rate per deme
        rate_mig = mig_rate_per_lineage * n * n_nbrs

        rate_co_tot  = rate_co.sum()
        rate_mig_tot = rate_mig.sum()
        rate_tot     = rate_co_tot + rate_mig_tot

        # ── Time to next event ──
        t += rng.exponential(1.0 / rate_tot)

        # ── Migration or coalescence? ──
        if rng.random() < rate_mig_tot / rate_tot:
            # Migration: pick deme, pick lineage, move to random neighbor
            probs = rate_mig / rate_mig_tot
            src   = int(rng.choice(n_demes, p=probs))
            cands = slots[src]
            i     = cands.pop(int(rng.choice(len(cands))))
            dst   = rng.choice(neighbors[src])
            n[src] -= 1
            n[dst] += 1
            pop_list[i] = dst
            insort(slots[dst], i)

        else:
            # Coalescence: pick deme, pick 2 lineages, merge
            probs    = rate_co / rate_co_tot
            which    = int(rng.choice(n_demes, p=probs))
            cands    = slots[which]
            ia, ib   = rng.choice(len(cands), size=2, replace=False)
            a, b     = cands[ia], cands[ib]
            lo, hi   = min(a, b), max(a, b)

            # Record internal node
            node_times[current_node] = t
            node_d1[current_node]    = k_list[a]
            node_d2[current_node]    = k_list[b]

            # Replace one lineage with new node, remove the other
            k_list[lo]   = current_node
            pop_list[lo] = which
            del cands[bisect_left(cands, hi)]
            last = n_sum - 1
            if last != hi:
                d = pop_list[last]
                lst = slots[d]
                del lst[bisect_left(lst, last)]
                insort(lst, hi)
            k_list[hi]   = k_list[last]
            pop_list[hi] = pop_list[last]

            n[which]      -= 1
            n_sum         -= 1
            current_node  += 1

    return node_times, node_d1, node_d2, n_tot

def drop_mutations(node_times, node_d1, node_d2, n_tot, n_mutations, rng):
    """
    Drop n_mutations onto the tree proportional to branch length.
    For each mutation, find which node it lands on and return
    the genotype vector (1=carries mutation, 0=does not).
    
    Returns:
        genotypes : (n_mutations, n_tot) binary array
    """
    n_nodes = 2 * n_tot - 1

    # Compute branch length for each node
    # = time of parent - time of node
    # Root has no parent → branch length 0
    branch_lengths = np.zeros(n_nodes)
    internal = np.arange(n_tot, n_nodes)
    d1 = node_d1[internal]
    d2 = node_d2[internal]
    branch_lengths[d1] = node_times[internal] - node_times[d1]
    branch_lengths[d2] = node_times[internal] - node_times[d2]

    # Sample mutation positions proportional to branch length
    total_bl = branch_lengths.sum()
    if total_bl == 0:
        return np.zeros((n_mutations, n_tot), dtype=np.int8)

    probs      = branch_lengths / total_bl
    mut_nodes  = rng.choice(n_nodes, size=n_mutations, p=probs)

    # For each mutation, set all leaf descendants (a contiguous block in a
    # depth-first leaf ordering, computed once per tree)
    order, start, count = _leaf_order(node_d1, node_d2, n_tot)
    genotypes = np.zeros((n_mutations, n_tot), dtype=np.int8)
    for mi, mut_node in enumerate(mut_nodes):
        s = start[mut_node]
        genotypes[mi, order[s:s + count[mut_node]]] = 1

    return genotypes


def _leaf_order(node_d1, node_d2, n_tot):
    """Depth-first leaf ordering of the tree.

    Returns (order, start, count): the leaves under node ``v`` are exactly
    ``order[start[v]:start[v] + count[v]]``. Daughters always have smaller ids
    than their parent, so one ascending and one descending pass suffice.
    """
    n_nodes = 2 * n_tot - 1
    d1 = node_d1.tolist()
    d2 = node_d2.tolist()
    count = [1] * n_tot + [0] * (n_nodes - n_tot)
    for v in range(n_tot, n_nodes):
        count[v] = count[d1[v]] + count[d2[v]]
    start = [0] * n_nodes
    for v in range(n_nodes - 1, n_tot - 1, -1):
        s = start[v]
        start[d1[v]] = s
        start[d2[v]] = s + count[d1[v]]
    order = np.empty(n_tot, dtype=int)
    order[np.asarray(start[:n_tot], dtype=int)] = np.arange(n_tot)
    return order, start, count


def get_descendants(node, node_d1, node_d2, n_tot):
    """
    Return binary vector: 1 for all leaf descendants of node, 0 otherwise.
    Mirrors fill_in_genotypes() from kpop.c
    """
    gt = np.zeros(n_tot, dtype=np.int8)
    stack = [node]
    while stack:
        curr = stack.pop()
        if curr < n_tot:
            gt[curr] = 1          # it's a leaf
        else:
            stack.append(node_d1[curr])
            stack.append(node_d2[curr])
    return gt

def build_ancestral_genotype(n_loci, k, n_per_deme, Ne, M, neighbors, rng, n_mutations):
    rows = []
    for i in range(n_loci):
        if i % 50 == 0:
            print(f"  locus {i}/{n_loci}...")
    
        # Simulate one coalescent tree
        node_times, node_d1, node_d2, n_tot = simulate_tree(
            k, n_per_deme, Ne, M, neighbors, rng
        )
    
        # Drop mutations onto the tree
        geno = drop_mutations(
            node_times, node_d1, node_d2, n_tot, n_mutations, rng
        )
    
        rows.append(geno)
    
    # Stack into full genotype matrix
    G = np.vstack(rows)   # shape: (n_loci * n_mutations, n_samples)
    return G