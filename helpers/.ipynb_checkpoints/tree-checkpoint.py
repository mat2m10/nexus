import numpy as np

def simulate_tree(k, n_per_deme, Ne, M, neighbors, rng):
    """
    Simulate a single coalescent tree on a k×k stepping-stone grid.
    Returns:
        nodes      : list of (time, left, right) for each internal node
                     left/right are indices into the full node list
        n_samples  : number of leaf nodes (first n_samples entries)
    """
    n_demes  = k * k
    n_tot    = n_demes * n_per_deme

    # Each lineage: which deme is it currently in?
    # Lineages 0..n_tot-1 are leaves; internal nodes appended as coalescences happen
    pop_list = np.repeat(np.arange(n_demes), n_per_deme)  # deme of each lineage
    k_list   = np.arange(n_tot, dtype=int)                 # node id of each lineage
    n        = np.array([n_per_deme] * n_demes, dtype=float)  # lineage count per deme

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
        rate_mig = mig_rate_per_lineage * n * np.array([len(neighbors[m]) for m in range(n_demes)])

        rate_co_tot  = rate_co.sum()
        rate_mig_tot = rate_mig.sum()
        rate_tot     = rate_co_tot + rate_mig_tot

        # ── Time to next event ──
        t += rng.exponential(1.0 / rate_tot)

        # ── Migration or coalescence? ──
        if rng.random() < rate_mig_tot / rate_tot:
            # Migration: pick deme, pick lineage, move to random neighbor
            probs = rate_mig / rate_mig_tot
            src   = rng.choice(n_demes, p=probs)
            candidates = np.where(pop_list[:n_sum] == src)[0]
            i     = rng.choice(candidates)
            dst   = rng.choice(neighbors[src])
            n[pop_list[i]] -= 1
            n[dst]         += 1
            pop_list[i]     = dst

        else:
            # Coalescence: pick deme, pick 2 lineages, merge
            probs    = rate_co / rate_co_tot
            which    = rng.choice(n_demes, p=probs)
            cands    = np.where(pop_list[:n_sum] == which)[0]
            pair     = rng.choice(cands, size=2, replace=False)
            a, b     = pair[0], pair[1]

            # Record internal node
            node_times[current_node] = t
            node_d1[current_node]    = k_list[a]
            node_d2[current_node]    = k_list[b]

            # Replace one lineage with new node, remove the other
            k_list[min(a,b)]   = current_node
            pop_list[min(a,b)] = which
            k_list[max(a,b)]   = k_list[n_sum-1]
            pop_list[max(a,b)] = pop_list[n_sum-1]

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
    for i in range(n_tot, n_nodes):          # loop over internal nodes
        d1 = node_d1[i]
        d2 = node_d2[i]
        branch_lengths[d1] = node_times[i] - node_times[d1]
        branch_lengths[d2] = node_times[i] - node_times[d2]

    # Sample mutation positions proportional to branch length
    total_bl = branch_lengths.sum()
    if total_bl == 0:
        return np.zeros((n_mutations, n_tot), dtype=np.int8)

    probs      = branch_lengths / total_bl
    mut_nodes  = rng.choice(n_nodes, size=n_mutations, p=probs)

    # For each mutation, find all leaf descendants
    genotypes = np.zeros((n_mutations, n_tot), dtype=np.int8)
    for mi, mut_node in enumerate(mut_nodes):
        genotypes[mi] = get_descendants(mut_node, node_d1, node_d2, n_tot)

    return genotypes


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