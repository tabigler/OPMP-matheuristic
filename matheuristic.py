# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:56:25 2021

@author: tamara.bigler
"""

# Import standard packages
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from scipy.spatial import distance_matrix

# Import package for Gurobi solver
import gurobipy as gb


def initial_solutions_kmeans_plus_plus_random(instance, p, start_time,
                                              num_jobs=8, num_random=2):
    """Generates initial solution using kmeans++ algorithm and random generation
    (only for instances for which coordinates of facilities (and clients) are available"""

    # Set seed
    np.random.seed(11)

    # Read coordinates of facilities
    df_coordinates_f = pd.read_csv('data/' + instance + '_coord_f.csv')

    # Compute distance matrix between facilities (rounded to integer values)
    distances_f = pd.DataFrame(distance_matrix(df_coordinates_f[['coordinate1', 'coordinate2']],
                                                df_coordinates_f[['coordinate1', 'coordinate2']]).round(0))

    # Initialize empty list for initial solutions
    solutions = []

    # Get indices of facilities selected by kmeans++
    centers, indices = kmeans_plusplus(df_coordinates_f[['coordinate1', 'coordinate2']].values,
                                       n_clusters=num_jobs-num_random,
                                       random_state=11)

    # Derive start solutions based on kmeans++ algorithm
    for l in range(num_jobs-num_random):
        solutions.append(set(distances_f.loc[indices[l], :].sort_values().iloc[:p].index))

    # Select random facilities
    for l in range(num_random):
        solutions.append(set(np.random.choice(df_coordinates_f.index, size=p,
                                              replace=False)))

    # Stop time
    hit_time = time.time() - start_time

    return solutions, hit_time, df_coordinates_f, distances_f


def initial_solutions(instance, p, start_time, num_jobs=8, num_random=2,
                      rule_based_solution1=True, rule_based_solution2=True):
    """Generates initial solution when coordinates of clients and facilities are not available"""

    # Read distance matrix
    df = pd.read_csv('data/' + instance + '.csv', index_col=0)
    num_faci = df.shape[1]
    df = df.reset_index(drop=True)
    df.columns = range(num_faci)

    # Initialize empty list for initial solutions
    solutions = []

    if rule_based_solution1==True:
        # Facilities with the largest minimum distance to a client
        min_distances = df.min(axis=0)
        min_distances_sorted = min_distances.sort_values()
        solution = set(min_distances_sorted.iloc[-p:].index)
        solutions.append(solution)

    if rule_based_solution2==True:
        # Facilities with the largest sum of distance to all clients
        sum_distances = df.sum(axis=0)
        sum_distances_sorted = sum_distances.sort_values()
        solution = set(sum_distances_sorted.iloc[-p:].index)
        solutions.append(solution)

    # Set seed number
    seed = 1
    if num_jobs - num_random - int(rule_based_solution1) - int(rule_based_solution2) > 0:
        for l in range(num_jobs - num_random -  - int(rule_based_solution1)
                       - int(rule_based_solution2)):
            # Set seed
            np.random.seed(seed)
            # Select random client
            random_client = np.random.choice(df.index, size=1, replace=False)[0]
            # First solution
            # Select p facilities that are nearest to this client
            distances_sorted = df.loc[random_client, :].sort_values()
            solution = set(distances_sorted.iloc[:p].index)
            solutions.append(solution)
            # Second solution
            # Select facility that is furthest from this client
            distances_sorted = df.loc[random_client, :].sort_values()
            furthest_facility = distances_sorted.iloc[-1:].index[0]
            # Select nearest client to furthest facility
            nearest_client = df.loc[:, furthest_facility].sort_values().iloc[:1].index[0]
            # Select p nearest facilities to client
            distances_sorted = df.loc[nearest_client, :].sort_values()
            solution = set(distances_sorted.iloc[:p].index)
            solutions.append(solution)
            # Set seed for next iteration
            seed += 1

        # Select random facilities
        for l in range(num_random):
            np.random.seed(seed)
            solution = set(np.random.choice(df.columns, size=p, replace=False))
            solutions.append(solution)
            # Set seed for next iteration
            seed += 1

    # Stop time
    hit_time = time.time() - start_time

    return solutions, hit_time, df


def compute_distance_matrix(df_coordinates_c, df_coordinates_f):
    """Computes distance matrix between clients and facilities"""

    # Compute distance matrix (rounded to integers)
    df = pd.DataFrame(distance_matrix(df_coordinates_c[['coordinate1', 'coordinate2']],
                                      df_coordinates_f[['coordinate1', 'coordinate2']]).round(0))
    return df


def remove_facilities(df, p, b, num_cust, num_faci, current_solution,
                      partial_solution_old, time_limit, start_time):
    """Removes b facilities from solution"""

    # Store current solution
    current_solution = np.array(list(current_solution))

    # Get distance matrix
    distances = df.values

    if b == p:
        partial_solution = np.array([], dtype=int)
        return partial_solution, current_solution

    # Determine sets and facility sequence S
    I = np.arange(num_cust)
    J = current_solution
    S = {i: J[np.argsort(distances[i, J])][:b + 1] for i in I}

    # Compute upper bounds on distances
    ub = {i: np.sort(distances[i, J])[b] for i in I}

    # Create model
    m = gb.Model()

    # Add decision variables
    y = m.addVars(J, vtype=gb.GRB.BINARY, name='y')
    D = m.addVars(I, obj=1, ub=ub, name='D')
    vars = []
    counters = {j: 0 for j in J}
    for i in I:
        for j in S[i][:b]:
            vars.append((i, j))
            counters[j] += 1
    v = m.addVars(vars, vtype=gb.GRB.BINARY, name='v')

    # Provide warm start
    sum_distances = df.loc[:, J].sum(axis=0)
    sum_distances_sorted = sum_distances.sort_values()
    J_start = set(sum_distances_sorted.iloc[-(len(J)-b):].index)
    for j in J_start:
        y[j].start = 1.0

    # Set model sense
    m.setAttr('ModelSense', -1)

    # Add constraints
    m.addConstr(y.sum() == p - b)
    m.addConstrs(D[i] <= distances[i, S[i][-1]] 
                 - gb.quicksum((distances[i, S[i][k + 1]]
                                - distances[i, S[i][k]]) * v[i, S[i][k]]
                               for k in range(len(S[i]) - 1)) for i in I)
    m.addConstrs(v[i, S[i][k]] <= v[i, S[i][k + 1]] for i in I
                 for k in range(len(S[i]) - 2))
    m.addConstrs(v.sum('*', j) >= counters[j] * y[j] for j in J)

    # Specify Gurobi options
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', max(time_limit - (time.time() - start_time), 5))

    # Update model
    m.update()

    # Optimize
    m.optimize()

    # Retrieve values of decision variables
    nSolutions = m.SolCount
    if nSolutions >= 1:
        partial_solution = np.array([j for j in J if y[j].X > 0.5], dtype=int)
    else:
        partial_solution = partial_solution_old

    # Return information on partial solution
    return set(partial_solution), current_solution


def add_facilities(df, b, num_cust, num_faci, partial_solution, current_solution,
                   start_time, ofv_best, solution_best, hit_time, time_limit,
                   m_max, coord_avail=True, distances_to_center=None,
                   distances_f=None):
    """Adds b facilities to partial solution"""

    # Store partial solution
    partial_solution = np.array(list(partial_solution), dtype=int)
    current_solution = np.array(list(current_solution), dtype=int)

    # Get distance matrix
    distances = df.values

    # Determine sets
    I = np.arange(num_cust)
    J = np.setdiff1d(np.arange(num_faci), partial_solution)

    # Derive set of considered facilities
    if len(J) > m_max:
        # Coordinates of clients and facilities not available
        if coord_avail == False:
            J = np.random.choice(J, size=max(m_max, b), replace=False)
            J = np.array(list(J), dtype=int)

        # Coordinates of clients and facilities available
        if coord_avail == True:
            # Derive nearest facilities to facilities that are open in partial solution
            nearest_facilities = []
            J1 = np.setdiff1d(current_solution, partial_solution)
            J = np.setdiff1d(J, J1)
            num_fac_cons = max(m_max - len(J1), b)
            if len(partial_solution) > 0:
                for j in partial_solution:
                    nearest_facilities.append(
                        set(distances_f.loc[j, J].sort_values().iloc[:num_fac_cons].index))
            else:
                for j in current_solution:
                    nearest_facilities.append(
                        set(distances_f.loc[j, J].sort_values().iloc[:num_fac_cons].index))
            nf_union = set().union(*nearest_facilities)
            nf_union = list(nf_union)
            # Apply distance-based priority rule to select facilities into set of considered facilities
            d_sum = distances_to_center[nf_union].sum()
            d_prob = distances_to_center[nf_union] / d_sum
            J = np.random.choice(nf_union, size=num_fac_cons, replace=False,
                                 p=d_prob)
            J = set(J).union(J1)
            J = np.array(list(J), dtype=int)

    # Compute sets and parameters
    ub = {}
    S = {}
    vars = []
    counters = {j: 0 for j in J}
    for i in I:
        if len(partial_solution) == 0:
            ub[i] = np.sort(distances[i, :])[-b]
        else:
            ub[i] = distances[i, partial_solution].min()
        sequence = J[np.argsort(distances[i, J])]
        S[i] = [j for j in sequence if distances[i, j] < ub[i]]
        for j in S[i]:
            vars.append((i, j))
            counters[j] += 1

    # Create model
    m = gb.Model()

    # Add decision variables
    y = m.addVars(J, vtype=gb.GRB.BINARY)
    D = m.addVars(I, obj=1, ub=ub)
    v = m.addVars(vars, vtype=gb.GRB.BINARY)

    # Set model sense
    m.setAttr('ModelSense', -1)

    # Add constraints
    m.addConstr(y.sum() == b)
    m.addConstrs(D[i] <= ub[i] - (ub[i] - distances[i, S[i][-1]]) * v[i, S[i][-1]]
                 - gb.quicksum((distances[i, S[i][k + 1]] - distances[i, S[i][k]]) * v[i, S[i][k]]
                               for k in range((len(S[i]) - 1)))
                 for i in I if len(S[i]) > 0)
    m.addConstrs(v[i, S[i][k]] <= v[i, S[i][k + 1]] for i in I
                 for k in range(len(S[i]) - 1))
    m.addConstrs(v.sum('*', j) >= counters[j] * y[j] for j in J)

    # Specify Gurobi options
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', max(time_limit - (time.time() - start_time), 5))

    # Provide warm start
    counter = 0
    for i in np.setdiff1d(current_solution, partial_solution):
        if i in J:
            y[i].Start = 1
            counter += 1

    # Update model
    m.update()

    # Optimize
    m.optimize()

    # Retrieve values of decision variables
    nSolutions = m.SolCount
    if nSolutions >= 1:
        J_new = np.concatenate((partial_solution, np.array([j for j in J if y[j].X > 0.5], dtype=int)))
    else:
        J_new = current_solution

    # Store new solution if improvement was found
    new_ofv = distances[:, J_new].min(axis=1).sum()
    if new_ofv > ofv_best:
        hit_time = time.time() - start_time
        ofv_best = new_ofv
        solution_best = J_new.copy()

    # Return information on new solution
    return set(J_new), J, new_ofv, solution_best, ofv_best, hit_time


def run(instance, p, start_time, time_limit, current_solution=set(),
        b_init=5, m_max_numerator=450**2, delta=2, hit_time=0, df_coordinates_f=None,
        distances_f=None, coord_avail=True, df=None, clustering=False,
        clustering_percentage=0.05, l=0):
    """Runs improvement procedure of matheuristic"""

    # Set seed
    np.random.seed(11)

    if coord_avail == True:
        # Read coordinate of clients
        df_coordinates_c = pd.read_csv('data/' + instance + '_coord_c.csv')
    
        # Compute distance matrix between clients and facilities
        df = compute_distance_matrix(df_coordinates_c=df_coordinates_c,
                                     df_coordinates_f=df_coordinates_f)

    # Cluster clients
    if clustering == True:
        # Create a copy of original distance matrix
        df2 = df.copy()
        # Define number of clusters
        num_clusters = int(clustering_percentage * df_coordinates_c.shape[0])
        # Use kmeans to cluster clients
        kmeans = KMeans(n_clusters=num_clusters,
                        random_state=11).fit(df_coordinates_c[['coordinate1',
                                                               'coordinate2']].values)
        df_coordinates_c['clustering_labels'] = kmeans.labels_
        # Derive adjusted distance matrix based on clusters of clients
        df = pd.DataFrame(index=range(num_clusters),
                          columns=df2.columns, dtype='float64')
        for i in range(num_clusters):
            idx = df_coordinates_c['clustering_labels']==i
            idx2 = df_coordinates_c.loc[idx, :].index
            df.iloc[i, :] = df2.iloc[idx2, :].sum()

    # Reset indices of distance matrix
    num_cust = df.shape[0]
    num_faci = df.shape[1]
    df = df.reset_index(drop=True)
    df.columns = range(num_faci)

    # Store best solution
    solution_best = np.array(list(current_solution), dtype=int)

    # For instances, for which coordinates are available compute centroid of
    # facilities and compute distances between centroid and all facilities
    if coord_avail == True:
        centroid = df_coordinates_f[['coordinate1', 'coordinate2']].mean(axis=0)
        distances_to_center = distance_matrix(centroid.values.reshape((1,centroid.shape[0])),
                                              df_coordinates_f[['coordinate1', 'coordinate2']])[0]
        distances_to_center = distances_to_center - distances_to_center.min() + 0.01

    # Initialize input parameters
    b = b_init
    m_max = int(m_max_numerator / num_cust)
    ofv = 0
    ofv_previous = 0
    ofv_best = 0
    num_iter = 0
    partial_solution = set()

    # Improve initial solution
    while time.time() - start_time < time_limit:
        # Increase iteration number
        num_iter += 1
        # If no improvement increase b
        if (num_iter > 1) and (ofv_previous >= ofv):
            b += delta
        # If improvement reset b
        else:
            b = b_init
        # Terminate if all facilities were removed in previous iteration
        if b > p:
            break
        ofv_previous = ofv

        # Remove b facilities
        partial_solution, current_solution, \
            = remove_facilities(df, p, b, num_cust, num_faci, current_solution,
                                partial_solution, time_limit, start_time)

        # Add b facilities
        if coord_avail == True:
            current_solution, J_considered, ofv, solution_best, ofv_best, hit_time, \
                = add_facilities(df, b, num_cust, num_faci, partial_solution, current_solution,
                                 start_time, ofv_best, solution_best, hit_time,
                                 time_limit, m_max, coord_avail,
                                 distances_to_center, distances_f)
        if coord_avail == False:
            current_solution, J_considered, ofv, solution_best, ofv_best, hit_time, \
                = add_facilities(df, b, num_cust, num_faci, partial_solution, current_solution,
                                 start_time, ofv_best, solution_best, hit_time,
                                 time_limit, m_max, coord_avail)

    # If clients were clustered, compute objective function value based on original distance matrix
    if clustering == True:
        print('OFV based on adjusted distance matrix', ofv_best)
        # Compute objective function value based on original distance matrix
        distances = df2.values
        ofv_best = distances[:, solution_best].min(axis=1).sum()
        print('OFV based on original distance matrix', ofv_best)

    return set(solution_best), ofv_best, hit_time

