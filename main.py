# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:56:11 2021

@author: tamara.bigler
"""

# Import standard packages
import time

# Import algorithm
import matheuristic as mh


# ----------------------------------------------------------------------------
# Instances for which coordinates of clients and facilities are available
# ----------------------------------------------------------------------------

# Define instance
instance = 'ID_1'

# Define parameter p
p = 67

# Coordinates of instances available
coord_avail = True

# Clustering clients (only possible if coordinates of clients and facilites are available)
clustering = False # True or False

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Instances for which coordinates of clients and facilities are not available
# (only distances between clients and facilities)
# ----------------------------------------------------------------------------

# Define instance (note that this instance can be downloaded from https://grafo.etsii.urjc.es/optsicom/opm.html)
# instance = 'pmed17-p25.A'

# Define parameter p
# p = 25

# Coordinates of instances available
# coord_avail = False

# ----------------------------------------------------------------------------

# Define time limit
time_limit = 10

# Define number of initial solutions that are generated
num_jobs = 8

# Define number of initial solutions that should be generated randomly
num_random = 2

# Define parameter values
b_init = 5
delta = 1
m_max_numerator = 450**2 # will be divided by number of clients to compute m_max

# Run matheuristic
# Start clock
start_time = time.time()

# Generate num_jobs initial solutions
if coord_avail==True:
    solutions, hit_time, df_coordinates_f, distances_f = \
        mh.initial_solutions_kmeans_plus_plus_random(instance, p, start_time,
                                                     num_jobs=num_jobs,
                                                     num_random=num_random)
if coord_avail==False:
    solutions, hit_time, df = \
        mh.initial_solutions(instance, p, start_time, num_jobs=num_jobs,
                             num_random=num_random,
                             rule_based_solution1=True, rule_based_solution2=True) # two rule-based solutions are generated if rule_based_solution1 and rule_based_solution2 are set to True

# Define index of initial solution to which improvement procedure is applied
l = 1 # parameter takes values from 0 to num_jobs - 1

# Run improvement procedure
if coord_avail==True:
    solution, ofv, hit_time = mh.run(instance=instance, p=p, start_time=start_time,
                                      time_limit=time_limit, current_solution=solutions[l],
                                      b_init=b_init, m_max_numerator=m_max_numerator,
                                      delta=delta, hit_time=hit_time,
                                      df_coordinates_f=df_coordinates_f, distances_f=distances_f, # only available if coordinates of facilities available
                                      coord_avail=coord_avail,
                                      clustering=clustering, l=l)
if coord_avail==False:
    solution, ofv, hit_time = mh.run(instance=instance, p=p, start_time=start_time,
                                      time_limit=time_limit, current_solution=solutions[l],
                                      b_init=b_init, m_max_numerator=m_max_numerator,
                                      delta=delta, hit_time=hit_time,
                                      coord_avail=coord_avail,
                                      df=df, l=l)

print('OFV: ', ofv)
print('Open facilities: ', sorted(solution))
print('Hit time: ', round(hit_time, 2))
print('Running time: ', round(time.time() - start_time, 2))

