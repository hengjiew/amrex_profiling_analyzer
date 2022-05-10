#%%
import sys
import matplotlib.pyplot as plt
import numpy as np
from amrex_profiling_parser import *

#%%
# build one call tree per process
# In most cases, we have one top level function e.g., main(), solve(), etc.
# We can specify its name via the argument "main". If not specified, a dummy
# node will used as the tree root.

# This module aims at analyzing the overall performance statistics instead of
# individual function calls. The call tree merges call paths with same function
# names e.g., calls of the same function at different iterations.

roots = build_calltree_list(path='../mfix_post/bl_prof_bin', nProc=18)

# this also works if you profile the main function and name the profiler 'main()'.
# roots = build_calltree_list(path='./bl_prof_bin', nProc=18, main='main()')

#%%
# From base profiling (or tiny profiling), we can see the most expensive functions
# at the end of program's output. These functions may be called multiple times at
# different locations. The following API helps to find where exactly these expsensive
# functions are called.
# More usage of `print_callpath` can be found in its doc in amrex_prifliing_parser.py

searchPath = ['FabArray::ParallelCopy_finish()']
print_callpath(roots, searchPath, sortBy='time', pid=0, nPrint=4)

#%%
# In some cases, we know a specfic high level function is very expensive. The following
# API helps to detail what child functions it calls including their call counts and costs.
# The second, third, and fouth columns of the stdout are the time cost, cost percentage of
# the parent function, and the call counts, respectively.

# Note that if multiple full pathes are found, they will merged into # one path.
# Unless intended, it is better to specify the input path # with sufficient functions
# so that only the full path of interest is matched. Fox instance, the following example
# outputs the costs the MG solver in mac projector. If the third function is omitted, it
# will output MG sovler's statistics collected from both the mac and nodal projectors.

# MG performance in mac projector
callPath = ['mfix_solve','mfix::EvolveFluid',
            'mfix::compute_MAC_projected_velocities()',
            'MLMG::mgVcycle()']
print_runtime(roots, callPath, depth=1, pid=4)

# Each function's cost are is the sum over its calls in both mac and nodal projector.
callPath = ['mfix_solve','mfix::EvolveFluid',
            'MLMG::mgVcycle()']
print_runtime(roots, callPath, depth=3, pid=4)

# %%
# This module comes handy when one wants to generate performance plots for slides/reports.
# The following examples shows how to plot the time usage of a function across different
# processes (to identify load imbalances) and visualize the time breakdown of a particular
# solver.

# Assuming we indentifies the hotspots of the particle solver from previous steps, we can
# plot the per process usage for further analysis.
callPaths = [
  ['mfix_solve', 'mfix_dem::EvolveParticles()',],
  ['mfix_solve', 'mfix_dem::EvolveParticles()','particles_computation'],
  ['mfix_solve', 'mfix_dem::EvolveParticles()', 'NeighborParticleContainer::updateNeighborsGPU'],
]


times = extract_runtime(roots, callPaths)

fig   =plt.figure(figsize=(8,4))
nProc = 18 # number of process
x     = [i for i in range(nProc)]
plt.xticks(x, x)
plt.title("Particle Solver's Performance")
plt.ylabel('Time(s)', fontsize=12)
plt.xlabel('Rank, 1 GPU per rank', fontsize=12)

plt.plot(times[0], 'b-', label='EvolveParticles',   fillstyle='none')
plt.plot(times[1], 'bs', label='computation',       markersize=6, fillstyle='none')
plt.plot(times[2], 'bo', label='updateNeighborGPU', markersize=6, fillstyle='none')
plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1))

# %%
# The following **roughly** estimates MG solver in the mac projection with embed
# boundaris, including the computation and communication cost of levels above
# the coarest grid and the bottom solver's cost.

# Note that due to AMReX's grid aggregation the bottom solver may only be called
# by part of the processes. So the plot may present a significant imbalance, which
# is not necessarily a performance hit.

callPaths = [
  ['mfix::compute_MAC_projected_velocities()', 'MLMG::mgVcycle()',],
  ['mfix::compute_MAC_projected_velocities()', 'MLMG::mgVcycle()',
   'MLMG::mgVcycle_bottom',],
  ['mfix::compute_MAC_projected_velocities()', 'MLMG::mgVcycle()',
   'MLEBABecLap::applyBC()', 'FabArray::FillBoundary()',],
  ['mfix::compute_MAC_projected_velocities()', 'MLMG::mgVcycle()',
   'MLEBABecLap::Fsmooth()',],
]

times = extract_runtime(roots, callPaths)
nProc  = 18

w = 0.6
ranks = [repr(i) for i in range(nProc)]
others = times[0]-times[1]-times[2]-times[3]
fig = plt.figure(figsize=(8,4))
plt.bar(ranks, times[2], label='halo exchange', width=w)
plt.bar(ranks, times[3], bottom=times[2], label='smoothing', width=w)
plt.bar(ranks, times[1], bottom=times[2]+times[3], label='bottom solver', width=w)
plt.legend(fontsize=16)
plt.ylabel('Time(s)', fontsize=16)
plt.xlabel('Rank',fontsize=16)
plt.title('MG Breakdown', fontsize=16)
