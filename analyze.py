#%%
import sys
import matplotlib.pyplot as plt
import matplotlib
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

roots = []
# we can specify ranks to be extracted from the profiling file
procs =  [i for i in range(4)] + [i for i in range(8,12)]
roots.append(build_calltree_list(path='../cpuvsgpu/bl_64c', procs=procs))
# we can also specify the # ranks. This will read rank 0-7 from file
roots.append(build_calltree_list(path='../cpuvsgpu/bl_1g', nProc=8))

# this also works if you profile the main function and name the profiler 'main()'.
# roots = build_calltree_list(path='./bl_prof_bin', nProc=18, main='main()')

#%%
# From base profiling (or tiny profiling), we can see the most expensive functions
# at the end of program's output. These functions may be called multiple times at
# different locations. The following API helps to find where exactly these expsensive
# functions are called.
# More usage of `print_callpath` can be found in its doc in amrex_prifliing_parser.py

searchPath = ['FabArray::ParallelCopy_finish()']
print_callpath(roots[0], searchPath, sortBy='time', pid=0, nPrint=4)

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

# if pid=-1, it will output the results of all ranks

callPath = ['mfix_solve', 'mfix::EvolveFluid',
            'mfix::mfix_apply_predictor',
            'mfix::mfix_apply_nodal_projection',
            'MLMG::solve()', 'MLMG::mgVcycle()',
            'MLMG::actualBottomSolve()',]

print("----------------------------------------------------------------")
print("cpu")
print("----------------------------------------------------------------")
print_runtime(roots[0], callPath, depth=4, pid=0)
print("----------------------------------------------------------------")
print("gpu")
print("----------------------------------------------------------------")
print_runtime(roots[1], callPath, depth=4, pid=0)

# %%
# This module comes handy when one wants to generate performance plots for slides/reports.
# The following examples shows how to plot the time usage of a function across different
# processes (to identify load imbalances) and visualize the time breakdown of a particular
# solver.

# Assuming we indentifies the hotspots of the particle solver from previous steps, we can
# plot the per process usage for further analysis.
callPaths = [
    ['mfix_solve','mfix_dem::EvolveParticles()',],
    ['mfix_solve','mfix_dem::EvolveParticles()','particles_computation'],    
    ['mfix_solve','mfix_dem::EvolveParticles()',
     'NeighborParticleContainer::updateNeighborsGPU',],
    ['mfix_solve','mfix_dem::EvolveParticles()',
     'ParticleContainer::RedistributeGPU()',],
]
r     = 1
nProc = len(roots[r])
t     = extract_runtime(roots[r], callPaths)
tTot  = t[0]
tComp = t[1]
tComm = t[2] + t[3]

fig   = plt.figure(figsize=(8,4))
x     = np.arange(nProc)
plt.xticks(x, x)
plt.title("Particle Solver's Performance on {:d} GPUs".format(nProc))
plt.ylabel('Time(s)', fontsize=12)
plt.xlabel('Rank, 1 GPU per rank', fontsize=12)

plt.plot(tTot,  'b-', label='EvolveParticles',   fillstyle='none')
plt.plot(tComp, 'bs', label='computation',   markersize=6, fillstyle='none')
plt.plot(tComm, 'bo', label='communication', markersize=6, fillstyle='none')
plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1))

# %%
# The following **roughly** estimates MG solver in the mac projection with embed
# boundaris, including the computation and communication cost of levels above
# the coarest grid and the bottom solver's cost.

# Note that due to AMReX's grid aggregation the bottom solver may only be called
# by part of the processes. So the plot may present a significant imbalance, which
# is not necessarily a performance hit.

top = 'mfix::mfix_apply_nodal_projection'
mg  = 'MLMG::mgVcycle()'
down= 'MLMG::mgVcycle_down::'
up  = 'MLMG::mgVcycle_up::'
FB  = 'FabArray::FillBoundary()'
PC  = 'FabArray::ParallelCopy()'
r   = 0

# total and bottom
callPaths = [[top, 'MLMG::mgVcycle()',],
             [top, 'MLMG::mgVcycle()', 'MLMG::mgVcycle_bottom',],]
times     = extract_runtime(roots[r], callPaths)
tTotal    = times[0]
tBott     = times[1]

# communication
nProc     = len(tTotal)
nLev      = 5
callPaths = []
for i in range(nLev): 
  callPaths.append([top, down+repr(i), FB])
  callPaths.append([top, up  +repr(i), FB])
times     = extract_runtime(roots[r], callPaths)
tFB       = sum(times)
callPaths = [[top, mg, PC],]
times     = extract_runtime(roots[r], callPaths)
tPC       = times[0]
tComm     = tFB + tPC

# computation
callPaths = []
for i in range(nLev):
  callPaths.append([top, down+repr(i), 'MLNodeLaplacian::Fsmooth()'])
  callPaths.append([top, up  +repr(i), 'MLNodeLaplacian::Fsmooth()'])
for i in range(nLev):
  callPaths.append([top, down+repr(i), 'MLMG:computeResOfCorrection()'])
  callPaths.append([top, down+repr(i), 'MLMG:computeResOfCorrection()', FB])
#
callPaths.append([top, mg, 'MLNodeLaplacian::restriction()'])
callPaths.append([top, mg, 'MLNodeLaplacian::restriction()', PC])
callPaths.append([top, mg, 'MLNodeLaplacian::restriction()', FB])
callPaths.append([top, mg, 'MLMG::addInterpCorrection()'])
callPaths.append([top, mg, 'MLMG::addInterpCorrection()', PC]) 
times    = extract_runtime(roots[r], callPaths)
tComp    = sum(times[:2*nLev])
for i in range(2*nLev, 4*nLev, 2):
  tComp += times[i] - times[i+1]
i        = 4*nLev
tComp   += times[i] - times[i+1] - times[i+2]
tComp   += times[i+3] - times[i+4]

tOther   = tTotal - tComm - tBott - tComp
# print(tOther)

ranks = [repr(i) for i in range(nProc)]
w     = 0.6

fig = plt.figure(figsize=(12,4))
plt.bar(ranks, tFB, label='FB', width=w)
plt.bar(ranks, tPC, bottom=tFB, label='PC', width=w)
plt.bar(ranks, tComp,  bottom=tComm, label='comp', width=w)
plt.bar(ranks, tBott,  bottom=tComm+tComp, label='bottom', width=w)
plt.bar(ranks, tOther, bottom=tComm+tComp+tBott, label='other', width=w)
plt.xlim(-1, nProc)
plt.ylim(0.0, np.amax(tTotal)+0.5)
plt.legend(fontsize=12)
plt.ylabel('Time(s)', fontsize=16)
plt.xlabel('Rank',fontsize=16)
plt.title('MG {:d} Breakdown'.format(r), fontsize=16)
# print(np.amax(tTotal), np.amin(tTotal))
# print("rank 0 - {:.4e} {:.4e} {:.4e} {:.4e}".format(tTotal[-1], tComp[-1], tComm[-1], tBott[-1]))


