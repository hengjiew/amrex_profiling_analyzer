import argparse
import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from amrex_profiling_parser import *


#%%
# compare multiple tests
nProc    = 16
roots_sg = []
for i in range(nProc):
  idstr = '0000' + repr(i) if i < 10 else '000' + repr(i)
  roots_sg.append(build_call_tree(
    'fuel_reactor_linearsolver/bl_hfix_tune/bl_call_stats_D_' + idstr))
  roots_sg[i].merge_same_subtrees()

roots_gr = []
for i in range(nProc):
  idstr = '0000' + repr(i) if i < 10 else '000' + repr(i)
  roots_gr.append(build_call_tree(
    'fuel_reactor_linearsolver/bl_h220/bl_call_stats_D_' + idstr))
  roots_gr[i].merge_same_subtrees()

#%%
# callPath = ['mfix_solve','mfix_dem::EvolveParticles()',]
# , 'NeighborParticleContainer::updateNeighborsGPU',
callPath = ['mfix_solve','mfix::EvolveFluid',
            'HypreABecLap3::prepareSolver()']
            # 'mfix::compute_MAC_projected_velocities()',
            # 'MLMG::solve()',
            # 'MLMG::oneIter()',
            # 'MLMG::mgVcycle()',
            # 'MLMG::mgVcycle_bottom']
print_runtime(roots_sg, callPath, depth=2, pid=4)
# searchPath = ['FabArray::setVal()']
# print_callpath(roots_sg, searchPath, pid=0)

#%%

callPaths = [
  ['mfix_solve','mfix::EvolveFluid', 'mfix::mfix_apply_predictor',
   'mfix::compute_MAC_projected_velocities()', 'MLMG::solve()',
   'MLMG::oneIter()', 'MLMG::mgVcycle()',],
  ['mfix_solve','mfix::EvolveFluid', 'mfix::mfix_apply_predictor',
   'mfix::compute_MAC_projected_velocities()', 'MLMG::solve()',
   'MLMG::oneIter()', 'MLMG::mgVcycle()','MLMG::mgVcycle_bottom',],
  ['mfix_solve','mfix::EvolveFluid', 'mfix::mfix_apply_predictor',
   'mfix::compute_MAC_projected_velocities()', 'MLMG::solve()',
   'MLMG::oneIter()', 'MLMG::mgVcycle()','MLCellLinOp::smooth()',
   'MLEBABecLap::applyBC()',],
  ['mfix_solve','mfix::EvolveFluid', 'mfix::mfix_apply_predictor',
   'mfix::compute_MAC_projected_velocities()', 'MLMG::solve()',
   'MLMG::oneIter()', 'MLMG::mgVcycle()','MLCellLinOp::smooth()',
   'MLEBABecLap::Fsmooth()',],
]

times0 = extract_runtime(roots_sg, callPaths)
times1 = extract_runtime(roots_gr, callPaths)

def plot_bar(times, fname, nProc):
  w = 0.6
  ranks = [repr(i) for i in range(nProc)]
  others = times[0]-times[1]-times[2]-times[3]
  fig = plt.figure(figsize=(8,4))
  plt.bar(ranks, times[2], label='gmg comm', width=w)
  plt.bar(ranks, times[3], bottom=times[2], label='gmg comp', width=w)
  plt.bar(ranks, times[1], bottom=times[2]+times[3], label='bottom', width=w)
  plt.bar(ranks, others, bottom=times[1]+times[2]+times[3], label='other',
          color='gray', width=w)
  plt.legend(fontsize=16)
  plt.ylabel('Time(s)', fontsize=16)
  plt.xlabel('Rank',fontsize=16)
  plt.ylim(0, 6.0)
  plt.title('MG Breakdown ' + fname,fontsize=16)
  # plt.savefig(fname + '.png', bbox_inches="tight")

# plot_bar(times1, 'hypre-cpu', 18)


#%%
# nProc = 18
callPaths = [
  ['mfix_solve', 'mfix_dem::EvolveParticles()','particles_computation'],
  ['mfix_solve', 'mfix_dem::EvolveParticles()', 'NeighborParticleContainer::updateNeighborsGPU'],
  ['mfix_solve', 'mfix_dem::EvolveParticles()',],
  ['mfix_solve',],
]
    # ['mfix_solve', 'mfix_dem::EvolveParticles()',
    # 'NeighborParticleContainer::buildNeighborList' ],

times_sg = extract_runtime(roots_sg, callPaths)
times_gr = extract_runtime(roots_gr, callPaths)

fig=plt.figure(figsize=(8,4))
plt.ylabel('Time(s)', fontsize=12)
plt.xlabel('Rank, 1 GPU per rank', fontsize=12)
x = [i for i in range(nProc)]
plt.xticks(x, x)

name1 = 'SG0'
name2 = 'SG1'
plt.title("fuel-reactor SG, reduced ghost particles")
plt.plot(times_sg[0], 'bs', label=name1 + '-computation',
          markersize=6, fillstyle='none')
# plt.plot(times_sg[3], 'b^', label=name1 + '-buildNeighborList',
#           markersize=6, fillstyle='none')
plt.plot(times_sg[1], 'bo',label=name1 + '-updateNeighborGPU',
          markersize=6, fillstyle='none')
plt.plot(times_sg[2], 'b-',label=name1 + '-EvolveParticles',
          markersize=6, fillstyle='none')

plt.plot(times_gr[0], 'ks', label=name2 + '-computation',
          markersize=6, fillstyle='none')
# plt.plot(times_gr[3], 'k^', label=name2 + '-buildNeighborList',
#           markersize=6, fillstyle='none')
plt.plot(times_gr[1], 'ko',label=name2 + '-updateNeighborsGPU',
          markersize=6, fillstyle='none')
plt.plot(times_gr[2], 'k-',label=name2 + '-EvolveParticles',
          markersize=6, fillstyle='none')

plt.legend(fontsize=12, bbox_to_anchor=(1.02, 1))
# plt.title('coarse-clr {} GPUs'.format(nProc), fontsize=12)
# plt.savefig('parallelcopy_time.png', bbox_inches="tight")
#%%
# unmerged call trees
rootLists = []
nProc     = 24
for dirName in ['bl_sg', 'bl_gr']:
  roots = []
  for i in range(nProc):
    idstr = '0000' + repr(i) if i < 10 else '000' + repr(i)
    roots.append(build_call_tree(
      dirName + '/bl_call_stats_D_' + idstr))
  rootLists.append(roots)

#%%
# plot the (max) function time per step

callPath = ['mfix_solve', 'mfix_dem::EvolveParticles()',]
timeLists = []
for i, roots in enumerate(rootLists):
  times = []
  for j, root in enumerate(roots):
    nodeList = root.search(callPath)
    times.append([node.runTime for node in nodeList])
  timeLists.append(times)

# # find max time per step per test
# maxtimes = []
# for times in timeLists:
#   maxtime = []
#   for i in range(len(times[0])):
#     tmp = 0.0
#     for timePerProc in times:
#       tmp = max(tmp, timePerProc[i])
#     maxtime.append(tmp)
#   maxtimes.append(maxtime)

# plt.plot(maxtimes[0], 'bx')
for i in range(24):
  plt.plot(timeLists[0][i][:20], label=repr(i))
plt.title(callPath[-1])
# x = [i for i in range(len(timeLists[0][0]))]
# plt.xticks(x, x)
# plt.legend()

