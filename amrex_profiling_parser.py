import sys
import os
import re
import numpy as np
import pandas as pd


#-------------------------------------------------------------------
# classes for parsing the ASCII output of BL profiling
#-------------------------------------------------------------------

class Call:
  def __init__(self, name='', depth=0, runTime=0.0):
    self.name     = name
    self.depth    = depth
    self.runTime  = runTime


class TreeNode:
  def __init__(self, call, parent=None):
    # function name
    self.name = call.name
    # calls issued in current call, depth incremented by 1
    self.children = []
    # parent tree node
    self.parent   = parent
    #
    self.runTime  = call.runTime
    # times called
    self.count    = 1


  def deep_copy(self):
    node = TreeNode(Call())
    node.name    = self.name
    node.parent  = self.parent
    node.runTime = self.runTime
    node.count   = self.count
    for child in self.children:
      node.children.append(child.deep_copy())
    return node


  def merge_time(self, node):
    # assert len(self.children) == len(node.children)
    self.runTime += node.runTime
    self.count   += node.count
    # for i in range(len(self.children)):
    #   self.children[i].merge_time(node.children[i])
    unseenChildren = []
    for nodeChild in node.children:
      childSeen = False
      for myChild in self.children:
        if myChild.name == nodeChild.name:
          childSeen = True
          myChild.merge_time(nodeChild)
      if not childSeen:
        unseenChildren.append(nodeChild)
    for unseenChild in unseenChildren:
      unseenChild.parent = self
      self.children.append(unseenChild)


  def is_mergable(self, node):
    if self.name != node.name:
      return False

    if len(self.children) != len(node.children):
      return False

    for i in range(len(self.children)):
      if not self.children[i].is_mergable(node.children[i]):
        return False

    return True


  def merge_same_subtrees(self):
    for child in self.children:
      child.merge_same_subtrees()

    i = 0
    while i < len(self.children):
      mergeIDs = []
      for j in range(i+1, len(self.children)):
        # if self.children[i].is_mergable(self.children[j]):
        if self.children[i].name == self.children[j].name:
          mergeIDs.append(j)
      for j in reversed(mergeIDs):
        self.children[i].merge_time(self.children[j])
        self.children.pop(j)
      i += 1


  def print_call_tree(self, maxDepth=1):

    def help_print(root, maxDepth, total, indent):
      print(indent + ' {:<40} {:.2e}  {:.4f}  {:d}'.format(\
              root.name, root.runTime, root.runTime/total, root.count))
      if maxDepth > 0:
        for child in root.children:
          help_print(child, maxDepth-1, root.runTime, indent+'  ')

    print(self.name + '  {:.2e}  {:d}'.format(self.runTime, self.count))
    if maxDepth > 0:
      for child in self.children:
        help_print(child, maxDepth-1, self.runTime, '  ')


  def search(self, nameList):

    def help_search(root, nameList, resList):
      if root.name == nameList[0]:
        if len(nameList) > 1:
          for child in root.children:
            help_search(child, nameList[1:], resList)
        else:
          resList.append(root)
      else:
        for child in root.children:
          help_search(child, nameList, resList)
      return

    resList = []
    help_search(self, nameList, resList)

    return resList


  def diff(self, node):
    def help_diff(node0, node1, diffList):
      if node0.name != node1.name or len(node0.children) != len(node1.children):
        diffList.append((node0, node1))
      else:
        for i in range(len(node0.children)):
          help_diff(node0.children[i], node1.children[i], diffList)

    diffList = []
    help_diff(self, node, diffList)
    return diffList


  def get_runtime(self, nameList):
    assert len(nameList) > 0
    nodeList = self.search(nameList)
    res      = merge_node_list(nodeList)
    return res.runTime


#-------------------------------------------------------------------
# functions creating call trees
#-------------------------------------------------------------------


def build_calltree_list(path, nProc=0, fileType='binary', endian='little', \
                        longIntSize=8, main=None, procs=[]):
  '''
    Build call trees based on AMReX Base profiling, one tree per process.

    Input:
    path        - path to where base profiling data are stored
    nProc       - number of process profiled
    fileType    - This is for backward consistency with my old implementation,
                  will be remove in future
    endian      -
    longIntSize - depends on where the profiling is performed
    main        - the name of the main function of the top profiled function
                  that calls child functions. If not specified, tree node will
                  be a dummy node with depth -1 to encapsulate all the profiled
                  functions.
  '''

  assert endian.lower() in ['little', 'big']
  assert longIntSize == 4 or longIntSize == 8
  assert fileType.lower() in ['ascii', 'binary']

  if fileType.lower() == 'binary':
    prefix = '/bl_call_stats_H_'
  else:
    prefix = '/bl_call_stats_D_'
    
  assert nProc*len(procs) == 0 or len(procs) == nProc
  ranks = procs if len(procs) > 0 else [i for i in range(nProc)]
  
  roots = []
  for i in ranks:
    fname = path + prefix + '{:05d}'.format(i)
    roots.append(build_call_tree(fname, fileType=fileType,\
                                 endian=endian,\
                                 longIntSize=longIntSize, \
                                 main=main))
    roots[-1].merge_same_subtrees()
    
  return roots


def build_call_tree(fname, endian='little', longIntSize=8, fileType='binary', main=None):
  endianChar = '<' if endian.lower() == 'little' else '>'
  root = None
  if fileType.lower() == 'ascii':
    root = build_from_ascii(fname, main)
  elif fileType.lower() == 'binary':
    # amrex bl region type
    RegionType = np.dtype([('rssTime',    endianChar+'f8'),
                           ('rssRNumber', endianChar+'i4'),
                           ('rssStart',   endianChar+'i4'),])
    # amrex bl call status type
    CallType   = np.dtype([('depth',     endianChar+'i4'),
                           ('fnameId',   endianChar+'i4'),
                           ('nCSCalls',  endianChar+'i'+repr(longIntSize)),
                           ('totalTime', endianChar+'f8'),
                           ('stackTime', endianChar+'f8'),
                           ('callTime',  endianChar+'f8'),])
    root = build_from_binary(fname, RegionType, CallType, main)

  return root


def build_from_ascii(fname, main):
  callStack = []
  seekMain  = not (main == None)
  mainFound = False
  root      = None if seekMain else TreeNode(Call('root', -1, 0.0))
  node      = root

  with open(fname, 'r') as callFile:
    for iline, line in enumerate(callFile):
      # name = namePattern.search(line).group()
      items = line.split()
      call  = Call(' '.join(items[:-3]), int(items[-3]), float(items[-1]))

      if seekMain:
        if call.name == main:
          mainFound = True
          root      = TreeNode(call)
          node      = root

      if mainFound or (not seekMain):
        # pop out deeper calls and rewind tree node tracker
        while len(callStack) > 0 and call.depth <= callStack[-1].depth:
          callStack.pop()
          node = node.parent
        # append call to parent node and stack
        node.children.append(TreeNode(call, parent=node))
        node = node.children[-1]

      # push node to stack
      callStack.append(call)

  return root


def merge_node_list(nodeList):
  assert (len(nodeList) > 0)
  res = nodeList[0].deep_copy()
  if (len(nodeList) > 1):
    for i in range(1, len(nodeList)):
      res.merge_time(nodeList[i])
  return res


def build_from_binary(fnameH, RegionType, CallType, main):
  # header file
  fHeader = open(fnameH, 'r')
  lines   = fHeader.readlines()

  # first line
  items  = lines[0].strip().split()
  nRss   = int(items[3]) # number of RStartStop (AMReX_BLProfiler.H)
  nCall  = int(items[5]) # number of call statuses

  # lines for function name and its number
  # construct a map <int, functionName>
  funcIdNameMap = {}
  for line in lines[1:-1]:
    items = line.strip().split('"')
    try:
      funcIdNameMap[int(items[2].strip())] = items[1]
    except Exception as e:
      print(str(e))
      print("Error in", fnameH, line)
      print(lines[-1])
      print(lines[-2])
      return
  fHeader.close()

  # read binary file to a buffer
  # the data file's name is different than header's
  # by replacing _H_ with _D_
  fBin    = open('_D_'.join(fnameH.rsplit('_H_', 1)), 'rb')
  buffer  = fBin.read()
  fBin.close()
  # load region and call data from buffer
  regionData = np.frombuffer(buffer, dtype=RegionType, count=nRss)
  callData   = np.frombuffer(buffer, dtype=CallType, count=nCall, \
                             offset=regionData.nbytes)

  callStack = []
  seekMain  = not (main == None)
  mainFound = False
  root      = None if seekMain else TreeNode(Call('root', -1, 0.0))
  node      = root

  # setup call tree
  for callStat in callData[1:]:
    call = Call(funcIdNameMap[callStat['fnameId']], \
                callStat['depth'], \
                callStat['totalTime'])

    if seekMain and not mainFound:
      if call.name == main:
        mainFound = True
        root      = TreeNode(call)
        node      = root
        continue

    if mainFound or (not seekMain):
      # pop out deeper calls and rewind tree node tracker
      while len(callStack) > 0 and call.depth <= callStack[-1].depth:
        callStack.pop()
        node = node.parent
      # append call to parent node and stack
      node.children.append(TreeNode(call, parent=node))
      node = node.children[-1]
      # push node to stack
      callStack.append(call)

  if seekMain and not mainFound:
    print("Error: cannot find input function " + main)

  return root


#-------------------------------------------------------------------
# functions extracting info from call trees
#-------------------------------------------------------------------


def extract_runtime(roots, callPaths):
  '''
    Give a list of (partial) call paths, find the matching full pathes
    and get the time cost of the paths'last functions.

    Return:
    a list of times corresponding to the input list of paths
  '''

  times = []
  for callPath in callPaths:
    callPathTime = []
    for root in roots:
      callPathTime.append(root.get_runtime(callPath))
    times.append(np.array(callPathTime))
  return times


def print_runtime(roots, callPath, depth=1, pid=-1):
  '''
    Given a (partial) call path, find the matching full paths and print
    out the last function's child functions' call counts and time costs.

    Note that if multiple full paths are found, they will merged into
    one path. Unless intended, it is better to specify the input path
    with sufficient functions so that only the full path of interest is
    matched.

    Inputs:
    roots    - list of call tree roots
    callPath - a (partial) call path to be searched in the call tree
    depth    - number of levels of children functions to be printed
    pid      - process id
  '''
  for i, root in enumerate(roots):
    if pid >= 0 and pid < len(roots) and i != pid: continue
    nodeList = root.search(callPath)
    assert (len(nodeList) > 0)
    res = merge_node_list(nodeList)
    print('-------- processes {:d} --------'.format(i))
    if (len(nodeList) > 1):
      print('merge {:d} paths'.format(len(nodeList)))
    res.print_call_tree(maxDepth=depth)
    print()


def print_callpath(roots, searchPath, sortBy='count', pid=-1, nPrint=5):
  '''
    Give a (partial) call path, print out full call pathes that contains
    the input (partial) path.

    Inputs:
    roots      - list of call tree roots
    searchPath - a (partial) call path to be searched in the call tree
    sortBy     - a key by which sort the found pathes. This can either be
                 call count or time cost.
    pid        - process id
    nPrint     - number of top pathes to be printed
  '''
  assert sortBy.lower() in ['count', 'time']
  assert pid < len(roots)

  if sortBy.lower() == 'count':
    sortKey, sortKind = 1, 'referenced'
  else:
    sortKey, sortKind = 2, 'expensive'

  for i, root in enumerate(roots):
    if pid >= 0 and pid < len(roots) and i != pid: continue
    nodeList = root.search(searchPath)
    print('-------- processes {:d} --------'.format(i))
    print('find {:d} paths and print the most {} ones'.format(\
          len(nodeList), sortKind))
    # sort node by its call count
    nodeCallCounts = []
    for node in nodeList:
      nodeCallCounts.append((node, node.count, node.runTime))

    nodeCallCounts.sort(key=lambda x: x[sortKey], reverse=True)
    # print the most frequently called nodes
    for j in range(min(nPrint, len(nodeCallCounts))):
      node = nodeCallCounts[j][0]
      # find the call path by backtracking
      path = []
      while node != None:
        path.append(node.name)
        node = node.parent
      path.reverse()
      # print the count, time cost, full path
      print('{} {:.4e}'.format(nodeCallCounts[j][1], \
                               nodeCallCounts[j][0].runTime), \
                               end='  ')
      for k, name in enumerate(path):
        if k < len(path) - 1:
          print(name + ' -> ', end='')
        else:
          print(name)


#-------------------------------------------------------------------
# functions extracting info from mfix's stdout
#-------------------------------------------------------------------


def read_step_time(filepath):
  '''
  Extract time from mfix's stdout per step
  '''
  times = {'step':[], 'fluid':[], 'particle':[], 'substep':[], \
           'couple':[], 't':[], 'peff':[]}
  with open(filepath, 'r') as file:
    for i, line in enumerate(file):
      #
      if line.find("Time per fluid step") != -1:
        times['fluid'].append(float(line.strip().split()[-1]))
      #
      if line.find("particle steps") != -1:
        words = line.strip().split()
        times['substep'].append(int(words[2]))
        times['particle'].append(float(words[-1]))
      #
      if line.find("Coupling time") != -1:
        times['couple'].append(float(line.strip().split()[-1]))
      #
      if line.find("Time per step") != -1:
        times['step'].append(float(line.strip().split()[-1]))
      #
      if line.find("new time") != -1:
        words = line.strip().split()
        for j in range(len(words)-2):
          if words[j] == "new":
            times['t'].append(float(words[j+2]))
      #
      if line.find("Particle load efficiency") != -1:
        times['peff'].append(float(line.strip().split()[-1]))

  return times
