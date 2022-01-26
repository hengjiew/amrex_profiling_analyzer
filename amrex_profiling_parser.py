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
      print(indent + root.name + '  {:.2e}  {:.4f}  {:d}'.format(\
              root.runTime, root.runTime/total, root.count))
      if maxDepth > 0:
        for child in root.children:
          help_print(child, maxDepth-1, root.runTime, indent+'  ')

    print(self.name + '  {:.2e}  {:d}'.format(self.runTime, self.count))
    if maxDepth > 0:
      for child in self.children:
        help_print(child, maxDepth-1, self.runTime, '  ')

    #print(indent + self.name + '  {:.2e}'.format(self.runTime))
    #if maxDepth > 0:
      #for child in self.children:
        #child.print_call_tree(maxDepth-1, indent+'  ')


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


def build_call_tree(fname):
  callStack = []
  root      = None
  # namePattern = re.compile(r'[a-zA-Z()\-: ]*')
  with open(fname, 'r') as callFile:
    findMain = False
    for iline, line in enumerate(callFile):
      # name = namePattern.search(line).group()
      items = line.split()
      call  = Call(' '.join(items[:-3]), int(items[-3]), float(items[-1]))
      if call.name == 'main()':
        findMain = True
        root     = TreeNode(call)
        node     = root
      # calls functions called in main()
      elif findMain:
        # pop out deeper calls and rewind tree node tracker
        while call.depth <= callStack[-1].depth:
          callStack.pop()
          node = node.parent
        # append call to parent node and stack
        node.children.append(TreeNode(call, node))
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


#-------------------------------------------------------------------
# functions extracting info from call trees
#-------------------------------------------------------------------

def extract_runtime(roots, callPaths):
  times = []
  for callPath in callPaths:
    callPathTime = []
    for root in roots:
      callPathTime.append(root.get_runtime(callPath))
    times.append(np.array(callPathTime))
  return times


def print_runtime(roots, callPath, depth=1, pid=-1):
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


def print_callpath(roots, searchPath, pid=-1):
  for i, root in enumerate(roots):
    if pid >= 0 and pid < len(roots) and i != pid: continue
    nodeList = root.search(searchPath)
    print('-------- processes {:d} --------'.format(i))
    print('find {:d} paths and print few frequently called ones'.format(len(nodeList)))
    nodeCallCounts = []
    for node in nodeList:
      nodeCallCounts.append((node, node.count))
    nodeCallCounts.sort(key=lambda x: x[1], reverse=True)
    for j in range(min(5, len(nodeCallCounts))):
      node = nodeCallCounts[j][0]
      # find the call path by backtracking
      path = []
      while node != None:
        path.append(node.name)
        node = node.parent
      path.reverse()
      #
      print('{} {:.4e}'.format(nodeCallCounts[j][1], nodeCallCounts[j][0].runTime), end='  ')
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


def read_function_time_base(filepath):
  '''
  Extract info from tiny profiling's summary in stdout.
  '''
  funcName  = []
  timeStat  = {}
  iTitle    = sys.maxsize # line number of titles
  titles    = []
  with open(filepath, 'r') as file:
    for i, line in enumerate(file):
      # first time see Total Times
      # The title line is below this line
      if line.find("Total times") != -1:
        iTitle = i + 1
      # ends at first blanck line after starting extraction
      endExacting = i > iTitle and len(line.strip()) == 0
      # extract titles,
      # !!!! hardcodeed from 2 to -2 !!!!
      if i == iTitle:
        titles = line.strip().split()[2:-1]
        for title in titles:
          timeStat[title] = []
      # starts extracting statistics after passing the title line
      if i > iTitle and (not endExacting):
        # words = line.strip().split('  ')
        words = re.split('\s{2,}', line.strip())
        # print(iTitle, words)
        # statistics
        for j, title in enumerate(titles):
          timeStat[title].append(float(
            words[j+1][:-2] if '%' in words[j+1] else words[j+1]))
        # function name
        funcName.append(words[0])
      #
      if endExacting: break

  # print(timeStat)
  #create the pandas dataframe
  return pd.DataFrame(timeStat, index=funcName)


def read_function_time_tiny(filepath, inclusive=False):
  '''
  Extract info from tiny profiling's summary in stdout.
  '''
  funcName  = []
  timeStat  = {}
  iTitle    = sys.maxsize # line number of titles
  titles    = []
  searchKey = 'Excl.' if not inclusive else 'Incl.'
  with open(filepath, 'r') as file:
    for i, line in enumerate(file):
      # first time see Total Times
      # The title line is below this line
      if line.find(searchKey) != -1:
        iTitle = i
      # ends at first blanck line after starting extraction
      endExacting = i > iTitle and len(line.strip()) == 0
      # extract titles,
      # !!!! hardcodeed from 1 to end !!!!
      if i == iTitle:
        titles = re.split('\s{2,}', line.strip())[1:]
        for title in titles:
          timeStat[title] = []
      # starts extracting statistics after passing the title line
      if i > iTitle and (not endExacting):
        # words = line.strip().split('  ')
        words = re.split('\s{2,}', line.strip())
        if len(words) <= 1:
          continue
        # print(iTitle, words)
        # statistics
        for j, title in enumerate(titles):
          timeStat[title].append(float(
            words[j+1][:-2] if '%' in words[j+1] else words[j+1]))
        # function name
        funcName.append(words[0])
      #
      if endExacting: break

  # print(timeStat)
  #create the pandas dataframe
  return pd.DataFrame(timeStat, index=funcName)


def print_mg_time(df):
  '''
  Extract mg time at each layer from tiny profiling's summary.
  '''
  upTimes = []
  downTimes = []
  nLevel = 0
  while 1:
    upName = 'MLMG::mgVcycle_up::' + repr(nLevel)
    if upName in df.index:
      upTimes.append((df.loc[upName, 'Incl. Avg'], \
                      df.loc[upName, 'Incl. Avg'] / df.loc[upName, 'Incl. Max']))
      downName = 'MLMG::mgVcycle_down::' + repr(nLevel)
      downTimes.append((df.loc[downName, 'Incl. Avg'], \
                        df.loc[downName, 'Incl. Avg'] / df.loc[downName, 'Incl. Max']))
      nLevel += 1
    else:
      break
  # bottom time
  name = 'MLMG::mgVcycle_bottom'
  bottomTime = (df.loc[name, 'Incl. Avg'], \
                df.loc[name, 'Incl. Avg'] / df.loc[name, 'Incl. Max'])
  # print out levels' time
  for i in range(nLevel):
    sep    = 2*(nLevel - i) * '    '
    indent = (i + 1) * '    '
    print('{:d}{}({:.1f}, {:.1f}){}({:.1f}, {:.1f})'.format(i,\
          indent, downTimes[i][0], downTimes[i][1],\
          sep, upTimes[i][0], upTimes[i][1]))
  # print out bottom time
  indent = (nLevel + 3) * '    '
  print('{:d}{}({:.1f}, {:.1f})'.format(nLevel, \
         indent, bottomTime[0], bottomTime[1]))
