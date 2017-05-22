#!/usr/bin/env python

# This should be used as
# mkdir -p report
# grep '\(per_sec\|startup\)' log/*/*.log | python make-report.py

import sys
import re
from collections import defaultdict

stats = defaultdict(lambda: {})
allstats = defaultdict(lambda: [])

##### Regexes
fnameregex = re.compile(r"log/([a-z-]+?)(-gpu|)/(dynet-py|dynet-cpp|dynet-seq|chainer|theano|tensorflow)-(.*?)-t([123]).log:(.*)")
startregex = re.compile(r"startup time: (.*)")
eqregex = re.compile(r"(.*)=(.*)")
commentregex = re.compile(r"^ *((#|//).*)?")

##### Various data
canonicalize = {
  "word_per_sec": "speed",
  "words_per_sec": "speed",
  "sent_per_sec": "speed",
  "nll": "accuracy",
  "tag_acc": "accuracy",
  "acc": "accuracy",
  "time": "time"
}
taskna = {
  ("tensorflow", "bilstm-tagger-withchar"): 1,
  ("tensorflow", "treenn"): 1,
  ("theano", "treenn"): 1,
  ("dynet-seq", "bilstm-tagger"): 1,
  ("dynet-seq", "bilstm-tagger-withchar"): 1,
  ("dynet-seq", "treenn"): 1,
}
toolkits = ["dynet-cpp", "dynet-py", "chainer", "dynet-seq", "theano", "tensorflow"]
prettyname = {
  "dynet-cpp": "DyC++",
  "dynet-py":  "DyPy",
  "dynet-seq": "DyC++ Seq",
  "tensorflow":"TF",
  "chainer":   "Chainer",
  "theano":    "Theano"
}

##### Load from log files
for line in sys.stdin:
  line = line.replace("rnnlm-seq/dynet-cpp", "rnnlm-batch/dynet-seq")
  line = line.replace("rnnlm-seq-gpu/dynet-cpp", "rnnlm-batch-gpu/dynet-seq")
  m = re.search(fnameregex, line.strip())
  if m:
    task = m.group(1)
    device = "gpu" if m.group(2) == "-gpu" else "cpu"
    toolkit = m.group(3)
    params = m.group(4)
    trial = int(m.group(5))
    idtup = (task, device, toolkit, params, trial)
    data = m.group(6)
    m = re.search(startregex, data)
    if m:
      stats[idtup]["startup"] = float(m.group(1))
    else:
      mystats = {}
      for val in data.split(", "):
        m = re.search(eqregex, val)
        if not m:
          print("unmatched line: %s" % line)
          sys.exit(1)
        if m.group(1) in canonicalize:
          can = canonicalize[m.group(1)]
          val = float(m.group(2))
          mystats[can] = val
          if can == "accuracy":
            if "rnnlm" not in task: val *= 100
            else: val *= -1
            stats[idtup][can] = max(val, stats[idtup].get(can,-1e10))
          else:
            stats[idtup][can] = val
      allstats[idtup].append(mystats)
  else:
    print("unmatched line: %s" % line)
    sys.exit(1)
# print(stats)

# def format_num(num):
#   if num > 1e6:
#     return "%.03gM" % (float(num)/1e6)
#   elif num > 1e3:
#     return "%.03gk" % (float(num)/1e3)
#   else:
#     return "%.03g" % float(num)

# TODO: There must be a better way to do this...
def format_num(num):
  fnum = float(num)
  val = "%.03g" % fnum
  if fnum >= 1 and fnum < 10:
    val = "%.2f" % fnum
  elif fnum >= 10 and fnum < 100:
    val = "%.1f" % fnum
  elif float(num) > 1000:
    val = "%.f" % float(val)
  return val

def getmaxstat(task, device, toolkit, setting, stat, mult=1):
  my_stats = []
  for trial in range(1,4):
    my_id = (task, device, toolkit, setting, trial)
    if my_id in stats and stat in stats[my_id]:
      my_stats.append(mult*stats[my_id][stat])
  return format_num(mult*max(my_stats)) if len(my_stats) > 0 else "TODO"
def getminstat(task, device, toolkit, setting, stat):
  return getmaxstat(task, device ,toolkit, setting, stat, mult=-1)

###### First section: toolkit comparison

# CPU/GPU speeds for all toolkits/tasks
tasks = [
  ("RNNLM (MB=1) ", "rnnlm-batch", "ms01-es128-hs256-sp0"),
  ("RNNLM (MB=4)",  "rnnlm-batch", "ms04-es128-hs256-sp0"),
  ("RNNLM (MB=16)", "rnnlm-batch", "ms16-es128-hs256-sp0"),
  ("RNNLM (MB=64)", "rnnlm-batch", "ms64-es128-hs256-sp0"),
  ("BiLSTM Tag", "bilstm-tagger", "ws128-hs50-mlps32-su0"),
  ("BiLSTM Tag +sparse", "bilstm-tagger", "ws128-hs50-mlps32-su1"),
  ("BiLSTM Tag+Char", "bilstm-tagger-withchar", "cs20-ws128-hs50-mlps32-su0"),
  ("BiLSTM Tag+Char +sparse", "bilstm-tagger-withchar", "cs20-ws128-hs50-mlps32-su1"),
  ("TreeLSTM", "treenn", "ws128-hs128-su0"),
  ("TreeLSTM +sparse", "treenn", "ws128-hs128-su1"),
]
def make_speed_table(device):
  print("\\begin{table}")
  print("\\begin{tabular}{c|rrr|rrr}")
  print(" & "+" & ".join([prettyname[x] for x in toolkits])+" \\\\ \hline")
  for name, task, setting in tasks:
    cols = [name]
    for i, toolkit in enumerate(toolkits):
      if (toolkit, task) in taskna:
        cols.append("\\multicolumn{1}{c}{-}")
      else:
        cols.append(getmaxstat(task, device, toolkit, setting, "speed"))
    print(" & ".join(cols)+" \\\\")
  print("\\end{tabular}")
  print("\\caption{Processing speed for each toolkit on %s. Speeds are measured in words/sec for RNNLM and Tagger and sentences/sec for TreeLSTM.}" % device.upper())
  print("\\label{tab:speeds%s}" % device)
  print("\\end{table}")
  print("")
make_speed_table("cpu")
make_speed_table("gpu")

# Startup time table
tasks = [
  ("RNNLM", "rnnlm-batch", "ms01-es128-hs256-sp0"),
  ("BiLSTM Tag", "bilstm-tagger", "ws128-hs50-mlps32-su0"),
  ("BiLSTM Tag+Char", "bilstm-tagger-withchar", "cs20-ws128-hs50-mlps32-su0"),
  ("TreeLSTM", "treenn", "ws128-hs128-su0"),
]
print("\\begin{table}")
print("\\begin{tabular}{c|rrr|rrr}")
print(" & "+" & ".join([prettyname[x] for x in toolkits])+" \\\\ \hline")
for name, task, setting in tasks:
  cols = [name]
  for i, toolkit in enumerate(toolkits):
    if (toolkit, task) in taskna:
      cols.append("\\multicolumn{1}{c}{-}")
    else:
      cols.append(getminstat(task, device, toolkit, setting, "startup"))
  print(" & ".join(cols)+" \\\\")
print("\\end{tabular}")
print("\\caption{Startup time for programs written in each toolkit.}")
print("\\label{tab:startup}")
print("\\end{table}")
print("")

# Code complexities
def get_code_complexity(toolkit, task):
  chars = 0
  if toolkit == "dynet-seq":
    if not task == "rnnlm-batch":
      return "\\multicolumn{1}{c}{-}"
    toolkit = "dynet-cpp"
    task = "rnnlm-seq"
  if (toolkit, task) in taskna:
    return "\\multicolumn{1}{c}{-}"
  with open("%s/%s.%s" % (toolkit, task, "cc" if toolkit == "dynet-cpp" else "py"), "r") as f:
    for line in f:
      line = re.sub(commentregex, "", line.strip())
      chars += len(line)
  return str(chars)

tasks = [
  ("RNNLM", "rnnlm-batch"),
  ("BiLSTM Tag", "bilstm-tagger"),
  ("BiLSTM Tag+Char", "bilstm-tagger-withchar"),
  ("TreeLSTM", "treenn"),
]
print("\\begin{table}")
print("\\begin{tabular}{c|rrrrrr}")
print(" & "+" & ".join([prettyname[x] for x in toolkits])+" \\\\ \hline")
for name, task in tasks:
  cols = [name]
  for i, toolkit in enumerate(toolkits):
    cols.append(get_code_complexity(toolkit, task))
  print(" & ".join(cols)+" \\\\")
print("\\end{tabular}")
print("\\caption{Number of non-comment characters in the implementation of each toolkit.}")
print("\\label{tab:complexity}")
print("\\end{table}")
print("")


###### Second section: effect of minibatching and net size


###### Third section: effect of sparse update
tasks = [
  ("RNNLM (MB=1) ", "rnnlm-batch", "ms01-es128-hs256-sp"),
  ("RNNLM (MB=16)", "rnnlm-batch", "ms16-es128-hs256-sp"),
  ("BiLSTM Tag", "bilstm-tagger", "ws128-hs50-mlps32-su"),
  ("BiLSTM Tag+Char", "bilstm-tagger-withchar", "cs20-ws128-hs50-mlps32-su"),
  ("TreeLSTM", "treenn", "ws128-hs128-su"),
]
print("\\begin{table}")
print("\\begin{tabular}{c|rr|rr|rr|rr}")
print(" & \\multicolumn{4}{c|}{Speed} & \\multicolumn{4}{c}{Accuracy} \\\\")
print(" & \\multicolumn{2}{c|}{Dense} & \\multicolumn{2}{c|}{Sparse} & \\multicolumn{2}{c|}{Dense} & \\multicolumn{2}{c}{Sparse} \\\\")
print(" & "+" & ".join(["CPU & GPU"] * 4)+" \\\\ \\hline")
for name, task, setting in tasks:
  cols = [name]
  for criterion in ("speed", "accuracy"):
    for ds in ("0", "1"):
      for device in ("cpu", "gpu"):
        cols.append(getmaxstat(task, device, "dynet-cpp", setting+ds, criterion))
  print(" & ".join(cols)+" \\\\")
print("\\end{tabular}")
print("\\caption{Processing speed and accuracy after 10 minutes with dense or sparse updates.}")
print("\\label{tab:sparseresults}")
print("\\end{table}")
print("")
