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
fnameregex = re.compile(r"log/([a-z-]+?)(-gpu|)/(dynet-py|dynet-cpp|chainer|theano|tensorflow)-(.*?)-t([123]).log:(.*)")
startregex = re.compile(r"startup time: (.*)")
eqregex = re.compile(r"(.*)=(.*)")

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
}
toolkits = ["dynet-cpp", "dynet-py", "chainer", "theano", "tensorflow"]
prettyname = {
  "dynet-cpp": "DyNet C++",
  "dynet-py":  "DyNet Py",
  "tensorflow":"TensorFlow",
  "chainer":   "Chainer",
  "theano":    "Theano"
}

##### Load data
for line in sys.stdin:
  m = re.search(fnameregex, line.strip())
  if m:
    task = m.group(1)
    device = "gpu" if m.group(2) == "gpu" else "cpu"
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
          mystats[can] = float(m.group(2))
          val = float(m.group(2))
          if can == "accuracy":
            if task != "rnnlm-batch":
              val *= 100
            stats[idtup][can] = max(val, stats[idtup].get(can,-1e10))
          else:
            stats[idtup][can] = val
      allstats[idtup].append(mystats)
  else:
    print("unmatched line: %s" % line)
    sys.exit(1)
# print(stats)

def getmaxstat(task, device, toolkit, setting, stat):
  my_stats = []
  for trail in range(1,4):
    my_id = (task, device, toolkit, setting, trial)
    if my_id in stats and stat in stats[my_id]:
      my_stats.append(stats[my_id][stat])
  return "%.2f" % max(my_stats) if len(my_stats) else "TODO"

###### First table: CPU/GPU speeds for all toolkits/tasks
tasks = [
  ("RNNLM (MB=1) ", "rnnlm-batch", "ms01-es128-hs256-sp0"),
  ("RNNLM (MB=16)", "rnnlm-batch", "ms16-es128-hs256-sp0"),
  ("BiLSTM Tagger", "bilstm-tagger", "ws128-hs50-mlps32-su0"),
  ("BiLSTM Tagger w/Char", "bilstm-tagger-withchar", "cs20-ws128-hs50-mlps32-su0"),
  ("TreeLSTM", "treenn", "ws128-hs128-su0"),
]
def make_speed_table(device):
  print("\\begin{table}")
  print("\\begin{tabular}{c|r|r|r|r|r}")
  print(" & "+" & ".join([prettyname[x] for x in toolkits])+" \\\\ \hline")
  for name, task, setting in tasks:
    cols = [name]
    for i, toolkit in enumerate(toolkits):
      if (toolkit, task) in taskna:
        cols.append("\\multicolumn{1}{"+("c" if i == len(toolkits)-1 else "c|")+"}{-}")
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

###### Second table: effect of sparse update
tasks = [
  ("RNNLM (MB=1) ", "rnnlm-batch", "ms01-es128-hs256-sp"),
  ("RNNLM (MB=16)", "rnnlm-batch", "ms16-es128-hs256-sp"),
  ("BiLSTM Tagger", "bilstm-tagger", "ws128-hs50-mlps32-su"),
  ("BiLSTM Tagger w/Char", "bilstm-tagger-withchar", "cs20-ws128-hs50-mlps32-su"),
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
print("\\caption{Processing speed with dense or sparse updates.}")
print("\\label{tab:speeds}")
print("\\end{table}")
print("")
