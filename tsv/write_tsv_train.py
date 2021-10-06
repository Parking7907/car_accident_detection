import sys, time, os, pdb, argparse, pickle, subprocess
from glob import glob
import numpy as np
from shutil import rmtree
import re
i=0

import numpy as np
import csv

f = open('./tsv/Train.tsv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
title = ["filename","onset","offset","event_label"]
wr.writerow(title)

alpha = glob("/home/jinyoung/share/car_accident_dataset/data/Train/*")

new = sorted(alpha)
#print(new)
for name in new:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    print(expan)
    if expan == 'txt':
        tx = open(name, 'r')
        while True:
            line = tx.readline()
            if not line: break
            new_line = []
            n_l = line.split('\t')
            f_name = nn[0] + ".wav"
            new_line.append(f_name)
            for i in range(2):
                new_line.append(n_l[i])
            new_line.append(n_l[2].split('\n')[0])

            wr.writerow(new_line)
            print(new_line)
        tx.close()

f.close()
