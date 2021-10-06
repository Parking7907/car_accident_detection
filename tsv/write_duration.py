import sys, time, os, pdb, argparse, pickle, subprocess
from glob import glob
import numpy as np
from shutil import rmtree
import re
i=0

import numpy as np
import csv

f = open('./tsv/Test_duration.tsv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
title = ["filename","duration"]
wr.writerow(title)

alpha = glob("/home/jinyoung/share/car_accident_dataset/Test/*")
new = sorted(alpha)
for name in new:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan == 'wav':
        new_line = []
        new_line.append(vid_n)
        new_line.append("10.0")
        wr.writerow(new_line)
        print(new_line)

f.close()


f = open('./tsv/Train_duration.tsv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
title = ["filename","duration"]
wr.writerow(title)

alpha = glob("/home/jinyoung/share/car_accident_dataset/Train/*")
new = sorted(alpha)
for name in new:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan == 'wav':
        new_line = []
        new_line.append(vid_n)
        new_line.append("10.0")
        wr.writerow(new_line)
        print(new_line)

f.close()



f = open('./tsv/Validation_duration.tsv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f, delimiter='\t')
title = ["filename","duration"]
wr.writerow(title)

alpha = glob("/home/jinyoung/share/car_accident_dataset/Validation/*")
new = sorted(alpha)
for name in new:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan == 'wav':
        new_line = []
        new_line.append(vid_n)
        new_line.append("10.0")
        wr.writerow(new_line)
        print(new_line)

f.close()

