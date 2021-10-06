import sys, time, os, pdb, argparse, pickle, subprocess
from glob import glob
import numpy as np
from shutil import rmtree
import re
i=0
#print("YAHO")
#pdb.set_trace()

#command = ("glob("/home/nas/DB/RWF-2000_Dataset/Validation/Fight")")
#output = subprocess.call(command, shell=True, stdout=None)    
Validation = glob("/home/jinyoung/share/car_accident_dataset/Validation/*")
Train = glob("/home/jinyoung/share/car_accident_dataset/Train/*")
Test = glob("/home/jinyoung/share/car_accident_dataset/Test/*")
#/home/nas/DB/RWF-2000_Dataset/Test/NonFight
#/home/nas/DB/RWF-2000_Dataset/val/Fight
#/home/nas/DB/RWF-2000_Dataset/val/NonFight
#print(Test)
for name in Validation:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan != 'py':
        numbers = re.sub(r'[^0-9]', '', vid_n)
        Validation_name = numbers + '.' + expan
        print(Validation_name)
        command = ("mv /home/jinyoung/share/car_accident_dataset/Validation/%s /home/jinyoung/share/car_accident_dataset/Validation/%s"%(vid_n,Validation_name))
        output = subprocess.call(command, shell=True, stdout=None)    


for name in Train:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan != 'py':
        numbers = re.sub(r'[^0-9]', '', vid_n)
        Validation_name = numbers + '.' + expan
        print(Validation_name)
        command = ("mv /home/jinyoung/share/car_accident_dataset/Train/%s /home/jinyoung/share/car_accident_dataset/Train/%s"%(vid_n,Validation_name))
        output = subprocess.call(command, shell=True, stdout=None)    


for name in Test:
    vid_n = os.path.basename(name)
    print(vid_n)
    nn = vid_n.split(".")
    expan = nn[1]
    if expan != 'py':
        numbers = re.sub(r'[^0-9]', '', vid_n)
        Validation_name = numbers + '.' + expan
        print(Validation_name)
        command = ("mv /home/jinyoung/share/car_accident_dataset/Test/%s /home/jinyoung/share/car_accident_dataset/Test/%s"%(vid_n,Validation_name))
        output = subprocess.call(command, shell=True, stdout=None)    

command = ("mv /home/jinyoung/share/car_accident_dataset/Test/*.jams /home/jinyoung/share/car_accident_dataset/data/Test/")
output = subprocess.call(command, shell=True, stdout=None)
command = ("mv /home/jinyoung/share/car_accident_dataset/Test/*.txt /home/jinyoung/share/car_accident_dataset/data/Test/")
output = subprocess.call(command, shell=True, stdout=None)

command = ("mv /home/jinyoung/share/car_accident_dataset/Train/*.jams /home/jinyoung/share/car_accident_dataset/data/Train/")
output = subprocess.call(command, shell=True, stdout=None)
command = ("mv /home/jinyoung/share/car_accident_dataset/Train/*.txt /home/jinyoung/share/car_accident_dataset/data/Train/")
output = subprocess.call(command, shell=True, stdout=None)    


command = ("mv /home/jinyoung/share/car_accident_dataset/Validation/*.jams /home/jinyoung/share/car_accident_dataset/data/Validation/")
output = subprocess.call(command, shell=True, stdout=None)
command = ("mv /home/jinyoung/share/car_accident_dataset/Validation/*.txt /home/jinyoung/share/car_accident_dataset/data/Validation/")
output = subprocess.call(command, shell=True, stdout=None)
