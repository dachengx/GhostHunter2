# -*- coding: utf-8 -*-

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input file')
psr.add_argument('-o', dest='opt', help='output symbolic link to net')
psr.add_argument('-P', '--pretrained', dest='pretrained', type=str)
args = psr.parse_args()
output = args.opt
nettxt = args.pretrained

import os
import re

inputdir = os.path.dirname(args.ipt)
inputfile = os.path.split(args.ipt)[-1]

fileSet = os.listdir(inputdir)
matchrule = re.compile(r'(\d)_epoch(\d+)_loss(\d+(\.\d*)?|\.\d+)')
NetLoss_reciprocal = []
for filename in fileSet :
    if '_epoch' in filename : NetLoss_reciprocal.append(1 / float(matchrule.match(filename)[3]))
    else : NetLoss_reciprocal.append(0)
net_name = fileSet[NetLoss_reciprocal.index(max(NetLoss_reciprocal))]
modelpath = inputdir + '/' + net_name

with open(nettxt, 'w') as fp:
    fp.write(modelpath)
    fp.close()

os.system('ln -s ' + modelpath + ' ' + output)
