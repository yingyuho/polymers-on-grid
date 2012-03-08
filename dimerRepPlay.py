from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

colorScale = ( [
	[0.20, 0.20, 0.20],\
	[0.30, 0.00, 0.00],\
	[0.00, 0.30, 0.00],\
	[0.00, 0.00, 0.00]] )
	
def colorMap(atomNumMap):
	return clip(tensordot(atomNumMap, colorScale, axes=([0,0])), 0, 1)
	
def readRow(file):
	lst = map(int, file.readline().split())
	return lst
	
def readAtomType(file):
	tbl = []
	lst = readRow(file)
	while len(lst) > 0:
		tbl.append(lst)
		lst = readRow(file)
	return tbl

def readFrame():
	frame = []
	tbl = readAtomType(file)
	while len(tbl) > 0:
		frame.append(tbl)
		tbl = readAtomType(file)
	return frame

import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('-f', dest="fileName", action='store')
opt = parser.parse_args()

file = open(opt.fileName, 'r')

frame = readFrame()

fig = plt.figure()
img = plt.imshow(colorMap(frame), interpolation='none')

def update(*args):
	frame = readFrame()
	if len(frame)>0:
		img.set_array(colorMap(frame))
		return img,
	else:
		return []

ani = animation.FuncAnimation(fig, update, interval=200, blit=True, repeat=False)
plt.show()


file.close()
