from numpy import *
from numpy.fft import *

colorScale = ( [
	[0.20, 0.20, 0.20],\
	[0.40, 0.00, 0.00],\
	[0.00, 0.30, 0.00],\
	[0.00, 0.00, 0.00]] )
	
def colorMap(atomNumMap):
	return clip(tensordot(atomNumMap, colorScale, axes=([0,0])), 0, 1)
	
def readRow(file):
	line = file.readline()
	while line.startswith("#"): line = file.readline()
	lst = map(int, line.split())
	return lst
		
def readAtomType(file):
	tbl = []
	lst = readRow(file)
	while len(lst) > 0:
		tbl.append(lst)
		lst = readRow(file)
	return tbl

def readFrame(file):
	frame = []
	tbl = readAtomType(file)
	while len(tbl) > 0:
		frame.append(tbl)
		tbl = readAtomType(file)
	return frame
	
def readRowSharp(file):
	line = file.readline()
	while not line.startswith("#"): line = file.readline()
	lst = map(int, line.lstrip('# ').split())
	return lst

def readNewDimerNum(file):
	tbl = []
	line = file.readline()
	while not line.startswith("# New Dimer"):
		if line == "": return tbl
		line = file.readline()
	lst = readRowSharp(file)
	while len(lst) > 0:
		tbl.append(lst)
		lst = readRowSharp(file)
	return tbl

import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('-i', dest="inFileName", action='store')
parser.add_argument('-o', dest="outFileName", action='store')
parser.add_argument('-f', dest='function', action='store', nargs='+')
parser.add_argument('-t', dest='interval', action='store', type=int, default=200)
opt = parser.parse_args()
func = opt.function[0]

ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = 0, 1, 2, 3

inFile = open(opt.inFileName, 'r')
if opt.outFileName != None:
	outFlag = True
else:
	outFlag = False
	
if outFlag:	outFile = open(opt.outFileName, 'w')

if func == 'replay':
	import matplotlib.pyplot as plt
	import matplotlib.animation as animation

	frame = readFrame(inFile)

	fig = plt.figure()
	img = plt.imshow(colorMap(frame), interpolation='none')

	def update(*args):
		frame = readFrame(inFile)
		if len(frame)>0:
			img.set_array(colorMap(frame))
			return img,
		else:
			return []

	ani = animation.FuncAnimation(fig, update, interval=opt.interval, blit=True, repeat=False)
	plt.show()	
	
elif func == 'coarse' and outFlag and len(opt.function) >= 2:
	boxX = int(opt.function[1])
	if len(opt.function) >= 3:
		boxY = int(opt.function[2])
	else:
		boxY = boxX
	
	frame = array(readFrame(inFile))
	aNum = frame.shape[0]
	oldDimX, oldDimY = frame.shape[1], frame.shape[2]
	newDimX, newDimY = oldDimX/boxX, oldDimY/boxY
	
	while len(frame) > 0:
		newFrame = reshape(frame,(aNum,oldDimX,newDimY,boxY))
		newFrame = sum(newFrame,3)
		newFrame = transpose(newFrame,(0,2,1))
		newFrame = reshape(newFrame,(aNum,newDimY,newDimX,boxX))
		newFrame = sum(newFrame,3)
		newFrame = transpose(newFrame,(0,2,1))
		
		for atomType in newFrame:
			for row in atomType:
				for col in row:
					outFile.write('{} '.format(col))
				outFile.write('\n')
			outFile.write('\n')
		outFile.write('\n')
		
		frame = array(readFrame(inFile))
	
elif func == 'count' and outFlag:
	frame = array(readFrame(inFile))
	while len(frame) > 0:
		atomCount = frame.sum(2).sum(1)
		for num in atomCount:
			outFile.write('{} '.format(num))
		outFile.write('\n')
		frame = array(readFrame(inFile))
		
elif func == 'countrxn' and outFlag:
	frame = array(readNewDimerNum(inFile))
	#frameSum = zeros(frame.shape, int)
	while len(frame) > 0:
		#frameSum = frameSum + frame
		for row in frame:
			for col in row:
				outFile.write('{} '.format(col))
			outFile.write('\n')
		outFile.write('\n')
		frame = array(readNewDimerNum(inFile))
		
	#for row in frameSum:
	#	for col in row:
	#		outFile.write('{} '.format(col))
	#	outFile.write('\n')
	#outFile.write('\n')

elif func == 'scorr' and outFlag:
	#atomTypeA = ATOM_MONOMER
	#atomTypeB = ATOM_DIMER2
	atomTypeNum = 3
	#frame = array(readFrame(inFile), float)
	#frameSum = zeros(array(frame).shape, float)
	#area = frameSum.shape[1] * frameSum.shape[2]
	#totalTime = 0
	#
	#while len(frame) > 0:
	#	totalTime = totalTime + 1
	#	frameSum = frameSum + frame
	#	frame = array(readFrame(inFile), float)
	#	
	#
    #
	#ftA = fft2(frameSum[atomTypeA])
	#ftB = fft2(frameSum[atomTypeB])
	#sumMoment = float(real(ifft2( ftA * conj(ftB) )).round())
	
	inFile.seek(0)
	frame = array(readFrame(inFile))
	corrSum = zeros((atomTypeNum,atomTypeNum)+frame[0].shape)
	area = corrSum.shape[2] * corrSum.shape[3]
	totalTime = 0
	
#	for atomType in frameSum:
#		for row in atomType:
#			for col in row:
#				outFile.write('{} '.format(col))
#			outFile.write('\n')
#		outFile.write('\n')
#	outFile.write('\n')

#	for row in sumMoment:
#		for col in row:
#			outFile.write('{} '.format(col))
#		outFile.write('\n')
#	outFile.write('\n')
	
	while len(frame) > 0:
		totalTime = totalTime + 1
		for atomTypeA in range(atomTypeNum):
			for atomTypeB in range(atomTypeNum):
				frameA = frame[atomTypeA]
				frameB = frame[atomTypeB]
				frameA = frameA * area - sum(frameA)
				frameB = frameB * area - sum(frameB)
				ftA = fft2(frameA)
				ftB = fft2(frameB)
				corr = real(ifft2( ftA * conj(ftB) ))
				corrSum[atomTypeA][atomTypeB] = corrSum[atomTypeA][atomTypeB] + corr
		#corr = float(real(ifft2( ftA * conj(ftB) )).round())
		
		frame = array(readFrame(inFile))

	corrSum = 1. * corrSum / totalTime / area / area
	
	
	for atomTypeA in corrSum:
		for atomTypeB in atomTypeA:
			for row in atomTypeB:
				for col in row:
					outFile.write('{} '.format(col))
				outFile.write('\n')
			outFile.write('\n')
		outFile.write('\n')
	outFile.write('\n')
	
inFile.close()
if outFlag:	outFile.close()
