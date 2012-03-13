from numpy import *
from numpy.fft import *

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

def readFrame(file):
	frame = []
	tbl = readAtomType(file)
	while len(tbl) > 0:
		frame.append(tbl)
		tbl = readAtomType(file)
	return frame

import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('-i', dest="inFileName", action='store')
parser.add_argument('-o', dest="outFileName", action='store')
parser.add_argument('-f', dest='function', action='store')
parser.add_argument('-t', dest='interval', action='store', type=int, default=200)
opt = parser.parse_args()

ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = 0, 1, 2, 3

inFile = open(opt.inFileName, 'r')
if opt.outFileName != None:
	outFlag = True
else:
	outFlag = False
	
if outFlag:	outFile = open(opt.outFileName, 'w')

if opt.function == 'replay':
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
	
elif opt.function == 'count' and outFlag:
	frame = array(readFrame(inFile))
	while len(frame) > 0:
		atomCount = frame.sum(2).sum(1)
		for num in atomCount:
			outFile.write('{} '.format(num))
		outFile.write('\n')
		frame = array(readFrame(inFile))
		
elif opt.function == 'scorr' and outFlag:
	atomTypeA = ATOM_MONOMER
	atomTypeB = ATOM_DIMER2
	
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
	frame = readFrame(inFile)
	corrSum = zeros(array(frame[0]).shape)
	area = corrSum.shape[0] * corrSum.shape[1]
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
		frameA = array(frame[atomTypeA])
		frameB = array(frame[atomTypeB])
		frameA = frameA * area - sum(frameA)
		frameB = frameB * area - sum(frameB)
		ftA = fft2(frameA)
		ftB = fft2(frameB)
		corr = real(ifft2( ftA * conj(ftB) ))
		#corr = float(real(ifft2( ftA * conj(ftB) )).round())
		corrSum = corrSum + corr		
		frame = readFrame(inFile)

	corrSum = 1. * corrSum / totalTime / area / area
	
	for row in corrSum:
		for col in row:
			outFile.write('{} '.format(col))
		outFile.write('\n')
	outFile.write('\n')

	
inFile.close()
if outFlag:	outFile.close()
