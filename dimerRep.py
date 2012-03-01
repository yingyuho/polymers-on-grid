#########################################################################
# dimerRep.py
#
# This simulates the diffusion of polymers consisting of monomers and 
# two types of dimers with I-shape and L-shape using Monte Carlo method.
#
# The translational move set is {left, right, up, down}. 
# The rotational move set is {rotateCW, rotateCCW}. 
# A movement could be rejected if there is repulsion or adsorption energy.
#
#########################################################################

from numpy import *
from numpy.fft import *
from numpy.random import rand, randint

from lattice import Atom, Molecule, Lattice

class Experiment:

	ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = 0, 1, 2, 3
	MOL_MONOMER, MOL_DIMER = 0, 1
	
	# Monomer Temp
	moleculeTemp1 = map(Atom.fromTuple, [\
		(ATOM_MONOMER,0,0)])
	# I Dimer
	moleculeTemp2 = map(Atom.fromTuple, [\
		(ATOM_DIMER1,	0,	0),		(ATOM_DIMER1,	0,	1), \
		(ATOM_SITE,		1,	0),		(ATOM_SITE,		1,	1), \
		(ATOM_SITE,		-1,	0),		(ATOM_SITE,		-1,	1)])
	# L Dimer
	moleculeTemp3 = map(Atom.fromTuple, [\
		(ATOM_DIMER2,	0,	-1),	(ATOM_DIMER2,	-1,	0), \
		(ATOM_SITE,		1,	-1),	(ATOM_SITE,		-1,	1), \
		(ATOM_SITE,		0,	0,	2)])
	
	def __init__(\
			self, dimX, dimY, energyScale, \
			monomerAppear, monomerVanish, dimerVanish, dimerBreak, \
			monomerInitNum, dimer1InitNum, dimer2InitNum):
			
		MOL_MONOMER, MOL_DIMER =  self.MOL_MONOMER,  self.MOL_DIMER
		
		self.lat 			= Lattice(dimX, dimY)
		self.energyScale	= energyScale
		
		self.monomerNum 	= monomerInitNum
		self.dimer1Num 		= dimer1InitNum
		self.dimer2Num 		= dimer2InitNum
		
		self.monomerAppear	= monomerAppear
		self.monomerVanish	= monomerVanish
		self.dimerVanish	= dimerVanish
		self.dimerBreak		= dimerBreak
				
		self.__initVars()
		
		self.lat.setAtomTypeNum(4)
		self.lat.setEnergyMatrix(array([\
			[ 1,  1,  1,  0],\
			[ 1,  1,  1,  0],\
			[ 1,  1,  1,  0],\
			[ 0,  0,  0,  0]]) * self.energyScale)
											
		self.lat.bindingEnergy = -1.0 * self.energyScale;
		
		mNum = monomerInitNum + dimer1InitNum + dimer2InitNum
		self.moleculeNum = mNum
		
		initX = randint(0, self.lat.dimX, mNum)
		initY = randint(0, self.lat.dimY, mNum)
		
		for n in range(self.monomerNum):
			self.lat.addMolecule(Molecule(MOL_MONOMER, initX[n], initY[n], self.moleculeTemp1))
		for n in range(self.dimer1Num):
			self.lat.addMolecule(Molecule(MOL_DIMER,   initX[n], initY[n], self.moleculeTemp2))
		for n in range(self.dimer2Num):
			self.lat.addMolecule(Molecule(MOL_DIMER,   initX[n], initY[n], self.moleculeTemp3))
		
	def __initVars(self):	
		ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = \
		self.ATOM_MONOMER, self.ATOM_DIMER1, self.ATOM_DIMER2, self.ATOM_SITE
		
		MOL_MONOMER, MOL_DIMER =  self.MOL_MONOMER,  self.MOL_DIMER

		self.newDimerTemp = range(8)
		self.newDimerTemp[0] = map(Atom.fromTuple, [\
			(ATOM_DIMER1,	0,	0), 	(ATOM_DIMER1,	1,	0), \
			(ATOM_SITE,		0,	1),		(ATOM_SITE,		1,	1), \
			(ATOM_SITE,		0,	-1),	(ATOM_SITE,		1,	-1)] )
		self.newDimerTemp[1] = map(Atom.fromTuple, [\
			(ATOM_DIMER1,	0,	0),		(ATOM_DIMER1,	-1,	0), \
			(ATOM_SITE,		0,	1),		(ATOM_SITE,		-1,	1), \
			(ATOM_SITE,		0,	-1),	(ATOM_SITE,		-1,	-1)] )
		self.newDimerTemp[2] = map(Atom.fromTuple, [\
			(ATOM_DIMER1,	0,	0),		(ATOM_DIMER1,	0,	1), \
			(ATOM_SITE,		1,	0),		(ATOM_SITE,		1,	1), \
			(ATOM_SITE,		-1,	0),		(ATOM_SITE,		-1,	1)] )
		self.newDimerTemp[3] = map(Atom.fromTuple, [\
			(ATOM_DIMER1,	0,	0),		(ATOM_DIMER1,	0,	-1), \
			(ATOM_SITE,		1,	0),		(ATOM_SITE,		1,	-1), \
			(ATOM_SITE,		-1,	0),		(ATOM_SITE,		-1,	-1)] )
		self.newDimerTemp[4] = map(Atom.fromTuple, [\
			(ATOM_DIMER2,	0,	1),		(ATOM_DIMER2,	-1,	0), \
			(ATOM_SITE,		1,	1),		(ATOM_SITE,		-1,	-1),\
			(ATOM_SITE,		0,	0,	2)])
		self.newDimerTemp[5] = map(Atom.fromTuple, [\
			(ATOM_DIMER2,	0,	1),		(ATOM_DIMER2,	1,	0), \
			(ATOM_SITE,		-1,	1),		(ATOM_SITE,		1,	-1),\
			(ATOM_SITE,		0,	0,	2)])
		self.newDimerTemp[6] = map(Atom.fromTuple, [\
			(ATOM_DIMER2,	0,	-1),	(ATOM_DIMER2,	1,	0), \
			(ATOM_SITE,		-1,	-1),	(ATOM_SITE,		1,	1), \
			(ATOM_SITE,		0,	0,	2)])
		self.newDimerTemp[7] = map(Atom.fromTuple, [\
			(ATOM_DIMER2,	0,	-1),	(ATOM_DIMER2,	-1,	0), \
			(ATOM_SITE,		1,	-1),	(ATOM_SITE,		-1,	1), \
			(ATOM_SITE,		0,	0,	2)])
		self.reactDirMap = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]]
		
	colorScale = ( [
		[0.20, 0.20, 0.20],\
		[0.30, 0.00, 0.00],\
		[0.00, 0.30, 0.00],\
		[0.05, 0.05, 0.05]] )
		
	def colorMap(self, lattice):
		return clip(tensordot(lattice.atomNumMap, self.colorScale, axes=([0,0])), 0, 1)
		
	def run(self, simulationTime, display, outputFileName):
		simTime		= simulationTime
		#display		= display
		#outputFileName	= outputFileName
		
		lat = self.lat
		reactDirMap = self.reactDirMap
		moleculeNum = self.moleculeNum
		moleculeTemp1 = self.moleculeTemp1
		newDimerTemp = self.newDimerTemp
		ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = \
			self.ATOM_MONOMER, self.ATOM_DIMER1, self.ATOM_DIMER2, self.ATOM_SITE
		MOL_MONOMER, MOL_DIMER = self.MOL_MONOMER,  self.MOL_DIMER

		
		if outputFileName == None:
			fileFlag = False
		else:
			fileFlag = True
			file = open(outputFileName, 'w')
			
		if display:
			fig = figure()
			img = imshow(self.colorMap(lat), interpolation='none')
			ioff()
		
		tMoveMol 		= zeros(simTime, int)
		tMoveDir 		= zeros(simTime, int)
		tMoveDis 		= zeros(simTime, int)
		tMoveProb 		= zeros(simTime)
		
		rMoveMol 		= zeros(simTime, int)
		rMoveDir 		= zeros(simTime, int)
		rMoveProb 		= zeros(simTime)
		
		reactMol 		= zeros(simTime, int)
		reactDir 		= zeros(simTime, int)
		reactAtom 		= zeros(simTime, int)
		
		degProb 		= rand(simTime)
		
		newX 			= zeros(simTime, int)
		newY 			= zeros(simTime, int)
		newProb 		= rand(simTime)
		
		bindProb 		= rand(simTime)
		
		for t in range(simTime):

			# Translate one molecule
			tMoveMol[t] = randint(0, lat.moleculeNum)
			tMoveDir[t] = randint(0, 2)
			tMoveDis[t] = randint(0, 2) * 2 - 1
			tMoveProb[t] = rand()

			dx = tMoveDis[t] * tMoveDir[t]
			dy = tMoveDis[t] * (1 - tMoveDir[t])
			lat.translateMoleculeByIndex(tMoveMol[t], dx, dy, tMoveProb[t])
			
			# Rotate one molecule
			rMoveDir[t] = randint(0, 2)
			rMoveMol[t] = randint(0, lat.moleculeNum)
			rMoveProb[t] = rand()
			
			if rMoveDir[t] == 0:
				lat.rotateMoleculeCWByIndex(rMoveMol[t], rMoveProb[t])
			else:
				lat.rotateMoleculeCCWByIndex(rMoveMol[t], rMoveProb[t])
			
			# Pick one monomer atom and see if a dimer can be formed
			molN = len(lat.atomList[ATOM_MONOMER]) + len(lat.atomList[ATOM_DIMER1])/2 + len(lat.atomList[ATOM_DIMER2])/2;
			reactAtom[t] = randint(0, molN)
			
			if reactAtom[t] < len(lat.atomList[ATOM_MONOMER]):
				ra = lat.atomList[ATOM_MONOMER][reactAtom[t]]

				rd = reactDir[t] = randint(0, 8)
				
				x2 = ra.x + reactDirMap[rd][0]
				y2 = ra.y + reactDirMap[rd][1]
			
				#anl = lat.getAtomNumAt(x2, y2)
				
				if 0 < lat.getDataXY(lat.atomNumMap[ATOM_MONOMER], x2, y2):
					for ra2 in lat.getAtomListAt(x2, y2):
						if ra2.type == ATOM_MONOMER:
							# I-shaped dimer
							if rd < 4:
								lat.addMolecule(Molecule(MOL_DIMER, ra.x, ra.y, newDimerTemp[rd]))
							# L-shaped dimer
							else:
								lat.addMolecule(Molecule(MOL_DIMER, ra2.x, ra.y, newDimerTemp[rd]))
							
							# Remove the involved monomers
							for atom in [ra, ra2]:
								rm = atom.parent
								if rm.type == MOL_MONOMER:
									lat.removeMolecule(rm)
								else:
									lat.raiseMolecule(rm)
									atom.type = ATOM_SITE
									lat.layMolecule(rm)							
							break
			
			
			reactMol[t] = randint(0, lat.moleculeNum)
			rm = lat.moleculeList[reactMol[t]]
			
			# Death of monomer
			if rm.type == MOL_MONOMER:
				if degProb[t] < self.monomerVanish:
					lat.removeMolecule(rm)						
			elif (rm.type == MOL_DIMER):
				# Death of dimer
				if degProb[t] < self.dimerVanish:
					for atom in rm.atomList:
						if atom.type == ATOM_MONOMER:
							lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, moleculeTemp1))
					lat.removeMolecule(rm)
					moleculeNum -= 1
				# Degradation of dimer into    monomers
				elif degProb[t] > (1-self.dimerBreak):
					for atom in rm.atomList:
						if atom.type != ATOM_SITE:
							lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, moleculeTemp1))
					lat.removeMolecule(rm)
				else:
					for atom in rm.atomList:
						# Attach monomer to adsorption site of dimer
						if atom.type == ATOM_SITE:
							aL1 = lat.getAtomListAt(atom.x, atom.y)
							for atom1 in filter(lambda x: x.parent.type == MOL_MONOMER, aL1):
								lat.raiseMolecule(rm)
								atom.type = ATOM_MONOMER
								lat.layMolecule(rm)
								lat.removeMolecule(atom1.parent)
								break
						# Detach monomer from adsorption site of dimer
						elif atom.type == ATOM_MONOMER:
							if bindProb[t] < min(1,exp(atom.copies*lat.bindingEnergy)):
								lat.raiseMolecule(rm)
								atom.type = ATOM_SITE
								lat.layMolecule(rm)
								lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, moleculeTemp1))
				
			# Add monomers randomly to the system
			if newProb[t] < self.monomerAppear/lat.moleculeNum:
				newX[t] = randint(floor(lat.dimX*0.0), floor(lat.dimX*1.0))
				newY[t] = randint(floor(lat.dimY*0.0), floor(lat.dimY*1.0))
				lat.addMolecule(Molecule(MOL_MONOMER, newX[t], newY[t], moleculeTemp1))
			
			# Update the animation per ## moves
			if (t+1)%(200) == 0:
				if display:
					img.set_array(self.colorMap(lat))
					draw()
				#monomerNumMapFT = fft2(lat.atomNumMap[ATOM_DIMER2])
				#corrFunction = int32(real(ifft2( monomerNumMapFT * conj(monomerNumMapFT) )).round())
				#for row in corrFunction:
				#	for col in row:
				#		file.write('{} '.format(col))
				#	file.write('\n')
				#file.write('\n')
				
				#for row in corrFunction:
				#	for col in row:
				#		print('{}'.format(col)),
				#	print '\n',
				#print '\n'
				if fileFlag:
					for atomType in lat.atomNumMap:
						for row in atomType:
							for col in row:
								file.write('{} '.format(col))
							file.write('\n')
						file.write('\n')
					file.write('\n')
					#str = '{} {} {}\n'.format(\
					#	len(lat.atomList[ATOM_MONOMER]), \
					#	len(lat.atomList[ATOM_DIMER1]), \
					#	len(lat.atomList[ATOM_DIMER2]))
					#file.write(str)
					#print(str),
		file.close()
	

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('-d', '--disp', dest='display', action='store_true')
	parser.add_argument('-s', '--size', dest='size', action='store', type=int, default=32)
	parser.add_argument('-o', '--out', dest="outputFileName", action='store')
	parser.add_argument('-e', '--energy', dest="energyScale", action='store', type=float, default=1.)
	parser.add_argument('--ma', dest="ma", action='store', type=float, default=0)
	parser.add_argument('--mv', dest="mv", action='store', type=float, default=0)
	parser.add_argument('--dv', dest="dv", action='store', type=float, default=0)
	parser.add_argument('--db', dest="db", action='store', type=float, default=0)
	parser.add_argument('--nm', dest='nm', action='store', type=int, default=0)
	parser.add_argument('--nd1', dest='nd1', action='store', type=int, default=0)
	parser.add_argument('--nd2', dest='nd2', action='store', type=int, default=0)
	opt = parser.parse_args()
	
#	if	opt.outputFileName != None:
#		fileFlag = True
#	else:
#		fileFlag = False
		
	if opt.display == True:
		from matplotlib.pyplot import *
		import matplotlib.animation as animation
		
	experiment = Experiment(\
		dimX=opt.size, dimY=opt.size, energyScale=opt.energyScale, \
		monomerAppear=opt.ma, monomerVanish=opt.mv, dimerVanish=opt.dv, dimerBreak=opt.db, \
		monomerInitNum=opt.nm, dimer1InitNum=opt.nd1, dimer2InitNum=opt.nd2)
		
	experiment.run(simulationTime=100*50, display=opt.display, outputFileName=opt.outputFileName)
	
	