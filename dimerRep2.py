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
from operator import itemgetter, attrgetter
from time import sleep


from lattice import Atom, Molecule, Lattice

class Experiment:

	ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = 0, 1, 2, 3
	MOL_MONOMER, MOL_DIMER1, MOL_DIMER2 = 0, 1, 2
	
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
			self, dimX, dimY, energyScale, rate1, rate2, cat, \
			monomerAppear, monomerVanish, dimerVanish, dimerBreak, \
			monomerInitNum, dimer1InitNum, dimer2InitNum):
			
		MOL_MONOMER, MOL_DIMER1, MOL_DIMER2 =  self.MOL_MONOMER,  self.MOL_DIMER1,  self.MOL_DIMER2
		
		self.lat 			= Lattice(dimX, dimY)
		self.energyScale	= energyScale
		self.rate1			= rate1
		self.rate2			= rate2
		self.cat			= cat
		
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
			[ 2,  2,  2,  0],\
			[ 2,  2,  2,  0],\
			[ 2,  2,  2,  0],\
			[ 0,  0,  0,  0]]) * self.energyScale)
											
		self.lat.bindingEnergy = -1.0 * self.energyScale;
		
		mNum = monomerInitNum + dimer1InitNum + dimer2InitNum
		self.moleculeNum = mNum
		
		initX = randint(0, self.lat.dimX, mNum)
		initY = randint(0, self.lat.dimY, mNum)
		
		for n in range(self.monomerNum):
			self.lat.addMolecule(Molecule(MOL_MONOMER, initX[n], initY[n], self.moleculeTemp1))
		for n in range(self.dimer1Num):
			self.lat.addMolecule(Molecule(MOL_DIMER1,   initX[n], initY[n], self.moleculeTemp2))
		for n in range(self.dimer2Num):
			self.lat.addMolecule(Molecule(MOL_DIMER2,   initX[n], initY[n], self.moleculeTemp3))
		
	def __initVars(self):	
		ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = \
		self.ATOM_MONOMER, self.ATOM_DIMER1, self.ATOM_DIMER2, self.ATOM_SITE
		
		MOL_MONOMER, MOL_DIMER1, MOL_DIMER2 =  self.MOL_MONOMER,  self.MOL_DIMER1,  self.MOL_DIMER2

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
		
	def run(self, simulationTime, outputPeriod, display, outputFileName):
		simTime		= simulationTime
		#display		= display
		#outputFileName	= outputFileName
		
		lat = self.lat
		colorMap = self.colorMap
		reactDirMap = self.reactDirMap
		moleculeNum = self.moleculeNum
		moleculeTemp1 = self.moleculeTemp1
		newDimerTemp = self.newDimerTemp
		ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = \
			self.ATOM_MONOMER, self.ATOM_DIMER1, self.ATOM_DIMER2, self.ATOM_SITE
		MOL_MONOMER, MOL_DIMER1, MOL_DIMER2 =  self.MOL_MONOMER,  self.MOL_DIMER1,  self.MOL_DIMER2

		
		if outputFileName == None:
			fileFlag = False
		else:
			fileFlag = True
			file = open(outputFileName, 'w')
			
		if display:
			fig = figure()
			img = imshow(self.colorMap(lat), interpolation='none')
			ioff()
		
		lat.newDimerNum = zeros((3,3), int)
		
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
		reactProb		= rand(simTime)
		
		degProb 		= rand(simTime)
		
		newX 			= zeros(simTime, int)
		newY 			= zeros(simTime, int)
		newProb 		= rand(simTime)
		
		bindProb 		= rand(simTime)
		
		def expRandTime(rate=1.):
			if rate > 0.:
				return -log(rand())/rate
			else:
				return float('inf')
		
		class Movement:
			def reset(self):
				newSpeed = float(self.speedFunc())
				if newSpeed > 0.:
					self.wait = self.timeFunc()/newSpeed
					self.currentSpeed = newSpeed
				else:
					self.wait = float('inf')
					self.currentSpeed = 0.
				return self.wait
			def updateParam(self):
				newSpeed = float(self.speedFunc())
				if newSpeed == self.currentSpeed:
					pass
				elif self.currentSpeed > 0.:
					if newSpeed > 0.:
						self.wait = self.wait * (self.currentSpeed/newSpeed)
						self.currentSpeed = newSpeed
					else:
						self.wait = float('inf')
						self.currentSpeed = 0.
				else:
					return self.reset()
			def __call__(self):
				return self.moveFunc(self)
			def __init__(self,moveFunc,timeFunc,speedFunc):
				self.moveFunc = moveFunc
				self.timeFunc = timeFunc
				self.speedFunc = speedFunc
				self.currentSpeed = 0.
				self.wait = 0.
				
		def doTranslation(self):
			# Translate one molecule
			tMoveMol = randint(0, lat.moleculeNum)
			tMoveDir = randint(0, 2)
			tMoveDis = randint(0, 2) * 2 - 1
			tMoveProb = rand()

			dx = tMoveDis * tMoveDir
			dy = tMoveDis * (1 - tMoveDir)
			lat.translateMoleculeByIndex(tMoveMol, dx, dy, tMoveProb)
			return False
		def doRotation(self):
			# Rotate one molecule
			rMoveDir = randint(0, 2)
			rMoveMol = randint(0, lat.moleculeNum)
			rMoveProb = rand()
			
			if lat.moleculeList[rMoveMol] != MOL_MONOMER:
				if rMoveDir == 0:
					lat.rotateMoleculeCWByIndex(rMoveMol, rMoveProb)
				else:
					lat.rotateMoleculeCCWByIndex(rMoveMol, rMoveProb)
			return False
		def subVacateMonomer(ra):
			rm = ra.parent
			rm.boundMonomer -= ra.copies
			lat.raiseMolecule(rm)
			ra.type = ATOM_SITE
			lat.layMolecule(rm)
		def subGetMonomer(ra):
			rm = ra.parent
			rm.boundMonomer += ra.copies
			lat.raiseMolecule(rm)
			ra.type = ATOM_MONOMER
			lat.layMolecule(rm)
		def doAttachDetach(self):
			lm = lat.atomList[ATOM_MONOMER]
			ra = lm[randint(0,len(lm))]
			if ra.parent.type == MOL_MONOMER:
				ls = filter(lambda a:a.type == ATOM_SITE,lat.getAtomListAt(ra.x, ra.y))
				if len(ls) == 0:
					return False
				ra2 = ls[0]
				subGetMonomer(ls[0])
				lat.removeMolecule(ra.parent)
				return True
			else:
				p = rand()
				if p < exp(ra.copies*lat.bindingEnergy):
					subVacateMonomer(ra)
					lat.addMolecule(Molecule(MOL_MONOMER, ra.x, ra.y, moleculeTemp1))
					return True
			return False
		def doMonomerBirth(self):
			newX = randint(floor(lat.dimX*0.0), floor(lat.dimX*1.0))
			newY = randint(floor(lat.dimY*0.0), floor(lat.dimY*1.0))
			lat.addMolecule(Molecule(MOL_MONOMER, newX, newY, moleculeTemp1))
			return True
		def doMonomerBirth2(self):
			newX = randint(floor(lat.dimX*0.4), floor(lat.dimX*0.6))
			newY = randint(floor(lat.dimY*0.4), floor(lat.dimY*0.6))
			lat.addMolecule(Molecule(MOL_MONOMER, newX, newY, moleculeTemp1))
			return True
		def doMonomerDeath(self):
			al = lat.atomList[ATOM_MONOMER]
			ra = al[randint(0,len(al))]
			rm = ra.parent
			if rm.type == MOL_MONOMER:
				lat.removeMolecule(rm)
				return True
			else:
				p = rand()
				if p < exp(ra.copies*lat.bindingEnergy):
					subVacateMonomer(ra)
					return True
			return False
		def subRemoveDimer(rm):
			for atom in rm.atomList:
				if atom.type == ATOM_MONOMER:
					lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, moleculeTemp1))
			lat.removeMolecule(rm)
		def doDimerDeath(self):
			a1,a2 = lat.atomList[ATOM_DIMER1],lat.atomList[ATOM_DIMER2]
			n1,n2 = len(a1),len(a2)
			m = randint(0,n1+n2)
			if m < n1:
				ra = a1[m]
			else:
				ra = a2[m-n1]
			subRemoveDimer(ra.parent)
			return True
		def subMakeDimer(type,rd,ra,ra2):
			# I-shaped dimer
			if type == 0:
				lat.addMolecule(Molecule(MOL_DIMER1, ra.x, ra.y, newDimerTemp[rd]))
			# L-shaped dimer
			else:
				lat.addMolecule(Molecule(MOL_DIMER2, ra2.x, ra.y, newDimerTemp[rd]))
			lat.newDimerNum[ra.parent.type, ra2.parent.type] += 1
			# Remove the involved monomers
			for atom in [ra, ra2]:
				rm = atom.parent
				if rm.type == MOL_MONOMER:
					lat.removeMolecule(rm)
				else:
					subVacateMonomer(atom)
		def doDimerization(self,type=0):
			al = lat.atomList[ATOM_MONOMER]
			ra = al[randint(0,len(al))]
			rd = randint(0, 4) + type*4
			x2 = ra.x + reactDirMap[rd][0]
			y2 = ra.y + reactDirMap[rd][1]
			
			if 0 < lat.getDataXY(lat.atomNumMap[ATOM_MONOMER], x2, y2):
				for ra2 in lat.getAtomListAt(x2, y2):
					if ra2.type == ATOM_MONOMER:
						subMakeDimer(type,rd,ra,ra2)
						return True
			return False
		doDimerizationI = lambda s: doDimerization(s,0)
		doDimerizationL = lambda s: doDimerization(s,1)
		def doCatDimerization(self,type=0):
			al = lat.atomList[ATOM_MONOMER]
			ra = al[randint(0,len(al))]
			if ra.parent.type != MOL_MONOMER:
				rd = randint(0, 4) + type*4
				x2 = ra.x + reactDirMap[rd][0]
				y2 = ra.y + reactDirMap[rd][1]
				if 0 < lat.getDataXY(lat.atomNumMap[ATOM_MONOMER], x2, y2):
					for ra2 in lat.getAtomListAt(x2, y2):
						if ra2.type == ATOM_MONOMER and ra2.parent == ra.parent:
							subMakeDimer(type,rd,ra,ra2)
							return True
			return False
		doCatDimerizationI = lambda s: doCatDimerization(s,0)
		doCatDimerizationL = lambda s: doCatDimerization(s,1)
		def doAbsorbingBC(self):
			i = randint(0, lat.moleculeNum)
			rm = lat.moleculeList[i]
			x = rm.x%lat.dimX
			y = rm.y%lat.dimY
			if 0 == x*y:
				if rm.type == MOL_MONOMER:
					lat.removeMolecule(rm)
				else:
					subRemoveDimer(rm)
				return True
			return False
		def doDraw(self):
			img.set_array(colorMap(lat))
			draw()
			return False
		def doRecord(self):
			newDimerNum = lat.newDimerNum
			file.write('# New Dimer Num Per Interval\n')
			for mt1 in newDimerNum:
				file.write('# ')
				for mt2 in mt1:
					file.write('{} '.format(mt2))
				file.write('\n')
			file.write('# \n')
			newDimerNum *= 0
			for atomType in lat.atomNumMap:
				for row in atomType:
					for col in row:
						file.write('{} '.format(col))
					file.write('\n')
				file.write('\n')
			file.write('\n')
			return False
		
		mvList = [ \
			Movement(doTranslation,expRandTime,lambda:lat.moleculeNum), \
			Movement(doRotation,expRandTime,lambda:lat.moleculeNum), \
			Movement(doAbsorbingBC,expRandTime,lambda:5. * lat.moleculeNum), \
			Movement(doAttachDetach,expRandTime,lambda:5. * len(lat.atomList[ATOM_MONOMER])), \
			#Movement(doDimerizationI,expRandTime,lambda:self.rate1 * len(lat.atomList[ATOM_MONOMER])), \
			#Movement(doDimerizationL,expRandTime,lambda:self.rate2 * len(lat.atomList[ATOM_MONOMER])), \
			Movement(doCatDimerizationI,expRandTime,lambda:self.cat * self.rate1 * len(lat.atomList[ATOM_MONOMER])), \
			Movement(doCatDimerizationL,expRandTime,lambda:self.cat * self.rate2 * len(lat.atomList[ATOM_MONOMER])), \
			Movement(doDimerDeath,expRandTime,lambda:self.dimerVanish * (len(lat.atomList[ATOM_DIMER1])+len(lat.atomList[ATOM_DIMER2]))/2), \
			Movement(doMonomerBirth2,expRandTime,lambda:self.monomerAppear), \
			Movement(doMonomerDeath,expRandTime,lambda:self.monomerVanish * len(lat.atomList[ATOM_MONOMER]))  ]
		if display:
			mvList.append(Movement(doDraw,lambda:1,lambda:1./outputPeriod))
		if fileFlag:
			mv = Movement(doRecord,lambda:1,lambda:1./outputPeriod)
			#mv.newDimerNum = newDimerNum
			mvList.append(mv)

		t = 0
		for m in mvList:
			m.reset()
		mvList.sort(key=attrgetter('wait'))
		wait = mvList[0].wait
		t += wait
		while t < simTime:
			for m in mvList:
				m.wait -= wait
			isToUpdate = mvList[0]()
			if isToUpdate:
				for m in mvList:
					m.updateParam()
			mvList[0].reset()
			mvList.sort(key=attrgetter('wait'))
			wait = mvList[0].wait
			t += wait
			
		if fileFlag:
			file.close()

		return

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
	parser.add_argument('-d', '--disp', dest='display', action='store_true')
	parser.add_argument('-s', '--size', dest='size', action='store', type=int, default=32)
	parser.add_argument('-o', '--out', dest="outputFileName", action='store')
	parser.add_argument('-e', '--energy', dest="energyScale", action='store', type=float, default=1.)
	parser.add_argument('-t', '--time', dest="simTime", action='store', type=int, default=1000)
	parser.add_argument('-u', '--period', dest="outputPeriod", action='store', type=int, default=1)
	parser.add_argument('-f', '--rate1', dest="rate1", action='store', type=float, default=1.)
	parser.add_argument('-g', '--rate2', dest="rate2", action='store', type=float, default=-1.)
	parser.add_argument('-v', '--ma', dest="ma", action='store', type=float, default=0.)
	parser.add_argument('-i', '--mv', dest="mv", action='store', type=float, default=0.)
	parser.add_argument('-j', '--dv', dest="dv", action='store', type=float, default=0.)
	parser.add_argument('-k', '--db', dest="db", action='store', type=float, default=0.)
	parser.add_argument('-l', '--nm', dest='nm', action='store', type=int, default=0)
	parser.add_argument('-m', '--nd1', dest='nd1', action='store', type=int, default=0)
	parser.add_argument('-n', '--nd2', dest='nd2', action='store', type=int, default=0)
	parser.add_argument('-c', '--cat', dest='cat', action='store', type=float, default=1.)
	
	opt = parser.parse_args()
	if opt.rate2 == -1.:
		opt.rate2 = opt.rate1
			
	if opt.display == True:
		from matplotlib.pyplot import *
		import matplotlib.animation as animation
		
	experiment = Experiment(\
		dimX=opt.size, dimY=opt.size, energyScale=opt.energyScale, rate1=opt.rate1, rate2=opt.rate2, cat=opt.cat, \
		monomerAppear=opt.ma, monomerVanish=opt.mv, dimerVanish=opt.dv, dimerBreak=opt.db, \
		monomerInitNum=opt.nm, dimer1InitNum=opt.nd1, dimer2InitNum=opt.nd2)
		
	experiment.run(simulationTime=opt.simTime, outputPeriod=opt.outputPeriod, display=opt.display, outputFileName=opt.outputFileName)
	
	