#########################################################################
# polymerDiffusion.py
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
from numpy.random import rand, randint
#from pylab import *

class Atom:
	def __init__(self, type, x, y, copies=1):
		self.type, self.x, self.y, self.copies = type, x, y, copies
		
	def __repr__(self):
		return 'Atom(type=%r, x=%r, y=%r)' % (self.type, self.x, self.y)
		
	def setParent(self, molecule):
		self.parent = molecule
		return self
		
	@classmethod
	def fromTuple(cls, t): return cls(t[0],t[1],t[2])
	
	def rotateCW(self, x0, y0):
		self.x, self.y = x0-y0+self.y, y0+x0-self.x
		
	def rotateCCW(self, x0, y0):
		self.x, self.y = x0+y0-self.y, y0-x0+self.x
		
	def translate(self, dx, dy):
		self.x += dx
		self.y += dy
		

class Molecule:

	def __init__(self, type, x, y, atomList):
		self.type = type;
		self.x = self.y = 0
		self.atomList = []
		for atom in atomList:
			self.atomList.append(Atom(atom.type, atom.x, atom.y).setParent(self))
		self.translate(x, y)
		
	def __repr__(self):
		return 'Molecule(x=%r, y=%r, atomList=%r)' % (self.x, self.y, self.atomList)
	
	def rotateCW(self): 
		"""Rotate each component atom clockwise"""
		for atom in self.atomList: atom.rotateCW(self.x, self.y)
		
	def rotateCCW(self): 
		"""Rotate each component atom counterclockwise"""
		for atom in self.atomList: atom.rotateCCW(self.x, self.y)
	
	def translate(self, dx, dy):
		"""Translate each component atom by (dx, dy)"""
		self.x += dx
		self.y += dy
		for atom in self.atomList: atom.translate(dx, dy)
		

class PDLattice:
	def __init__(self, dimX, dimY):
		self.dimX, self.dimY = dimX, dimY
		self.atomListMap = tuple([ tuple([ [] for y in range(dimY) ]) for x in range(dimX) ])
		self.atomNumListMap = tuple([ tuple([ [] for y in range(dimY) ]) for x in range(dimX) ])
		self.moleculeList = [];
		self.atomList = [];
		self.moleculeNum = 0;
		
	def getDataXY(self, var, x, y):
		return var[x%self.dimX][y%self.dimY]
		
	def getAtomListAt(self, x, y):
		"""Return the list of atoms at (x,y)"""
		return self.atomListMap[x%self.dimX][y%self.dimY]

#	def getAtomNumListAt(self, x, y):
#		"""Return the list of atom types at (x,y)"""
#		return self.atomNumListMap[x%self.dimX][y%self.dimY]
		
	def getAtomNumAt(self, x, y):
		return self.atomNumMap[x%self.dimX][y%self.dimY]
		
	def addMolecule(self, molecule):
		self.moleculeList.append(molecule)
		self.layMolecule(molecule)
		self.moleculeNum += 1;
		
	def removeMolecule(self, molecule):
		self.raiseMolecule(molecule)
		self.moleculeList.remove(molecule)
		self.moleculeNum -= 1;
	
	def raiseMolecule(self, molecule):
		"""Temporarily remove a molecule from the lattice to facilate operations on that molecule"""
		for atom in molecule.atomList:
			self.getAtomListAt(atom.x, atom.y).remove(atom)
			#self.getAtomNumListAt(atom.x, atom.y).remove(atom.type)
			self.getAtomNumAt(atom.x, atom.y)[atom.type] -= 1
			self.atomList[atom.type].remove(atom)
			
	def layMolecule(self, molecule):
		"""Put back a raised molecule to the lattice after operations"""
		for atom in molecule.atomList:
			self.getAtomListAt(atom.x, atom.y).append(atom)
			#self.getAtomNumListAt(atom.x, atom.y).append(atom.type)
			self.getAtomNumAt(atom.x, atom.y)[atom.type] += 1
			self.atomList[atom.type].append(atom)
			
	def translateMolecule(self, molecule, dx, dy, p = 0):
	# Put doc
		self.raiseMolecule(molecule)
		e1 = self.getEnergy(molecule)
		molecule.translate(dx, dy)
		e2 = self.getEnergy(molecule)		
		if p > min(1,exp(e1-e2)): molecule.translate(-dx, -dy)
		self.layMolecule(molecule)
				
	def translateMoleculeByIndex(self, i, dx, dy, p = 0):
		self.translateMolecule(self.moleculeList[i], dx, dy, p)
				
	def rotateMoleculeCW(self, molecule, p = 0):
	# Put doc
		self.raiseMolecule(molecule)
		e1 = self.getEnergy(molecule)
		molecule.rotateCW()
		e2 = self.getEnergy(molecule)
		if p > min(1,exp(e1-e2)): molecule.rotateCCW()
		self.layMolecule(molecule)

	def rotateMoleculeCWByIndex(self, i, p = 0):
		self.rotateMoleculeCW(self.moleculeList[i], p)
		
	def rotateMoleculeCCW(self, molecule, p = 0):
		self.raiseMolecule(molecule)
		e1 = self.getEnergy(molecule)
		molecule.rotateCCW()
		e2 = self.getEnergy(molecule)
		if p > min(1,exp(e1-e2)): molecule.rotateCW()
		self.layMolecule(molecule)

	def rotateMoleculeCCWByIndex(self, i, p = 0):
		self.rotateMoleculeCCW(self.moleculeList[i], p)

	def toAtomNumMap(self):
		return [ [ len(self.atomListMap[x][y]) \
				for y in range(self.dimY) ] \
			for x in range(self.dimX) ]
			
	def setAtomTypeNum(self, atomTypeNum):
		self.atomTypeNum = atomTypeNum;
		self.atomList = [[] for x in range(atomTypeNum)]
		#self.atomNumMap = zeros((self.atomTypeNum, self.dimX, self.dimY))
		self.atomNumMap = tuple([ tuple([ [0 for a in range(self.atomTypeNum)] \
									for y in range(self.dimY) ]) \
							for x in range(self.dimX) ])
		
	def setEnergyMatrix(self, energyMatrix):
		self.energyMatrix = energyMatrix;
		
	def getEnergy(self, molecule):
		e = 0;
		for atom in molecule.atomList:
			ls = self.getAtomNumAt(atom.x, atom.y)
			for i in range(self.atomTypeNum):
				e += ls[i] * self.energyMatrix[atom.type][i]
		return e
		
	def setColorScale(self, colorScale):
		self.colorScale = array(colorScale)
		
	def getColorAt(self, x, y):
		c = zeros(3)
		for atomType in range(self.atomTypeNum):
			c += self.getAtomNumAt(x, y)[atomType] * self.colorScale[atomType]
		for i in range(3):
			c[i] = min(1,c[i])
		return c
		
	def toColorMap(self):
		return [ [ self.getColorAt(x, y) \
				for y in range(self.dimY) ] \
			for x in range(self.dimX) ]
			
	def getAtomNumberMap(self):
		amap = zeros((self.atomTypeNum,self.dimX,self.dimY))
		for x in range(self.dimX):
			for y in range(self.dimY):
				for atomType in range(self.atomTypeNum):
					amap[atomType][x][y] = self.getAtomNumAt(x, y)[atomType]
		return amap
		
			
lat = PDLattice(50, 50)
monomerNum = 200
dimer1Num = 0
dimer2Num = 0
mNum = monomerNum + dimer1Num + dimer2Num
simTime = mNum * 2500
moleculeNum = mNum

from optparse import OptionParser
output = True
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename", default='temp.txt')
parser.add_option("-e", type="float", nargs=1, dest="energyScale", default=3.)
(options, args) = parser.parse_args()
#parser.add_argument('--out', nargs='1', type=argparse.FileType('w'), default=sys.stdout)
#parser.add_argument('--ergscale', nargs='1', type=float, default=1.)

file = open(options.filename, 'w')

lat.setColorScale( [[0.20, 0.20, 0.20],\
					[0.30, 0.00, 0.00],\
					[0.00, 0.30, 0.00],\
				    [0.00, 0.00, 0.00]] )


ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = 0, 1, 2, 3
# Monomer Temp
atomTemp1 = map(Atom.fromTuple, [(ATOM_MONOMER,0,0)])
# I Dimer
atomTemp2 = map(Atom.fromTuple, [(ATOM_DIMER1,0,0), (ATOM_DIMER1,0,1), \
								 (ATOM_SITE,1,0),   (ATOM_SITE,1,1), \
								 (ATOM_SITE,-1,0),  (ATOM_SITE,-1,1)])
# L Dimer
atomTemp3 = map(Atom.fromTuple, [(ATOM_DIMER2,0,-1), (ATOM_DIMER2,-1,0), \
								 (ATOM_SITE,1,-1),   (ATOM_SITE,-1,1), \
								 (ATOM_SITE,0,0,2)])
								 
lat.setAtomTypeNum(4)
lat.setEnergyMatrix(array([[ 1,  1,  1,  0],\
						   [ 1,  1,  1,  0],\
						   [ 1,  1,  1,  0],\
						   [ 0,  0,  0,  0]]) * options.energyScale)
									
lat.bindingEnergy = -1.0 * options.energyScale;

print(lat.bindingEnergy)

MOL_MONOMER, MOL_DIMER = 0, 1

initX = randint(0, lat.dimX, mNum)
initY = randint(0, lat.dimY, mNum)

for n in range(monomerNum):
	lat.addMolecule(Molecule(MOL_MONOMER, initX[n], initY[n], atomTemp1))
for n in range(dimer1Num):
	lat.addMolecule(Molecule(MOL_DIMER,   initX[n], initY[n], atomTemp2))
for n in range(dimer2Num):
	lat.addMolecule(Molecule(MOL_DIMER,   initX[n], initY[n], atomTemp3))

#print(lat.getEnergy(lat.moleculeList[0]))

tMoveMol = zeros(simTime, int)
tMoveDir = zeros(simTime, int)
tMoveDis = zeros(simTime, int)
tMoveProb = zeros(simTime)

rMoveMol = zeros(simTime, int)
rMoveDir = zeros(simTime, int)
rMoveProb = zeros(simTime)

reactMol = zeros(simTime, int)
reactDir = zeros(simTime, int)
reactAtom = zeros(simTime, int)

degProb = rand(simTime)

newX = zeros(simTime, int)
newY = zeros(simTime, int)
newProb = rand(simTime)

bindProb = rand(simTime)

if output:
	fig = imshow(lat.toAtomNumMap(), interpolation='none')
	ioff()

newDimerTemp = range(8)
newDimerTemp[0] = map(Atom.fromTuple, [(ATOM_DIMER1,0,0), (ATOM_DIMER1,1,0), \
									   (ATOM_SITE,0,1),   (ATOM_SITE,1,1), \
									   (ATOM_SITE,0,-1),  (ATOM_SITE,1,-1)] )
newDimerTemp[1] = map(Atom.fromTuple, [(ATOM_DIMER1,0,0), (ATOM_DIMER1,-1,0), \
									   (ATOM_SITE,0,1),   (ATOM_SITE,-1,1), \
									   (ATOM_SITE,0,-1),  (ATOM_SITE,-1,-1)] )
newDimerTemp[2] = map(Atom.fromTuple, [(ATOM_DIMER1,0,0), (ATOM_DIMER1,0,1), \
									   (ATOM_SITE,1,0),   (ATOM_SITE,1,1), \
									   (ATOM_SITE,-1,0),  (ATOM_SITE,-1,1)] )
newDimerTemp[3] = map(Atom.fromTuple, [(ATOM_DIMER1,0,0), (ATOM_DIMER1,0,-1), \
									   (ATOM_SITE,1,0),   (ATOM_SITE,1,-1), \
									   (ATOM_SITE,-1,0),  (ATOM_SITE,-1,-1)] )
newDimerTemp[4] = map(Atom.fromTuple, [(ATOM_DIMER2,0,1), (ATOM_DIMER2,-1,0), \
									   (ATOM_SITE,1,1),   (ATOM_SITE,-1,-1), \
									   (ATOM_SITE,0,0,2) ])#),   (ATOM_SITE,0,0) ] )
newDimerTemp[5] = map(Atom.fromTuple, [(ATOM_DIMER2,0,1), (ATOM_DIMER2,1,0), \
									   (ATOM_SITE,-1,1),  (ATOM_SITE,1,-1), \
									   (ATOM_SITE,0,0,2) ])#),    (ATOM_SITE,0,0) ] )
newDimerTemp[6] = map(Atom.fromTuple, [(ATOM_DIMER2,0,-1), (ATOM_DIMER2,1,0), \
									   (ATOM_SITE,-1,-1),  (ATOM_SITE,1,1), \
									   (ATOM_SITE,0,0,2) ])#),    (ATOM_SITE,0,0) ] )
newDimerTemp[7] = map(Atom.fromTuple, [(ATOM_DIMER2,0,-1), (ATOM_DIMER2,-1,0), \
									   (ATOM_SITE,1,-1),   (ATOM_SITE,-1,1), \
									   (ATOM_SITE,0,0,2) ])#,    (ATOM_SITE,0,0) ] )
reactDirMap = [[1,0],[-1,0],[0,1],  [0,-1],\
			   [1,1],[-1,1],[-1,-1],[1,-1]]
			   
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
	
		anl = lat.getAtomNumAt(x2, y2)
		
		if 0 < anl[ATOM_MONOMER]:
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
		if degProb[t] < 0*0.01:
			lat.removeMolecule(rm)
#		else:
#			rd = reactDir[t] = randint(0, 8)
#			
#			x2 = rm.x + reactDirMap[rd][0]
#			y2 = rm.y + reactDirMap[rd][1]
#		
#			anl = lat.getAtomNumListAt(x2, y2)
#			if 0 < anl.count(ATOM_MONOMER):
#				for atom in lat.getAtomListAt(x2, y2):
#					if atom.type == ATOM_MONOMER:
#						rm2 = atom.parent
#						if rd < 4:
#							lat.addMolecule(Molecule(MOL_DIMER, rm.x, rm.y, newDimerTemp[rd]))
#						else:
#							lat.addMolecule(Molecule(MOL_DIMER, rm2.x, rm.y, newDimerTemp[rd]))
#						lat.removeMolecule(rm)
#						lat.removeMolecule(rm2)
#						moleculeNum -= 1
#						break
						
	elif (rm.type == MOL_DIMER):
		# Death of dimer
		if degProb[t] < 0.002:
			for atom in rm.atomList:
				if atom.type == ATOM_MONOMER:
					lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, atomTemp1))
			lat.removeMolecule(rm)
			moleculeNum -= 1
		# Degradation of dimer into monomers
		elif degProb[t] > 0.99:
			for atom in rm.atomList:
				if atom.type != ATOM_SITE:
					lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, atomTemp1))
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
						lat.addMolecule(Molecule(MOL_MONOMER, atom.x, atom.y, atomTemp1))
		
	# Add monomers randomly to the system
	if newProb[t] < 0.05/lat.moleculeNum:
		newX[t] = randint(floor(lat.dimX*0.0), floor(lat.dimX*1.0))
		newY[t] = randint(floor(lat.dimY*0.0), floor(lat.dimY*1.0))
		lat.addMolecule(Molecule(MOL_MONOMER, newX[t], newY[t], atomTemp1))
	
	# Update the animation per ## moves
	if (t+1)%(200) == 0:
		if output:
			fig.set_array(lat.toColorMap())
			draw()
#			print(len(lat.atomList[ATOM_MONOMER])),
#			print(len(lat.atomList[ATOM_DIMER1])),
#			print(len(lat.atomList[ATOM_DIMER2]))
#			print(repr(lat.getAtomNumberMap()))
		file.write('{} {} {}\n'.format(len(lat.atomList[ATOM_MONOMER]), len(lat.atomList[ATOM_DIMER1]), len(lat.atomList[ATOM_DIMER2])))

file.close()