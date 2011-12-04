from numpy import *
from numpy.random import rand, randint
from pylab import *

class Atom:
	def __init__(self, type, x, y, parent=None):
		self.type, self.x, self.y, self.parent = type, x, y, parent
		
	def __repr__(self):
		return 'Atom(type=%r, x=%r, y=%r)' % (self.type, self.x, self.y)
		
	@classmethod
	def fromTuple(cls, t): return cls(t[0],t[1],t[2])
	
	@classmethod
	def fromTupleListToList(cls, list): 
		return [cls.fromTuple(atomT) for atomT in list]

	def rotateCW(self):
		self.x, self.y = self.y, -self.x
		
	def rotateCCW(self):
		self.x, self.y = -self.y, self.x
		

class Molecule:
	def __init__(self, type, x, y, atomList):
		self.type, self.x, self.y = type, x, y
		self.atomList = []
		for atom in atomList:
			self.atomList.append(Atom(atom.type, atom.x, atom.y, self))
		
	def __repr__(self):
		return 'Molecule(x=%r, y=%r, atomList=%r)' % (self.x, self.y, self.atomList)
	
	def rotateCW(self): 
		for atom in self.atomList: atom.rotateCW()
		
	def rotateCCW(self): 
		for atom in self.atomList: atom.rotateCCW()
	
	def translate(self,dx,dy):
		self.x += dx
		self.y += dy

class PDLattice:
	def __init__(self, dimX, dimY):
		self.dimX, self.dimY = dimX, dimY
		self.atomListMap = tuple([ tuple([ [] for y in range(dimY) ]) for x in range(dimX) ])
		self.atomNumListMap = tuple([ tuple([ [] for y in range(dimY) ]) for x in range(dimX) ])
		self.moleculeList = [];
		
	def getAtomListAt(self, x, y):
		return self.atomListMap[x%self.dimX][y%self.dimY]

	def getAtomNumListAt(self, x, y):
		return self.atomNumListMap[x%self.dimX][y%self.dimY]
		
	def addMolecule(self, molecule):
		self.moleculeList.append(molecule)
		self.layMolecule(molecule)
		
	def removeMolecule(self, molecule):
		self.raiseMolecule(molecule)
		self.moleculeList.remove(molecule)
	
	def raiseMolecule(self, molecule):
		for atom in molecule.atomList:
			self.getAtomListAt(molecule.x+atom.x, molecule.y+atom.y).remove(atom)
			self.getAtomNumListAt(molecule.x+atom.x, molecule.y+atom.y).remove(atom.type)
			
	def layMolecule(self, molecule):
		for atom in molecule.atomList:
			self.getAtomListAt(molecule.x+atom.x, molecule.y+atom.y).append(atom)
			self.getAtomNumListAt(molecule.x+atom.x, molecule.y+atom.y).append(atom.type)
			
	def translateMolecule(self, molecule, dx, dy, p = 0):
		self.raiseMolecule(molecule)
		e1 = self.getEnergy(molecule)
		molecule.translate(dx, dy)
		e2 = self.getEnergy(molecule)		
		if p > min(1,exp(e1-e2)): molecule.translate(-dx, -dy)
		self.layMolecule(molecule)
				
	def translateMoleculeByIndex(self, i, dx, dy, p = 0):
		self.translateMolecule(self.moleculeList[i], dx, dy, p)
				
	def rotateMoleculeCW(self, molecule, p = 0):
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
			
	def setEnergyMatrix(self, atomTypeNum, energyMatrix):
		self.atomTypeNum = atomTypeNum;
		self.energyMatrix = energyMatrix;
		
	def getEnergy(self, molecule):
		e = 0;
		for atom in molecule.atomList:
			ls = self.getAtomNumListAt(molecule.x+atom.x, molecule.y+atom.y)
			for i in range(self.atomTypeNum):
				e += ls.count(i) * self.energyMatrix[atom.type][i]
		return e
		
	def setColorScale(self, colorScale):
		self.colorScale = array(colorScale)
		
	def getColorAt(self, x, y):
		c = zeros(3)
		for atomType in self.getAtomNumListAt(x, y):
			c += self.colorScale[atomType]
		for i in range(3):
			c[i] = min(1,c[i])
		return c
		
	def toColorMap(self):
		return [ [ self.getColorAt(x, y) \
				for y in range(self.dimY) ] \
			for x in range(self.dimX) ]
		
			
lat = PDLattice(40, 40)
monomerNum = 100
dimer1Num = 0
dimer2Num = 0
moleculeNum = monomerNum + dimer1Num + dimer2Num
simTime = moleculeNum * 5000
mNum = moleculeNum

lat.setColorScale( [[0.20, 0.20, 0.20],\
					[0.30, 0.00, 0.00],\
					[0.00, 0.30, 0.00],\
				    [0.00, 0.00, 0.00]] )


ATOM_MONOMER, ATOM_DIMER1, ATOM_DIMER2, ATOM_SITE = 0, 1, 2, 3
atomTemp1 = Atom.fromTupleListToList( [(ATOM_MONOMER,0,0)] )
atomTemp2 = Atom.fromTupleListToList( [(ATOM_DIMER1,0,0), (ATOM_DIMER1,0,1), \
									   (ATOM_SITE,1,0),   (ATOM_SITE,1,1), \
									   (ATOM_SITE,-1,0),  (ATOM_SITE,-1,1)] )
atomTemp3 = Atom.fromTupleListToList( [(ATOM_DIMER2,0,-1), (ATOM_DIMER2,-1,0), \
									   (ATOM_SITE,1,-1),   (ATOM_SITE,-1,1), \
									   (ATOM_SITE,0,0),    (ATOM_SITE,0,0) ] )
lat.setEnergyMatrix(4, 1.5 * array([[ 2,  2,  2, -1],\
									[ 2,  2,  2,  0],\
									[ 2,  2,  2,  0],\
									[-1,  0,  0,  0]]))

MOL_MONOMER, MOL_DIMER = 0, 1

initX = randint(0, lat.dimX, moleculeNum)
initY = randint(0, lat.dimY, moleculeNum)

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

degProb = rand(simTime)

newX = zeros(simTime, int)
newY = zeros(simTime, int)
newProb = rand(simTime)

fig = imshow(lat.toAtomNumMap(), interpolation='none')
ioff()

newDimerTemp = range(8)
newDimerTemp[0] = Atom.fromTupleListToList( [(ATOM_DIMER1,0,0), (ATOM_DIMER1,1,0), \
											 (ATOM_SITE,0,1),   (ATOM_SITE,1,1), \
											 (ATOM_SITE,0,-1),  (ATOM_SITE,1,-1)] )
newDimerTemp[1] = Atom.fromTupleListToList( [(ATOM_DIMER1,0,0), (ATOM_DIMER1,-1,0), \
											 (ATOM_SITE,0,1),   (ATOM_SITE,-1,1), \
											 (ATOM_SITE,0,-1),  (ATOM_SITE,-1,-1)] )
newDimerTemp[2] = Atom.fromTupleListToList( [(ATOM_DIMER1,0,0), (ATOM_DIMER1,0,1), \
											 (ATOM_SITE,1,0),   (ATOM_SITE,1,1), \
											 (ATOM_SITE,-1,0),  (ATOM_SITE,-1,1)] )
newDimerTemp[3] = Atom.fromTupleListToList( [(ATOM_DIMER1,0,0), (ATOM_DIMER1,0,-1), \
											 (ATOM_SITE,1,0),   (ATOM_SITE,1,-1), \
											 (ATOM_SITE,-1,0),  (ATOM_SITE,-1,-1)] )
newDimerTemp[4] = Atom.fromTupleListToList( [(ATOM_DIMER2,0,1), (ATOM_DIMER2,-1,0), \
											 (ATOM_SITE,1,1),   (ATOM_SITE,-1,-1), \
											 (ATOM_SITE,0,0),   (ATOM_SITE,0,0) ] )
newDimerTemp[5] = Atom.fromTupleListToList( [(ATOM_DIMER2,0,1), (ATOM_DIMER2,1,0), \
											 (ATOM_SITE,-1,1),  (ATOM_SITE,1,-1), \
											 (ATOM_SITE,0,0),    (ATOM_SITE,0,0) ] )
newDimerTemp[6] = Atom.fromTupleListToList( [(ATOM_DIMER2,0,-1), (ATOM_DIMER2,1,0), \
											 (ATOM_SITE,-1,-1),  (ATOM_SITE,1,1), \
											 (ATOM_SITE,0,0),    (ATOM_SITE,0,0) ] )
newDimerTemp[7] = Atom.fromTupleListToList( [(ATOM_DIMER2,0,-1), (ATOM_DIMER2,-1,0), \
											 (ATOM_SITE,1,-1),   (ATOM_SITE,-1,1), \
											 (ATOM_SITE,0,0),    (ATOM_SITE,0,0) ] )
reactDirMap = [[1,0],[-1,0],[0,1],  [0,-1],\
			   [1,1],[-1,1],[-1,-1],[1,-1]]

for t in range(simTime):
	tMoveMol[t] = randint(0, moleculeNum)
	tMoveDir[t] = randint(0, 2)
	tMoveDis[t] = randint(0, 2) * 2 - 1
	tMoveProb[t] = rand()

	dx = tMoveDis[t] * tMoveDir[t]
	dy = tMoveDis[t] * (1 - tMoveDir[t])
	lat.translateMoleculeByIndex(tMoveMol[t], dx, dy, tMoveProb[t])
	
	rMoveDir[t] = randint(0, 2)
	rMoveMol[t] = randint(0, moleculeNum)
	rMoveProb[t] = rand()
	
	if rMoveDir[t] == 0:
		lat.rotateMoleculeCWByIndex(rMoveMol[t], rMoveProb[t])
	else:
		lat.rotateMoleculeCCWByIndex(rMoveMol[t], rMoveProb[t])
	
	reactMol[t] = randint(0, moleculeNum)
	rm = lat.moleculeList[reactMol[t]]
	
	if rm.type == MOL_MONOMER:
		if degProb[t] < 0.00:
			lat.removeMolecule(rm)
			moleculeNum -= 1
		else:
			rd = reactDir[t] = randint(0, 8)
			
			x2 = rm.x + reactDirMap[rd][0]
			y2 = rm.y + reactDirMap[rd][1]
		
			anl = lat.getAtomNumListAt(x2, y2)
			if 0 < anl.count(ATOM_MONOMER):
				for atom in lat.getAtomListAt(x2, y2):
					if atom.type == ATOM_MONOMER:
						rm2 = atom.parent
						if rd < 4:
							lat.addMolecule(Molecule(MOL_DIMER, rm.x, rm.y, newDimerTemp[rd]))
						else:
							lat.addMolecule(Molecule(MOL_DIMER, rm2.x, rm.y, newDimerTemp[rd]))
						lat.removeMolecule(rm)
						lat.removeMolecule(rm2)
						moleculeNum -= 1
						break
	elif (rm.type == MOL_DIMER):
		if degProb[t] < 0.005:
			lat.removeMolecule(rm)
			moleculeNum -= 1
		elif degProb[t] > 1:
			for atom in rm.atomList:
				if atom.type == ATOM_DIMER1 or atom.type == ATOM_DIMER2:
					lat.addMolecule(Molecule(MOL_MONOMER, rm.x+atom.x, rm.y+atom.y, atomTemp1))
			lat.removeMolecule(rm)
			moleculeNum += 1

		
	if newProb[t] < 1.50/moleculeNum:
		newX[t] = randint(floor(lat.dimX*0.0), floor(lat.dimX*1.0))
		newY[t] = randint(floor(lat.dimY*0.0), floor(lat.dimY*1.0))
		lat.addMolecule(Molecule(MOL_MONOMER, newX[t], newY[t], atomTemp1))
		moleculeNum += 1
	
	if (t+1)%(500) == 0:
		fig.set_array(lat.toColorMap())
		draw()
	