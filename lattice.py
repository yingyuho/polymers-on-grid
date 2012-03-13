from numpy.core.multiarray import zeros
from math import exp

class Atom:
	def __init__(self, type, x, y, copies=1):
		self.type, self.x, self.y, self.copies = type, x, y, copies
		
	def __repr__(self):
		return 'Atom(type=%r, x=%r, y=%r)' % (self.type, self.x, self.y)
		
	def setParent(self, molecule):
		self.parent = molecule
		return self
		
	@classmethod
	def fromTuple(cls, t): 
		if len(t) == 4:
			return cls(t[0],t[1],t[2],t[3])
		else:
			return cls(t[0],t[1],t[2])
	
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
			self.atomList.append(Atom(atom.type, atom.x, atom.y, atom.copies).setParent(self))
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

class Lattice:
	def __init__(self, dimX, dimY):
		self.dimX, self.dimY = dimX, dimY
		self.atomListMap = tuple([ tuple([ [] for y in range(dimY) ]) for x in range(dimX) ])
		self.atomNumListMap = tuple([ tuple([ [] for y in range(dimY) ]) for x in range(dimX) ])
		self.moleculeList = [];
		self.atomList = [];
		self.moleculeNum = 0;
		
	def getDataXY(self, mat, x, y):
		return mat[x%self.dimX][y%self.dimY]

	def setDataXY(self, mat, x, y, value):
		mat[x%self.dimX][y%self.dimY] = value
		
	def incDataXY(self, mat, x, y, inc):
		mat[x%self.dimX][y%self.dimY] += inc
		
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
	
	def raiseMolecule(self, molecule, perm = True):
		"""Temporarily remove a molecule from the lattice to facilate operations on that molecule"""
		if perm:
			for atom in molecule.atomList:
				self.getAtomListAt(atom.x, atom.y).remove(atom)
				self.incDataXY(self.atomNumMap[atom.type], atom.x, atom.y, -1)
				self.atomList[atom.type].remove(atom)
		else:
			for atom in molecule.atomList:
				self.getAtomListAt(atom.x, atom.y).remove(atom)
				self.incDataXY(self.atomNumMap[atom.type], atom.x, atom.y, -1)
			
	def layMolecule(self, molecule, perm = True):
		"""Put back a raised molecule to the lattice after operations"""
		if perm:
			for atom in molecule.atomList:
				self.getAtomListAt(atom.x, atom.y).append(atom)
				self.incDataXY(self.atomNumMap[atom.type], atom.x, atom.y, 1)
				self.atomList[atom.type].append(atom)
		else:
			for atom in molecule.atomList:
				self.getAtomListAt(atom.x, atom.y).append(atom)
				self.incDataXY(self.atomNumMap[atom.type], atom.x, atom.y, 1)
			
	def translateMolecule(self, molecule, dx, dy, p = 0):
	# Put doc
		if p > 1: return
		self.raiseMolecule(molecule, perm = False)
		e1 = self.getEnergy(molecule)
		molecule.translate(dx, dy)
		e2 = self.getEnergy(molecule)		
		if p > min(1,exp(e1-e2)): molecule.translate(-dx, -dy)
		self.layMolecule(molecule, perm = False)
				
	def translateMoleculeByIndex(self, i, dx, dy, p = 0):
		self.translateMolecule(self.moleculeList[i], dx, dy, p)
				
	def rotateMoleculeCW(self, molecule, p = 0):
	# Put doc
		if p > 1: return
		self.raiseMolecule(molecule, perm = False)
		e1 = self.getEnergy(molecule)
		molecule.rotateCW()
		e2 = self.getEnergy(molecule)
		if p > min(1,exp(e1-e2)): molecule.rotateCCW()
		self.layMolecule(molecule, perm = False)

	def rotateMoleculeCWByIndex(self, i, p = 0):
		self.rotateMoleculeCW(self.moleculeList[i], p)
		
	def rotateMoleculeCCW(self, molecule, p = 0):
		if p > 1: return
		self.raiseMolecule(molecule, perm = False)
		e1 = self.getEnergy(molecule)
		molecule.rotateCCW()
		e2 = self.getEnergy(molecule)
		if p > min(1,exp(e1-e2)): molecule.rotateCW()
		self.layMolecule(molecule, perm = False)

	def rotateMoleculeCCWByIndex(self, i, p = 0):
		self.rotateMoleculeCCW(self.moleculeList[i], p)
			
	def setAtomTypeNum(self, atomTypeNum):
		self.atomTypeNum = atomTypeNum;
		self.atomList = [[] for x in range(atomTypeNum)]
		self.atomNumMap = zeros((self.atomTypeNum, self.dimX, self.dimY), int)
		
	def setEnergyMatrix(self, energyMatrix):
		self.energyMatrix = energyMatrix;
		
	def getEnergy(self, molecule):
		e = 0;
		for atom in molecule.atomList:
			#ls = self.getAtomNumAt(atom.x, atom.y)
			for i in range(self.atomTypeNum):
				e += self.getDataXY(self.atomNumMap[i], atom.x, atom.y) * self.energyMatrix[atom.type][i]
		return e