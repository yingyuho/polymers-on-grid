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