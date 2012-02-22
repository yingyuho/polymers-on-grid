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
from numpy.fft import *
from numpy.random import rand, randint
		

from lattice import Atom, Molecule, Lattice

			
lat = Lattice(32,32)
monomerNum = 100
dimer1Num = 0
dimer2Num = 0
mNum = monomerNum + dimer1Num + dimer2Num
simTime = mNum * 5000
moleculeNum = mNum


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--disp', dest='display', action='store_true')
	parser.add_argument('-o', '--out', dest="outputFileName", action='store')
	parser.add_argument('-e', '--energy', dest="energyScale", action='store', type=int, default=3.)
	options = parser.parse_args()
	
file = open(options.outputFileName, 'w')

lat.setColorScale( [[0.20, 0.20, 0.20],\
					[0.30, 0.00, 0.00],\
					[0.00, 0.30, 0.00],\
				    [0.05, 0.05, 0.05]] )


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
	fig = imshow(lat.toColorMap(), interpolation='none')
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
	lat.translateMoleculeByIndex(tMoveMol[t], dx, dy, 9*tMoveProb[t])
	
	# Rotate one molecule
	rMoveDir[t] = randint(0, 2)
	rMoveMol[t] = randint(0, lat.moleculeNum)
	rMoveProb[t] = rand()
	
	if rMoveDir[t] == 0:
		lat.rotateMoleculeCWByIndex(rMoveMol[t], 9*rMoveProb[t])
	else:
		lat.rotateMoleculeCCWByIndex(rMoveMol[t], 9*rMoveProb[t])
	
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
		if degProb[t] < 0.15:
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
		# Degradation of dimer into    monomers
		elif degProb[t] > 1.99:
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
	if newProb[t] < 2.5/lat.moleculeNum:
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
		#file.write('{} {} {}\n'.format(len(lat.atomList[ATOM_MONOMER]), len(lat.atomList[ATOM_DIMER1]), len(lat.atomList[ATOM_DIMER2])))

file.close()