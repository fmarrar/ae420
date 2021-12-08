import numpy as np

def generate_mesh(w, h, p, m, element_type):
	"""Inputs
	w = width of the structure
	h = height of the structure
	p = number of portions in the x
	m = number of portions in the y
	element_type = a string. 'quadrilateral', 'triangular'

	output the mesh generated
	"""

	element_type = element_type.lower()
	DOF = 2

	s = np.array([[0,0], [w, 0], [0, h], [w, h]]) # This defines the entire structure span

	# Imagine a mesh that is p x m = 2 x 2, then the number of nodes will be p + 1 x m + 1
	NoN = (p+1)*(m+1)
	# By the same logic, the number of elements will be 2 x 2, the same shape as the number of portions
	NoE = p*m

	# Number of nodes per element depends on the element type. Assume 4 node quadrilateral
	# and 3 node triangular, so check the first letter of element_type

	NpE = 4

	# Node List will be number of nodes in rows, and degrees of freedom as columns
	NL = np.zeros([NoN, DOF])

	# Increment in x will be the entire width of the structure, divided by the number of portions
	ix = w/p
	iy = h/m # Same logic for y

	# Initialize the variable, n, which is the current node in the loop
	n = 0
	for i in range(1, m+2):
		for j in range(1, p+2):
			NL[n, 0] = s[0,0] + (j - 1)*ix # This will label the x coord of nodes throughout the mesh
			NL[n, 1] = s[0,1] + (i - 1)*iy # This will label the y coord of nodes throughout the mesh

			n += 1

	# Elements list 
	EL = np.zeros([NoE, NpE])

	for i in range(1, m+1):
		for j in range(1, p+1):
			if j == 1:
				EL[(i-1)*p+j-1, 0] = (i-1)*(p+1) + j
				EL[(i-1)*p+j-1, 1] = EL[(i-1)*p+j-1, 0] + 1
				EL[(i-1)*p+j-1, 3] = EL[(i-1)*p+j-1, 0] + (p+1)
				EL[(i-1)*p+j-1, 2] = EL[(i-1)*p+j-1, 3] + 1
			else:
				EL[(i-1)*p+j-1, 0] = EL[(i-1)*p+j-2, 1]
				EL[(i-1)*p+j-1, 3] = EL[(i-1)*p+j-2, 2]
				EL[(i-1)*p+j-1, 1] = EL[(i-1)*p+j-1, 0] + 1
				EL[(i-1)*p+j-1, 2] = EL[(i-1)*p+j-1, 3] + 1
	if element_type == 'triangular':
		NpE_t = 3
		NoE_t = 2*NoE
		EL_t = np.zeros([NoE_t, NpE_t])

		for i in range(1, NoE+1):
			EL_t[2*(i - 1), 0] = EL[i - 1, 0]
			EL_t[2*(i - 1), 1] = EL[i - 1, 1]
			EL_t[2*(i - 1), 2] = EL[i - 1, 2]

			EL_t[2*(i - 1)+1, 0] = EL[i - 1, 0]
			EL_t[2*(i - 1)+1, 1] = EL[i - 1, 2]
			EL_t[2*(i - 1)+1, 2] = EL[i - 1, 3]
		
		EL = EL_t

	EL = EL.astype(int)

	return (NL, EL)