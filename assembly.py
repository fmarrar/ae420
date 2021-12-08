import numpy as np

def element_stiffness(nl, NL, E, nu, GPE=4):
	"""NL: Node list
	GPE: Number of Gauss Points per element

	output the element stiffness matrix"""

	NpE = np.size(nl, 0)
	DOF = np.size(NL, 1)

	x = np.zeros([NpE, DOF])
	x[0:NpE, 0:DOF] = NL[nl[0:NpE]-1, 0:DOF]

	# Initialize element stiffness matrix
	K = np.zeros([NpE * DOF, NpE * DOF])

	coor = x.T

	for i in range(1, NpE + 1):
		for j in range(1, NpE + 1):
			k = np.zeros([DOF, DOF])
			for gp in range(1, GPE+1):
				J = np.zeros([DOF, DOF])
				grad = np.zeros([DOF, NpE])
				(xi, eta, w) = GaussPoint(NpE, GPE, gp)

				grad_nat = grad_N_nat(NpE, xi, eta)

				J = coor @ grad_nat.T

				grad = np.linalg.inv(J).T @ grad_nat

				for a in range(1, DOF+1):
					for c in range(1, DOF+1):
						for b in range(1, DOF+1):
							for d in range(1, DOF+1):
								k[a-1, c-1] = k[a-1, c-1] + grad[b-1, i-1] * constitutive(a,b,c,d, E, nu) * grad[d-1, j-1] * np.linalg.det(J) * w
				K[((i-1)*DOF+1)-1:i*DOF, ((j-1)*DOF+1)-1:j*DOF] = k
	return K

def GaussPoint(NpE, GPE, gp):
	if NpE == 4:
		if GPE == 1:
			if gp == 1:
				xi = 0
				eta = 0
				w = 4
		if GPE == 4:
			if gp == 1:
				xi = -1/np.sqrt(3)
				eta = -1/np.sqrt(3)
				w = 1
			if gp == 2:
				xi = 1/np.sqrt(3)
				eta = -1/np.sqrt(3)
				w = 1

			if gp == 3:
				xi = 1/np.sqrt(3)
				eta = 1/np.sqrt(3)
				w = 1

			if gp == 4:
				xi = -1/np.sqrt(3)
				eta = 1/np.sqrt(3)
				w = 1
	return (xi, eta, w)

def grad_N_nat(NpE, xi, eta):
	DOF = 2
	result = np.zeros([DOF, NpE])

	if NpE == 3:
		result[0,0] = 1
		result[0,1] = 0
		result[0,2] = -1

		result[1,0] = 0
		result[1,1] = 1
		result[1,2] = -1

	if NpE == 4:
		result[0,0] = -1/4*(1 - eta)
		result[0,1] = 1/4*(1 - eta)
		result[0,2] = 1/4*(1 + eta)
		result[0,3] = -1/4*(1 + eta)

		result[1,0] = -1/4*(1 - xi)
		result[1,1] = -1/4*(1+xi)
		result[1,2] = 1/4*(1+xi)
		result[1,3] = 1/4*(1-xi)

	return result

def constitutive(i,j,k,l, E, nu):
	C = (E/(2*(1+nu))) * (kronecker(i,l)*kronecker(j, k) + kronecker(i, k)*kronecker(j,l)) + (E*nu)/(1-nu**2) * kronecker(i, j) * kronecker(k,l)
	return C

def kronecker(i,j):
	if i == j:
		return 1
	else:
		return 0

def assemble_stiffness(NL, EL, E, nu):
	NoE = np.size(EL, 0)
	NpE = np.size(EL, 1)

	NoN = np.size(NL, 0)
	DOF = np.size(NL, 1)

	K = np.zeros([NoN * DOF, NoN * DOF])

	for i in range(1, NoE+1):
		nl = EL[i-1, 0:NpE]
		k = element_stiffness(nl, NL, E, nu)
		idx = np.concatenate((2*nl - 2, 2*nl-1)).tolist()
		[X, Y] = np.meshgrid(idx, idx)
		K[X, Y] = K[X, Y] + k

	return K

def reduce_K(K, nodes_BC):
	nodes_BC = np.array(nodes_BC)

	delete_list_x = 2*nodes_BC - 2
	delete_list_y = 2*nodes_BC - 1

	delete_list = np.concatenate((delete_list_x, delete_list_y))

	K = np.delete(K, delete_list, 0)
	K = np.delete(K, delete_list, 1)

	K_reduced = K
	return K_reduced

def assemble_displacements(u_reduced, nodes_BC, NL, EL):
	NoN = np.size(NL, 0)
	NoE = np.size(EL, 0)

	NpE = np.size(EL, 1)
	DOF = np.size(NL, 1)

	for n in nodes_BC:
		x_n = 2*n - 2
		y_n = 2*n - 1
		u_reduced = np.insert(u_reduced, x_n, 0, axis=0)
		u_reduced = np.insert(u_reduced, y_n, 0, axis=0)
	u = u_reduced
	return u

def assign_BC(EL, p, m):
	nodes_BC = []
	for i in range(1, m+1):
		current_element = EL[p*(i-1),:]
		first_node = current_element[0]
		nodes_BC.append(first_node)
		if i == m:
			nodes_BC.append(first_node + (p+1))
	return nodes_BC

def assign_forces(nodes_BC, p, m, flag=1):
	NoN = (p+1)*(m+1)
	nodes_forces = np.array([nodes_BC[-1]+i for i in range(0, p+1)])
	location = np.zeros(flag)

	if flag == 1:
		location[flag - 1] = 2*int(np.floor(np.median(nodes_forces))) - 1

	elif flag == 2:
		location[flag - 2] = 2*int(np.floor(np.median(nodes_forces))) - 1
		location[flag - 1] = 2*nodes_forces[-2] - 1

	elif flag == 3:
		location[flag - 3] = 2*int(np.floor(np.median(nodes_forces))) - 1
		location[flag - 2] = 2*nodes_forces[-2] - 1
		location[flag - 1] = 2*nodes_forces[1] - 1

	location_distribution = 2*nodes_forces[1:] - 1

	return (location.astype(int), location_distribution)

def reduce_R(R, nodes_BC):
	nodes_BC = np.array(nodes_BC)

	delete_list_x = 2*nodes_BC - 2
	delete_list_y = 2*nodes_BC - 1

	delete_list = np.concatenate((delete_list_x, delete_list_y))

	R = np.delete(R, delete_list, 0)

	R_reduced = R
	return R_reduced