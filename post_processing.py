import numpy as np
from assembly import *
def element_post_process(NL, EL, u, E, nu):
	DOF = np.size(NL, 1)
	NoE = np.size(EL, 0)
	NpE = np.size(EL, 1)

	GPE = 4

	disp = np.zeros([NoE, NpE, DOF, 1])

	stress = np.zeros([NoE, GPE, DOF, DOF])
	strain = np.zeros([NoE, GPE, DOF, DOF])

	for e in range(1, NoE + 1):
		nl = EL[e-1,0:NpE]
		x = np.zeros([NpE, DOF])
		x[0:NpE, 0:DOF] = NL[nl[0:NpE]-1, 0:DOF]
		uu = np.zeros([DOF, NpE])
		uu[0,:] = u[2*nl - 2].reshape((NpE,))
		uu[1,:] = u[2*nl - 1].reshape((NpE,))

		coor = x.T

		for gp in range(1, GPE+1):
			epsilon = np.zeros([DOF, DOF])

			for i in range(1, NpE + 1):
				J = np.zeros([DOF, DOF])
				grad = np.zeros([DOF, NpE])

				(xi, eta, w) = GaussPoint(NpE, GPE, gp)

				grad_nat = grad_N_nat(NpE, xi, eta)

				J = coor @ grad_nat.T

				grad = np.linalg.inv(J).T @ grad_nat

				epsilon = epsilon + 1/2 * (dyad(grad[:,i-1],uu[:,i-1]) + dyad(uu[:,i-1],grad[:,i-1]))

			sigma = np.zeros([DOF, DOF])

			for a in range(1, DOF+1):
				for b in range(1, DOF+1):
					for c in range(1, DOF+1):
						for d in range(1, DOF+1):
							sigma[a-1, b-1] = sigma[a-1, b-1] + constitutive(a,b,c,d, E, nu) * epsilon[c-1, d-1]

			for a in range(1, DOF+1):
				for b in range(1, DOF+1):
					strain[e-1, gp-1, a-1, b-1] = epsilon[a-1, b-1]
					stress[e-1, gp-1, a-1, b-1] = sigma[a-1, b-1]
	return stress, strain

def dyad(u, v):
	u = u.reshape(len(v),1)
	v = v.reshape(len(v),1)

	PD = 2

	A = u @ v.T

	return A

def post_process(NL,EL, u, E, nu, scale=1):
	DOF = np.size(NL, 1)
	NoE = np.size(EL, 0)
	NpE = np.size(EL, 1)

	stress, strain = element_post_process(NL, EL, u, E, nu)

	stress_xx = np.zeros([NpE, NoE])
	stress_xy = np.zeros([NpE, NoE])
	stress_yx = np.zeros([NpE, NoE])
	stress_yy = np.zeros([NpE, NoE])

	strain_xx = np.zeros([NpE, NoE])
	strain_xy = np.zeros([NpE, NoE])
	strain_yx = np.zeros([NpE, NoE])
	strain_yy = np.zeros([NpE, NoE])

	disp_x = np.zeros([NpE, NoE])
	disp_y = np.zeros([NpE, NoE])

	X = np.zeros([NpE, NoE])
	Y = np.zeros([NpE, NoE])

	u_X = np.zeros([NpE, NoE])
	u_Y = np.zeros([NpE, NoE])

	X_X = np.zeros([NpE, NoE])
	Y_Y = np.zeros([NpE, NoE])

	for i in range(1, NoE + 1):
		nl = EL[i-1,:]
		u_x = u[2*nl - 2,:].reshape((NpE,))
		u_y = u[2*nl - 1,:].reshape((NpE,))
		u_X[:,i-1] = u_x
		u_Y[:,i-1] = u_y
		X_x = NL[EL[i-1,:]-1, 0]
		Y_y = NL[EL[i-1,:]-1, 1]

		X_X[:,i-1] = X_x
		Y_Y[:,i-1] = Y_y

	if NpE in [3, 4]:
		X = X_X + scale*u_X
		Y = Y_Y + scale*u_Y
		stress_xx[:,:] = stress[:,:,0,0].T
		stress_xy[:,:] = stress[:,:,0,1].T
		stress_yx[:,:] = stress[:,:,1,0].T
		stress_yy[:,:] = stress[:,:,1,1].T

		strain_xx[:,:] = stress[:,:,0,0].T
		strain_xy[:,:] = stress[:,:,0,1].T
		strain_yx[:,:] = stress[:,:,1,0].T
		strain_yy[:,:] = stress[:,:,1,1].T

	return (stress_xx, stress_xy, stress_yx, stress_yy,strain_xx,strain_xy,strain_yx,strain_yy, X, Y)