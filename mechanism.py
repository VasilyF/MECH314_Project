#!/usr/bin/env python3

from sympy import Symbol, nsolve, sin, cos, atan
import numpy as np

init_guess = np.pi/4    # for solving non-lin equations of angles forming triangle

'''
solve for angle phi between R and global x axis (i)
note: phi will have two solutions: phi, phi + pi, 
return phi: [-pi, pi]
'''
def angle_with_i(R):
    # check that vector is not vertical -> asymptote of atan
    if(R[0] == 0):
        phi  = np.pi/2 if (R[1] > 0) else -np.pi/2
        return phi
    
    # phi_1 and phi_2 are solutions [-pi, pi], phi_1 < phi_2
    phi_1 = atan(R[1]/R[0]) # [-pi/2, pi/2]
    phi_2 = phi_1 + np.pi
    if(phi_1 > 0):
        phi_1 -= np.pi
        phi_2 -= np.pi

    # correct choice of angle by considering quadrant of R
    phi = phi_2 if (R[1] > 0) else phi_1

    return phi
    

'''
Solve for the vector position of Z in triangle XYZ
return tuple - two possible solutions
'''
def solve_pos(R_xo, R_yo, r_zx, r_zy):

    R_yx = R_yo - R_xo
    r_yx = np.linalg.norm(R_yx)

    alpha = Symbol('alpha')
    beta = Symbol('beta')

    f1 = r_zx * sin(alpha) - r_zy * sin(beta)
    f2 = r_zx * cos(alpha) + r_zy * cos(beta) - r_yx

    # solve for interior angles
    sol = nsolve((f1, f2), (alpha, beta), (init_guess, init_guess))
    alpha_sol = sol[0, 0]

    # get angle of R_yx with x-axis
    phi = angle_with_i(R_yx)

    # construct vector Z
    R_zx_1 = np.array([r_zx*cos(phi + alpha_sol), r_zx*sin(phi + alpha_sol), 0], dtype=np.float64)
    R_zo_1 = R_xo + R_zx_1

    R_zx_2 = np.array([r_zx*cos(phi - alpha_sol), r_zx*sin(phi - alpha_sol), 0], dtype=np.float64)
    R_zo_2 = R_xo + R_zx_2

    print("solution 1: ", R_zo_1)
    print("solution 2: ", R_zo_2)
    return (R_zo_1, R_zo_2)
    


'''
Vectors represented as [x, y, z], point a is origin
R -> position vectors, r  = ||R|| (distance scalar)
'''

# distances between points
r_ab = 15.0
r_bc = 50.0
r_bd = 61.9
r_ce = 41.5
r_cf = 55.8
r_de = 39.3
r_dg = 36.7
r_dh = 49.0
r_ef = 40.1
r_fg = 39.4
r_gh = 65.7

'''
INITIAL STATE at theta = 0
'''
# position analysis
# solutions to positions determined individually by inspection
theta = 0.0
R_ao = np.array([0., 0., 0.], dtype=np.float64)
R_eo = np.array([-38, -7.8, 0.], dtype=np.float64)
R_bo = np.array([r_ab, 0., 0.], dtype=np.float64)
R_co = solve_pos(R_eo, R_bo, r_ce, r_bc)[0]
R_fo = solve_pos(R_eo, R_co, r_ef, r_cf)[0]
R_do = solve_pos(R_eo, R_bo, r_de, r_bd)[1]
R_go = solve_pos(R_do, R_fo, r_dg, r_fg)[0]
R_ho = solve_pos(R_go, R_do, r_gh, r_dh)[1]

print("-------------")
print("A: ", R_ao) 
print("B: ", R_bo)
print("C: ", R_co)
print("D: ", R_do)
print("E: ", R_eo)
print("F: ", R_fo)
print("G: ", R_go)
print("H: ", R_ho)

# velocity analysis
# TODO
# fixed points
# R_ao = np.array([0., 0., 0.])
# R_eo = np.array([-38, -7.8, 0.])
# R_ba = np.array([r_ab * np.cos(theta), r_ab * np.sin(theta), 0])

'''
UPDATED STATE
'''