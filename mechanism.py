#!/usr/bin/env python3

from sympy import Symbol, sin, cos, atan, nsolve, linsolve
import numpy as np

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

    init_guess = np.pi/4    # for solving non-lin equations of angles forming triangle

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

    # print("solution 1: ", R_zo_1)
    # print("solution 2: ", R_zo_2)

    return (R_zo_1, R_zo_2)


'''
Solve system of equations of the form: 
V_z2 = V_z3
V_z2 = V_x2 + (W_2 x R_zx)
V_z3 = V_y3 + (W_3 x R_zy)

two same-member equations, W_2 and W_3 unknown
'''
def solve_Ws(V_x2, R_zx, V_y3, R_zy):

    _w_2 = Symbol('w_2')
    _w_3 = Symbol('w_3')

    # [0, 0, w] x [R_x, R_y, 0] = [-R_y*w, R_x*w, 0]
    # solving for x-component
    f1 = V_x2[0] - R_zx[1]*_w_2
    f2 = V_y3[0] - R_zy[1]*_w_3

    # solving for y-component
    f3 = V_x2[1] + R_zx[0]*_w_2
    f4 = V_y3[1] + R_zy[0]*_w_3

    sol_list = list(linsolve([f1 - f2, f3 - f4], (_w_2, _w_3))) # list of solution tuples
   
    # expect only single solution
    _W_2 = np.array([0., 0., sol_list[0][0]], dtype=np.float64)   
    _W_3 = np.array([0., 0., sol_list[0][1]], dtype=np.float64)
    return (_W_2, _W_3) 

'''
Solve system of equations of the form: 
A_z2 = A_z3
A_z2 = A_x2 + (Alph_2 x R_zx) - (w_2^2)R_zx
A_z3 = A_y3 + (Alph_3 x R_zy) - (w_3^2)R_zy

two same-member equations, Alph_2 and Alph_3 unknown
'''
def solve_Alphs(A_x2, R_zx, Omeg_2, A_y3, R_zy, Omeg_3):

    A_z2_known_comp = A_x2 - (Omeg_2[2]**2)*R_zx
    A_z3_known_comp = A_y3 - (Omeg_3[2]**2)*R_zy

    # system of equations of similar form to velocity equations
    # A_z2 = A_z3
    # A_z2 = [A_x2 - (w_2^2)R_zx] + (Alph_2 x R_zx) 
    # A_z3 = [A_y3 - (w_3^2)R_zy] + (Alph_3 x R_zy) 
    _Alph_2, _Alph_3 = solve_Ws(A_z2_known_comp, R_zx, A_z3_known_comp, R_zy)

    return _Alph_2, _Alph_3

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
# POSITION ANALYSIS
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

print("------ Position Analysis -------")
print("A: ", R_ao) 
print("B: ", R_bo)
print("C: ", R_co)
print("D: ", R_do)
print("E: ", R_eo)
print("F: ", R_fo)
print("G: ", R_go)
print("H: ", R_ho)
print()

# VELOCITY ANALYSIS
W_2 = np.array([0., 0., 0.5], dtype=np.float64)   # chosen

# fixed points
V_a2 = np.array([0., 0., 0.], dtype=np.float64)         # A
V_e6 = V_e5 = np.array([0., 0., 0.], dtype=np.float64)  # E

# determined points
R_ba = R_bo
V_b4 = V_b3 = V_b2 = V_a2 + np.cross(W_2, R_ba)         # B

R_cb = R_co - R_bo
R_ce = R_co - R_eo
W_3, W_5 = solve_Ws(V_b3, R_cb, V_e5, R_ce)
V_c5 = V_c3 = V_b3 + np.cross(W_3, R_cb)                # C

R_fc = R_fo - R_co
V_f7 = V_f5 = V_c5 + + np.cross(W_5, R_fc)              # F

R_db = R_do - R_bo
R_de = R_do - R_eo
W_4, W_6 = solve_Ws(V_b4, R_db, V_e6, R_de)
V_d8 = V_d6 = V_d4 = V_e6 + np.cross(W_6, R_de)         # D

R_gf = R_go - R_fo
R_gd = R_go - R_do
W_7, W_8 = solve_Ws(V_f7, R_gf, V_d8, R_gd)
V_g8 = V_g7 = V_f7 + np.cross(W_7, R_gf)                # G

R_hd = R_ho - R_do
V_h8 = V_d8 + np.cross(W_8, R_hd)                       # H

print("------ Velocity Analysis -------")
print("A: ", V_a2) 
print("B: ", V_b4)
print("C: ", V_c5)
print("D: ", V_d8)
print("E: ", V_e6)
print("F: ", V_f7)
print("G: ", V_g8)
print("H: ", V_h8)
print()

# ACCELERATION ANALYSIS
Alph_2 = np.array([0., 0., 0.], dtype=np.float64) # chosen

# fixed points
A_a2 = np.array([0., 0., 0.], dtype=np.float64)         # A
A_e6 = A_e5 = np.array([0., 0., 0.], dtype=np.float64)  # E

# determined points
A_b4 = A_b3 = A_b2 = A_a2 + np.cross(Alph_2, R_ba) - (W_2[2]**2)*R_ba   # B

Alph_3, Alph_5 = solve_Alphs(A_b3, R_cb, W_3, A_e5, R_ce, W_5)
A_c3 = A_c5 = A_e5 + np.cross(Alph_5, R_ce) - (W_5[2]**2)*R_ce          # C

A_f7 = A_f5 = A_c5 + np.cross(Alph_5, R_fc) - (W_5[2]**2)*R_fc          # F

Alph_4, Alph_6 = solve_Alphs(A_b4, R_db, W_4, A_e6, R_de, W_6)
A_d4 = A_d8 = A_d6 = A_e6 + np.cross(Alph_6, R_de) - (W_6[2]**2)*R_de   # D

Alph_7, Alph_8 = solve_Alphs(A_f7, R_gf, W_7, A_d8, R_gd, W_8)
A_g7 = A_g8 = A_d8 + np.cross(Alph_8, R_gd) - (W_8[2]**2)*R_gd          # G

A_h8 = A_d8 + np.cross(Alph_8, R_hd) - (W_8[2]**2)*R_hd                 # H

print("------ Accelereation Analysis -------")
print("A: ", A_a2) 
print("B: ", A_b4)
print("C: ", A_c3)
print("D: ", A_d4)
print("E: ", A_e6)
print("F: ", A_f7)
print("G: ", A_g7)
print("H: ", A_h8)
print()

'''
UPDATED STATE
'''