#!/usr/bin/env python3

from sympy import Symbol, linsolve
import numpy as np
import matplotlib.pyplot as plt


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

theta = 0
pos = {}        # {A, B, C, D, E, F, G, H}
vel = {}        # {A, B, C, D, E, F, G, H}
ang_vel = {}    # {W_2, W_3, W_4, W_5, W_6, W_7, W_8}
accl = {}       # {A, B, C, D, E, F, G, H}
ang_accl = {}   # {Alph_2, Alph_3, Alph_4, Alph_5, Alph_6, Alph_7, Alph_8}

'''
Account for precision errors - arccos accepts values in range [-1, 1]
'''
def sanitize_arccos_input(angle):
    
    if angle > 1:
        return 1
    
    if angle < -1:
        return -1

    return angle

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
    phi_1 = np.arctan(R[1]/R[0]) # [-pi/2, pi/2]
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
def solve_pos(R_xo, R_yo, r_zx, r_yz):

    R_yx = R_yo - R_xo
    r_yx = np.linalg.norm(R_yx)

    phi = angle_with_i(R_yx)

    # find deviation angle, account for precision errors
    arccos_input = sanitize_arccos_input((r_yx**2 + r_zx**2 - r_yz**2)/(2*r_zx*r_yx))
    dev_zx = np.arccos(arccos_input)

    # angle between R_zx and i
    theta_zx_1 = phi + dev_zx
    theta_zx_2 = phi - dev_zx

    # construct vector Z
    R_zx_1 = np.array([r_zx*np.cos(theta_zx_1), r_zx*np.sin(theta_zx_1), 0], dtype=np.float64)
    R_zo_1 = R_xo + R_zx_1

    R_zx_2 = np.array([r_zx*np.cos(theta_zx_2), r_zx*np.sin(theta_zx_2), 0], dtype=np.float64)
    R_zo_2 = R_xo + R_zx_2

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
   
    #print(sol_list)
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
Choose the next position from several solutions that
corresponds to having the shortest  distance from the previous position
of the point 
@param pos_solutions - tuple of numpy arrays corresponding to solution
@param previous_pos - numpy array of previous position vector

@return chosen solution
'''
def choose_closest(pos_solutions, previous_pos):

    chosen_sol = None
    closest_dist = None

    for sol in pos_solutions:

        if(chosen_sol is None):
            chosen_sol = sol
            closest_dist = np.linalg.norm(sol - previous_pos)
            continue
        
        curr_sol_dist = np.linalg.norm(sol - previous_pos)
        if (curr_sol_dist < closest_dist):
            chosen_sol = sol
            closest_dist = curr_sol_dist
    
    return chosen_sol



'''
Determines all position vectors given input angle theta. 
To resolve ambiguities, position tuple of previous iteration is used
such that position of X chosen is closest to the position of X for theta - d(theta)

Updates pos dictionary with new positions
'''
def position_analysis(theta):
    
    # fixed points
    R_ao = np.array([0., 0., 0.], dtype=np.float64)
    R_eo = np.array([-38, -7.8, 0.], dtype=np.float64)

    # determined points
    R_bo = np.array([r_ab*np.cos(theta), r_ab*np.sin(theta), 0.], dtype=np.float64)
    R_co = choose_closest(solve_pos(R_eo, R_bo, r_ce, r_bc), pos['C'])
    R_fo = choose_closest(solve_pos(R_eo, R_co, r_ef, r_cf), pos['F'])
    R_do = choose_closest(solve_pos(R_eo, R_bo, r_de, r_bd), pos['D'])
    R_go = choose_closest(solve_pos(R_do, R_fo, r_dg, r_fg), pos['G'])
    R_ho = choose_closest(solve_pos(R_go, R_do, r_gh, r_dh), pos['H'])

    # update pos dict
    pos['A'] = R_ao
    pos['B'] = R_bo
    pos['C'] = R_co
    pos['D'] = R_do
    pos['E'] = R_eo
    pos['F'] = R_fo
    pos['G'] = R_go
    pos['H'] = R_ho

    return

'''
Determine all velocity vectors given input angular velocity W_2. 
To be invoked after position_analysis to use updated position vectors

update vel and ang_vel dictionaries with new values
'''
def velocity_analysis(W_2):

    # get current positions
    R_ao = pos['A']
    R_bo = pos['B']
    R_co = pos['C']
    R_do = pos['D']
    R_eo = pos['E']
    R_fo = pos['F']
    R_go = pos['G']
    R_ho = pos['H']
    
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
    #print("V_f7: ", V_f7, "R_gf: ", R_gf, "V_d8: ", V_d8, "R_gd: ", R_gd)
    #print("W_7: ", W_7, "\t\t W_8", W_8)
    V_g8 = V_g7 = V_f7 + np.cross(W_7, R_gf)                # G

    R_hd = R_ho - R_do
    V_h8 = V_d8 + np.cross(W_8, R_hd)                       # H


    # update velocities
    vel['A'] = V_a2
    vel['B'] = V_b4
    vel['C'] = V_c5
    vel['D'] = V_d8
    vel['E'] = V_e6
    vel['F'] = V_f7
    vel['G'] = V_g8
    vel['H'] = V_h8

    # update angular velocities
    ang_vel['W_2'] = W_2
    ang_vel['W_3'] = W_3
    ang_vel['W_4'] = W_4
    ang_vel['W_5'] = W_5
    ang_vel['W_6'] = W_6
    ang_vel['W_7'] = W_7
    ang_vel['W_8'] = W_8

    return


'''
Determine all velocity vectors given input angular velocity W_2. 
To be invoked after position_analysis to use updated position vectors

updates accl and ang_accl dicts with new values
'''
def acceleration_analysis(Alph_2):

    # get current positions
    R_ao = pos['A']
    R_bo = pos['B']
    R_co = pos['C']
    R_do = pos['D']
    R_eo = pos['E']
    R_fo = pos['F']
    R_go = pos['G']
    R_ho = pos['H']

    # get current angular velocities of members
    W_2 = ang_vel['W_2']
    W_3 = ang_vel['W_3']
    W_4 = ang_vel['W_4']
    W_5 = ang_vel['W_5']
    W_6 = ang_vel['W_6']
    W_7 = ang_vel['W_7']
    W_8 = ang_vel['W_8']

    # fixed points
    A_a2 = np.array([0., 0., 0.], dtype=np.float64)                         # A
    A_e6 = A_e5 = np.array([0., 0., 0.], dtype=np.float64)                  # E

    # determined points
    R_ba = R_bo
    A_b4 = A_b3 = A_b2 = A_a2 + np.cross(Alph_2, R_ba) - (W_2[2]**2)*R_ba   # B

    R_cb = R_co - R_bo
    R_ce = R_co - R_eo
    Alph_3, Alph_5 = solve_Alphs(A_b3, R_cb, W_3, A_e5, R_ce, W_5)
    A_c3 = A_c5 = A_e5 + np.cross(Alph_5, R_ce) - (W_5[2]**2)*R_ce          # C

    R_fc = R_fo - R_co
    A_f7 = A_f5 = A_c5 + np.cross(Alph_5, R_fc) - (W_5[2]**2)*R_fc          # F

    R_db = R_do - R_bo
    R_de = R_do - R_eo
    Alph_4, Alph_6 = solve_Alphs(A_b4, R_db, W_4, A_e6, R_de, W_6)
    A_d4 = A_d8 = A_d6 = A_e6 + np.cross(Alph_6, R_de) - (W_6[2]**2)*R_de   # D

    R_gf = R_go - R_fo
    R_gd = R_go - R_do
    Alph_7, Alph_8 = solve_Alphs(A_f7, R_gf, W_7, A_d8, R_gd, W_8)
    A_g7 = A_g8 = A_d8 + np.cross(Alph_8, R_gd) - (W_8[2]**2)*R_gd          # G

    R_hd = R_ho - R_do
    A_h8 = A_d8 + np.cross(Alph_8, R_hd) - (W_8[2]**2)*R_hd                 # H

    # update accelerations
    accl['A'] = A_a2
    accl['B'] = A_b4
    accl['C'] = A_c3
    accl['D'] = A_d4
    accl['E'] = A_e6
    accl['F'] = A_f7
    accl['G'] = A_g7
    accl['H'] = A_h8

    # update angular accelerations
    ang_accl['Alph_2'] = Alph_2
    ang_accl['Alph_3'] = Alph_3
    ang_accl['Alph_4'] = Alph_4
    ang_accl['Alph_5'] = Alph_5
    ang_accl['Alph_6'] = Alph_6
    ang_accl['Alph_7'] = Alph_7
    ang_accl['Alph_8'] = Alph_8

    return


'''
INITIAL STATE (at theta = 0)
'''
# solutions to positions determined individually by inspection
R_ao = np.array([0., 0., 0.], dtype=np.float64)
R_eo = np.array([-38, -7.8, 0.], dtype=np.float64)
R_bo = np.array([r_ab, 0., 0.], dtype=np.float64)
R_co = solve_pos(R_eo, R_bo, r_ce, r_bc)[0]
R_fo = solve_pos(R_eo, R_co, r_ef, r_cf)[0]
R_do = solve_pos(R_eo, R_bo, r_de, r_bd)[1]
R_go = solve_pos(R_do, R_fo, r_dg, r_fg)[0]
R_ho = solve_pos(R_go, R_do, r_gh, r_dh)[1]

# update initial positions
pos['A'] = R_ao
pos['B'] = R_bo
pos['C'] = R_co
pos['D'] = R_do
pos['E'] = R_eo
pos['F'] = R_fo
pos['G'] = R_go
pos['H'] = R_ho

W_2 = np.array([0., 0., 0.5], dtype=np.float64)   # chosen
Alph_2 = np.array([0., 0., 0.], dtype=np.float64) # chosen

# conduct analysis and update corresponding dicts
velocity_analysis(W_2)
acceleration_analysis(Alph_2)


'''
UPDATE STATE
Use current state to calculate and update next state
at d(theta) = 0.02 rad
'''

# all values deal in magnitudes
max_disp = {'A': None, 'B': None, 'C': None, 'D': None, 'E': None, 'F': None, 'G': None, 'H': None}
max_speed = {'A': None, 'B': None, 'C': None, 'D': None, 'E': None, 'F': None, 'G': None, 'H': None}
max_accl = {'A': None, 'B': None, 'C': None, 'D': None, 'E': None, 'F': None, 'G': None, 'H': None}

def update_max_disp():

    for point, position in pos.items():
        curr_disp = np.linalg.norm(position)
        max_d = max_disp[point]

        if(max_d is None or curr_disp > max_d):
            max_disp[point] = curr_disp

    return


def update_max_speed():
    for point, velocity in vel.items():
        curr_speed = np.linalg.norm(velocity)
        max_v = max_speed[point]

        if(max_v is None or curr_speed > max_v):
            max_speed[point] = curr_speed
            
    return


def update_max_accl():
    for point, acceleration in accl.items():
        curr_accel = np.linalg.norm(acceleration)
        max_a = max_accl[point]

        if(max_a is None or curr_accel > max_a):
            max_accl[point] = curr_accel
            
    return

d_theta = 0.02  # increment iteration by 0.02 rad
theta = 0.02    # next input after initial

x_vals = {}
y_vals = {}

# create list to display position coords
for pnt in pos.keys():
    x_vals[pnt] = []
    y_vals[pnt] = []


# look at one full cycle of mechanism
while(theta < 2*np.pi):

    # analyze previous state
    update_max_disp()
    update_max_speed()
    update_max_accl()

    # add current point to list
    for pnt, coord in pos.items():
        x_vals[pnt].append(coord[0])
        y_vals[pnt].append(coord[1])


    # calculate current state
    position_analysis(theta)
    velocity_analysis(ang_vel['W_2'])
    acceleration_analysis(ang_accl['Alph_2'])

    # advance to next state
    theta += d_theta


'''
Display Results
'''
print('----- Max Displacements -----')
for pnt, disp in max_disp.items():
    print(pnt, disp)
print()

print('----- Max Speeds -----')
for pnt, speed in max_speed.items():
    print(pnt, speed)
print()

print('----- Max Acceleration Magnitudes -----')
for pnt, accel in max_accl.items():
    print(pnt, accel)
print()

# plot history of points
points_to_display = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}

for pnt in points_to_display:
    plt.plot(x_vals[pnt], y_vals[pnt], label=pnt)

plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()