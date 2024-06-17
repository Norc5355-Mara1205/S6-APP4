#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:19:16 2020

@author: alex
------------------------------------


Fichier d'amorce pour les livrables de la problématique GRO640'


"""

IMPEDANCE = "impedance"
FORCE = "force"

import numpy as np

from pyro.control  import robotcontrollers
from pyro.control  import nonlinear
from pyro.control.robotcontrollers import EndEffectorPD
from pyro.control.robotcontrollers import EndEffectorKinematicController
from scipy.optimize import fsolve

def equations(vars, l1, l2, l3, x, y, z):
    theta1, theta2, theta3 = vars
    eq1 = l2 * np.sin(theta2) * np.cos(theta1) + l3 * np.sin(theta3) * np.cos(theta2) * np.cos(theta1) + l3 * np.sin(theta2) * np.cos(theta3) * np.cos(theta1) - x
    eq2 = l1 + l2 * np.cos(theta2) + l3 * np.cos(theta2) * np.cos(theta3) - l3 * np.sin(theta3) * np.sin(theta2) - y
    eq3 = l2 * np.sin(theta2) * np.sin(theta1) + l3 * np.sin(theta2) * np.cos(theta3) * np.sin(theta1) + l3 * np.sin(theta3) * np.cos(theta2) * np.sin(theta1) - z
    return [eq1, eq2, eq3]


###################
# Part 1
###################

def dh2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float 1x1
    d     : float 1x1
    theta : float 1x1
    alpha : float 1x1
    
    4 paramètres de DH

    Returns
    -------
    T     : float 4x4 (numpy array)
            Matrice de transformation

    """
    
    T = np.zeros((4,4))
    
    ###################
    # Votre code ici
    ###################
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta) * np.cos(alpha)
    T[0, 2] = np.sin(theta) * np.sin(alpha)
    T[0, 3] = r * np.cos(theta)
    
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta) * np.cos(alpha)
    T[1, 2] = -np.cos(theta) * np.sin(alpha)
    T[1, 3] = r * np.sin(theta)
    
    T[2, 0] = 0
    T[2, 1] = np.sin(alpha)
    T[2, 2] = np.cos(alpha)
    T[2, 3] = d
    
    T[3, 0] = 0
    T[3, 1] = 0
    T[3, 2] = 0
    T[3, 3] = 1
    

    return T



def dhs2T( r , d , theta, alpha ):
    """

    Parameters
    ----------
    r     : float nx1
    d     : float nx1
    theta : float nx1
    alpha : float nx1
    
    Colonnes de paramètre de DH

    Returns
    -------
    WTT     : float 4x4 (numpy array)
              Matrice de transformation totale de l'outil

    """
    WTT = np.identity(4)
    for i in range(len(r)):
        WTT = WTT @ dh2T(r[i], d[i], theta[i], alpha[i])
        
    
    return WTT


def f(q):
    """
    

    Parameters
    ----------
    q : float 5x1
        Joint space coordinates

    Returns
    -------
    r : float 3x1 
        Effector (x,y,z) position
    

    """
    q[1] += np.pi/2
    q[3] += np.pi/2

    t = dhs2T([33, 155, 135, 0, 9.5], [147, 0, 0, 0, 222.16], q, [np.pi/2, 0, 0, np.pi/2, 0])
    r = t[0:3,3]
    
    return r


###################
# Part 2
###################
    
class CustomPositionController( EndEffectorKinematicController ) :
    
    ############################
    def __init__(self, manipulator ):
        EndEffectorKinematicController.__init__( self, manipulator, 1)
        self.gains = np.diag([5, 5])
        self.lamda = 0.5
        
    
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback law: u = c(y,r,t)
        
        INPUTS
        y = q   : sensor signal vector  = joint angular positions      dof x 1
        r = r_d : reference signal vector  = desired effector position   e x 1
        t       : time                                                   1 x 1
        
        OUPUTS
        u = dq  : control inputs vector =  joint velocities             dof x 1
        
        """
        
        # Feedback from sensors
        q = y
        
        # Jacobian computation
        J = self.J( q )
        J_T = J.transpose()
        
        # Ref
        r_desired   = r
        r_actual    = self.fwd_kin( q )
        
        r_r = self.gains @ (r_desired - r_actual)
        dq = np.linalg.inv(J_T @ J + self.lamda**2*np.identity(3)) @ J_T @ r_r
        
        return dq
    
    
###################
# Part 3
###################
        

        
class CustomDrillingController( robotcontrollers.RobotController ) :
    def __init__(self, robot_model, control_type=FORCE ):
        """ """
        
        super().__init__( dof = 3 )
        
        self.control_type = control_type
        self.is_at_target = False
        self.r_d = np.array([0.25, 0.25, 0.4])
        self.kp = np.diag([10,10,10])
        self.kd = np.diag([10,10,10])
        self.robot_model = robot_model
        
        # Label
        self.name = 'Custom Drilling Controller'
        
        
    #############################
    def c( self , y , r , t = 0 ):
        """ 
        Feedback static computation u = c(y,r,t)
        
        INPUTS
        y  : sensor signal vector     p x 1
        r  : reference signal vector  k x 1
        t  : time                     1 x 1
        
        OUPUTS
        u  : control inputs vector    m x 1
        
        """
        
        # Ref
        f_e = r

        
        # Feedback from sensors
        x = y
        [ q , dq ] = self.x2q( x )
        forces = np.array([0,0,-200])
        
        # Robot model
        r = self.robot_model.forward_kinematic_effector( q ) # End-effector actual position
        J = self.robot_model.J( q )      # Jacobian matrix
        J_T = J.transpose()
        g = self.robot_model.g( q )      # Gravity vector
        H = self.robot_model.H( q )      # Inertia matrix
        C = self.robot_model.C( q , dq ) # Coriolis matrix
            
        ##################################
        # Votre loi de commande ici !!!
        ##################################

        u = np.array([0, 0, 0]) # placeholder
        e = self.r_d - r
        u_i = J.T @ ( self.kp @ (e) + self.kd @ ( - J @ dq ) ) + g
        e_lim = 0.03
        
        if not self.is_at_target:
            if all(np.abs(x) < e_lim for x in e):
                self.is_at_target = True
                self.r_d = np.array([0.25, 0.25, 0.2])
                self.kp = np.diag([50,50,50])
                self.kd = np.diag([50,50,50])
                print('passed')
            else:
                u = u_i
        else:
            if self.control_type is FORCE:
                u = J_T @ forces + g
            elif self.control_type is IMPEDANCE:
                u = u_i
        
        return u
        
    
###################
# Part 4
###################
        
    
def goal2r( r_0 , r_f , t_f ):
    """
    
    Parameters
    ----------
    r_0 : numpy array float 3 x 1
        effector initial position
    r_f : numpy array float 3 x 1
        effector final position
    t_f : float
        time 

    Returns
    -------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l

    """
    # Time discretization
    l = 1000 # nb of time steps
    t = np.linspace(0,t_f,l)
    
    # Number of DoF for the effector only
    m = 3
    
    r = np.zeros((m,l))
    dr = np.zeros((m,l))
    ddr = np.zeros((m,l))
    
    #################################
    # Votre code ici !!!
    ##################################


    s = (3/t_f**2) * t**2 - (2/t_f**3) * t**3
    ds = (6/t_f**2) * t - (6/t_f**3) * t**2
    dds = (6/t_f**2) - (12/t_f**3) * t
    
    for i in range(m):
        r[i, :] = r_0[i] + s * (r_f[i] - r_0[i])
        dr[i, :] = ds * (r_f[i] - r_0[i])
        ddr[i, :] = dds * (r_f[i] - r_0[i])
    
    return r, dr, ddr


def r2q( r, dr, ddr , manipulator ):
    """

    Parameters
    ----------
    r   : numpy array float 3 x l
    dr  : numpy array float 3 x l
    ddr : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l

    """
    # Time discretization
    l = r.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    q = np.zeros((n,l))
    dq = np.zeros((n,l))
    ddq = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    l1 = manipulator.l1
    l2 = manipulator.l2
    l3 = manipulator.l3

    r1 = np.sqrt(r[0,:]**2 + (r[1,:])**2)
    r2 = (r[2, :]*l1)
    r3 = np.sqrt(r1**2 + r2**2)

    phi1 = np.arccos((l3**2 - l1**2 - r3**2)/-2*l2*r3)
    phi2 = np.arctan2(r2, r1)
    phi3 = np.arccos((r3**2 - l2**2 - l3**2)/-2*l2*l3)
    
    q[0, :] = np.pi + np.arctan2(r[0,:], r[1,:])
    q[1, :] = np.pi/2 - (phi1 - phi2)
    q[2, :] = np.pi - phi3
   
    for i in range(l):
        J = manipulator.J( q[:,i] )
        dq[:,i] = np.linalg.inv(J) @ dr[:,i]
        ddq[:,i] = np.linalg.inv(J) @ ddr[:,i]

    
    
    return q, dq, ddq



def q2torque( q, dq, ddq , manipulator ):
    """

    Parameters
    ----------
    q   : numpy array float 3 x l
    dq  : numpy array float 3 x l
    ddq : numpy array float 3 x l
    
    manipulator : pyro object 

    Returns
    -------
    tau   : numpy array float 3 x l

    """
    # Time discretization
    l = q.shape[1]
    
    # Number of DoF
    n = 3
    
    # Output dimensions
    tau = np.zeros((n,l))
    
    #################################
    # Votre code ici !!!
    ##################################
    for i in range(l):
  
        q_i = q[:, i]
        dq_i = dq[:, i]
        ddq_i = ddq[:, i]
        

        M = manipulator.mass_matrix(q_i)

        C = manipulator.coriolis_matrix(q_i, dq_i)
        g = manipulator.gravity_vector(q_i)
        

        tau[:, i] = M @ ddq_i + C @ dq_i + g
    

    
    
    return tau

print(f([0,0,0,0,0]))



