import numpy as np
import matplotlib.pyplot as plt

def reward(current_state):
    j = np.array([[current_state[0,0],np.sin(current_state[0,2]),np.cos(current_state[0,2])-1]])
    
    #T inverse matrix 
    T = np.zeros((3,3))
    T[0,0] = 1
    T[1,0] = 0.6
    T[0,1] = 0.6
    T[1,1] = 0.36
    T[2,2] = 0.36

    reward =-(1-np.exp(-0.5*(current_state[0,0]**2+1.2*current_state[0,0]*np.sin(current_state[0,2])+0.72-0.72*np.cos(current_state[0,2]))))
    return reward

#########################################################
############### Get the new state #######################
#########################################################
def next_state_training(current_state, theta_double_dot, x_doubledot, action):
    next_state = np.empty((1,4))
    next_state[0,2] = update_theta_training(current_state)
    next_state[0,0] = update_x_training(current_state)
    next_state[0,3] = update_theta_dot_training(current_state, theta_double_dot)
    next_state[0,1] = get_x_dot_training(current_state,x_doubledot)
    next_theta_double_dot = get_theta_doubledot(current_state,action)
    next_x_doubledot = update_x_doubledot(current_state, x_doubledot) 

    return next_state, next_theta_double_dot, next_x_doubledot

def next_state_testing(current_state, theta_double_dot, x_doubledot, action):
    next_state = np.empty((1,4))
    next_state[0,2] = update_theta_simulation(current_state)
    next_state[0,0] = update_x_simulation(current_state)
    next_state[0,3] = update_theta_dot_simulation(current_state, theta_double_dot)
    next_state[0,1] = get_x_dot_simulation(current_state,x_doubledot)
    next_theta_double_dot = get_theta_doubledot(current_state,action)
    next_x_doubledot = update_x_doubledot(current_state, action) 

    return next_state, next_theta_double_dot, next_x_doubledot
#########################################################
############### Updating the x values ###################
#########################################################
def update_x_training(current_state):
    next_x = current_state[0,0] + current_state[0,1]*0.1
    if next_x<-6:
        next_x=-6
    elif next_x>6:
        next_x=6

    return next_x

def update_x_simulation(current_state):
    next_x = current_state[0,0] + current_state[0,1]*0.01
    if next_x<-6:
        next_x=-6
    elif next_x>6:
        next_x=6
    return next_x

def get_x_dot_training(current_state, x_doubledot):
    next_x_dot = current_state[0,1] + x_doubledot*0.1
    if next_x_dot<-10:
        next_x_dot = -10
    elif next_x_dot>10:
        next_x_dot=10
    return next_x_dot

def get_x_dot_simulation(current_state, x_doubledot):
    next_x_dot = current_state[0,1] + x_doubledot*0.01
    if next_x_dot<-10:
        next_x_dot = -10
    elif next_x_dot>10:
        next_x_dot=10
    return next_x_dot

def update_x_doubledot(current_state,action):
    m_1 = 0.5
    m_2 = 0.5
    l = 0.6
    g = 9.82
    b = 0.1

    a1 = 2*m_2*l*(current_state[0,3]**2)*np.sin(current_state[0,2])
    a2 = 3*m_2*g*np.sin(current_state[0,2])*np.cos(current_state[0,2])
    a3 = 4*action - 4*b*current_state[0,1]
    b1 = 4*(m_1+m_2)
    b2 = -3*m_2*(np.cos(current_state[0,2])**2)

    x_doubledot = (a1+a2+a3)/(b1+b2)
    x_doubledot = np.around(x_doubledot, decimals=3)
    return x_doubledot

#########################################################
############### Double Theta calculations ###############
#########################################################
def get_theta_doubledot(current_state,action):
    m_1 = 0.5
    m_2 = 0.5
    l = 0.6
    g = 9.82
    b = 0.1

    a1 = -3*m_2*l*(current_state[0,3]**2)*np.sin(current_state[0,2])*np.cos(current_state[0,2])
    a2 = -6*(m_1+m_2)*g*np.sin(current_state[0,2])
    a3 = -6*(action-b*current_state[0,1])*np.cos(current_state[0,2])
    b1 = 4*l*(m_1+m_2)
    b2 = -3*m_2*l*(np.cos(current_state[0,2])**2)

    theta_doubledot = (a1+a2+a3)/(b1+b2)
    theta_doubledot = np.around(theta_doubledot, decimals=3)
    return theta_doubledot

#########################################################
############### Theta calculations ######################
#########################################################
def update_theta_training(current_state):
    next_theta = current_state[0,2]+current_state[0,3]*0.1
     
    if next_theta > np.pi:
        next_theta = next_theta - 2*np.pi

    if next_theta < -np.pi:
        next_theta = 2*np.pi + next_theta

    return next_theta

def update_theta_simulation(current_state):
    next_theta = current_state[0,2]+current_state[0,3]*0.01
     
    if next_theta > np.pi:
        next_theta = next_theta - 2*np.pi

    if next_theta < -np.pi:
        next_theta = 2*np.pi + next_theta

    return next_theta
#########################################################
############### Theta dot calculations ##################
#########################################################
def update_theta_dot_training(current_state, theta_double_dot):
    next_theta_dot = current_state[0,3] + theta_double_dot*0.1
    if next_theta_dot>10:
        next_theta_dot = 10
    elif next_theta_dot<-10:
        next_theta_dot = -10

    return next_theta_dot

def update_theta_dot_simulation(current_state, theta_double_dot):
    next_theta_dot = current_state[0,3] + theta_double_dot*0.01
    if next_theta_dot>10:
        next_theta_dot = 10
    elif next_theta_dot<-10:
        next_theta_dot = -10

    return next_theta_dot

def random_starting_state(k,max_k):
    state = np.zeros((1,4))
    state[0,0] = 0
    state[0,2] = -np.pi+(k+1)*np.pi*2/max_k

    return state

def random_starting_sample(k,max_k):
    sample = np.zeros((1,5))
    sample[0,2] = -np.pi+(k+1)*np.pi*2/max_k

    return sample