import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Set_Up import *
from pathlib import Path

def get_new_action(model,sample,next_state):
    possible_actions = np.linspace(-10,10,num=1000)
    possible_actions = possible_actions.reshape(1000,1)

    new_sample = np.repeat(sample,1000, axis=0)
    next_states = np.repeat(next_state,1000,axis=0)

    pre1_X = np.hstack((new_sample,next_states))
    pre_X = np.hstack((pre1_X,possible_actions))
    k,l = pre_X.shape

    X = pre_X.reshape((k,2,int(l/2))) 
    Y = model.predict(X)
    max_position = np.argmax(Y)
    chosen_action = possible_actions[max_position]

    return chosen_action

def state_2_sample(state,action):
    sample = np.hstack((state,action.reshape((1,1))))

    return sample

path = Path(__file__).parent/"model.h5"
state_path = Path(__file__).parent/"states.py"

Q_model = keras.models.load_model(path)
previous_sample = np.zeros((1,5))
previous_sample[0,2] = -np.pi

current_state = np.zeros((1,4))
current_state[0,2] = -np.pi

theta_double_dot = 0
x_double_dot = 0
step_size=500

Reward_saver = np.zeros((step_size,1))
x_values = np.zeros((step_size,1))
theta_values = np.zeros((step_size,1))

simulation_states = np.empty((5000,4))
all_actions = np.empty((500,1))
for step in range(step_size):
    print(step)
    chosen_action = get_new_action(Q_model,previous_sample,current_state)
    all_actions[step,0] = chosen_action
    current_sample = state_2_sample(current_state,chosen_action)
    Reward_saver[step,0] = reward(current_state)
    x_values[step,0] = current_state[0,0]
    theta_values[step,0] = current_state[0,2]
    for i in range(10):
        simulation_states[i+step*10,:] = current_sample[0,0:4] 
        #Get the next state with the current action
        next_state, theta_double_dot, x_double_dot = next_state_testing(current_sample, theta_double_dot, x_double_dot, chosen_action)
        previous_sample = current_sample.copy()
        current_sample = state_2_sample(next_state,chosen_action)
    
    current_state = next_state.copy()
#Save the data for simulation
np.save(state_path, simulation_states)
X = np.arange(500)
#############################################################
################ Plotting actions  ##########################
#############################################################
# plotting the points  
plt.plot(X, all_actions) 
  
# naming the x axis 
plt.xlabel('Step number') 
# naming the y axis 
plt.ylabel('Action') 
  
# giving a title to my graph 
plt.title('Action decided in current step') 
  
# function to show the plot 
plt.show()

#############################################################
################ Plotting x values ##########################
#############################################################
# plotting the points  
plt.plot(X, x_values) 
  
# naming the x axis 
plt.xlabel('Step number') 
# naming the y axis 
plt.ylabel('x values') 
  
# giving a title to my graph 
plt.title('x process according to the step number') 
  
# function to show the plot 
plt.show()

##############################################################
################## Plotting theta values #####################
##############################################################
# plotting the points  
plt.plot(X, theta_values) 
  
# naming the x axis 
plt.xlabel('Step number') 
# naming the y axis 
plt.ylabel('theta values') 
  
# giving a title to my graph 
plt.title('theta process according to the step number') 
  
# function to show the plot 
plt.show()

##############################################################
################## Plotting rewards ##########################
##############################################################
plt.plot(X, Reward_saver) 
  
# naming the x axis 
plt.xlabel('Step number') 
# naming the y axis 
plt.ylabel('Reward') 
  
# giving a title to my graph 
plt.title('Rewards for current step') 
  
# function to show the plot 
plt.show()

Reward_saver = np.cumsum(Reward_saver,axis = 0)
# plotting the points  
plt.plot(X, Reward_saver) 
  
# naming the x axis 
plt.xlabel('Step number') 
# naming the y axis 
plt.ylabel('Reward') 
  
# giving a title to my graph 
plt.title('Total rewards till the current step') 
  
# function to show the plot 
plt.show()