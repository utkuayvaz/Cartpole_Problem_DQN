import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from Set_Up import *

def get_RNN_Model():
    #1st number of examples
    #2nd number of timesteps
    #3rd input shape
    model = keras.Sequential()
    #Model Structure
    model.add(layers.LSTM(64,input_shape=(2,5)))
    model.add(layers.Dense(64,activation='tanh'))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1))

    #Loss Function
    model.compile(loss='MSE', optimizer='RMSprop', metrics=['MSE'])
    return model

def get_new_action_epsilon(model,sample,next_state):
    epsilon = np.random.rand(1)[0]
    if epsilon>0.7:
        chosen_action  = (np.random.rand(1)[0]-0.5)*20
    else:
        possible_actions = np.linspace(-10,10,num=1001)
        possible_actions = possible_actions.reshape(1001,1)

        new_sample = np.repeat(sample,1001, axis=0)
        next_states = np.repeat(next_state,1001,axis=0)

        pre1_X = np.hstack((new_sample,next_states))
        pre_X = np.hstack((pre1_X,possible_actions))
        k,l = pre_X.shape
        X = pre_X.reshape((k,2,int(l/2))) 
        Y = model.predict(X)
        max_position = np.argmax(Y)
        chosen_action = possible_actions[max_position]
    
    return chosen_action

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

def save_input(Training_Data,RNN_input,step):
    Training_Data_copy = Training_Data.copy()
    Training_Data_copy[step,:,:] = RNN_input

    return Training_Data_copy 

def save_reward(All_Rewards,reward,step):
    Reward_copy = All_Rewards.copy()
    Reward_copy[step] = reward

    return Rewar_copy

#Creating the model
Q_model = get_RNN_Model()
Q_theta_model = Q_model
path = Path(__file__).parent/"model1.h5"
#Hyperparameters
gamma = 0.9
step_size = 500
example = 15
episode_number = 20
C = 5

Reward_saver = np.zeros((episode_number,1))

#Creating first and second sample for training
first_state = np.zeros((1,4))
first_state[0,0] = 0
first_state[0,1] = 0
first_state[0,2] = -np.pi
first_state[0,3] = 0

#Model training
for episode in range(episode_number):
    theta_double_dot = np.zeros((10,1))
    x_double_dot = np.zeros((10,1))

    theta_double_dot = np.zeros((example,1))
    x_double_dot = np.zeros((example,1))
    current_state = np.zeros((example,4))
    previous_sample = np.zeros((example,5))

    T1_Sample = np.zeros((example,5))
    T2_Sample = np.zeros((example,5))
    chosen_action = np.zeros((example,1))
    next_action = np.zeros((example,1))
    current_sample = np.zeros((example,5))
    next_sample = np.zeros((example,5))
    Input_Samples = np.zeros((example,2,5))
    next_state = np.zeros((example,4))
    y = np.zeros((example,1))
    
    for i in range(example):
        current_state[i,:] = random_starting_state(i,example)
        previous_sample[i,:] = random_starting_sample(i,example)

    for step in range(step_size):
        Reward = np.zeros((example,1))
        #Preparation for examples
        for i in range(example):
            #Prepare Samples
            chosen_action[i,0] = get_new_action_epsilon(Q_model,previous_sample[i,:].reshape((1,5)),current_state[i,:].reshape((1,4)))
            current_sample[i,:] = state_2_sample(current_state[i,:].reshape((1,4)),chosen_action[i,0].reshape((1,1)))
            T1_Sample[i,:] = previous_sample[i,:]
            T2_Sample[i,:] = current_sample[i,:]
            #Prepare Rewards
            Reward[i,0] = reward(current_sample[i,:].reshape((1,5)))

        for i in range(example):
            for mini_step in range(10):
                #Get the next state with the current action
                next_state[i,:],theta_double_dot[i,0], x_double_dot[i,0] = next_state_testing(current_sample[i,:].reshape((1,5)), theta_double_dot[i,0], x_double_dot[i,0], chosen_action[i,0])
                previous_sample[i,:] = current_sample[i,:]
                current_sample[i,:] = state_2_sample(next_state[i,:].reshape((1,4)),chosen_action[i,0].reshape((1,1)))
            #Get the next actions for chosen states
            next_action[i,0] = get_new_action(Q_theta_model, previous_sample[i,:].reshape((1,5)), next_state[i,:].reshape((1,4)))
            
            next_sample[i,:] = np.hstack((next_state[i,:].reshape((1,4)),next_action[i,0].reshape((1,1))))

            Next_Training_sample = np.vstack((previous_sample[i,:].reshape((1,5)), next_sample[i,:].reshape((1,5))))
            Next_Training_sample = Next_Training_sample.reshape((1,2,5))

            y[i,0] = Reward[i,0].reshape((1,1))+ gamma * Q_theta_model.predict(Next_Training_sample) 
            #Train with epochs
            Input_sample = np.vstack((T1_Sample[i,:].reshape((1,5)),T2_Sample[i,:].reshape((1,5))))
            Input_sample = Input_sample.reshape((1,2,5))
            Input_Samples[i,:,:] = Input_sample
        Q_model.fit(Input_Samples,y,epochs=1)
        if step%C==0:
            Q_theta_model = Q_model
        print("current Episode: "+ str(episode+1)+" Current step is: "+str(step+1))
        for i in range(example):
            if np.abs(current_sample[i,0]) == 6:
                current_state[i,:] = random_starting_state(i,example)
                previous_sample[i,:] = random_starting_sample(i,example)
            else:
                #Samples for the next step
                current_state[i,:] = next_sample[i,0:4].reshape((1,4))

    #Test the current model
    test_previous_sample = np.zeros((1,5))
    test_previous_sample[0,2] = first_state[0,2]

    test_current_state = first_state.copy()
    test_theta_double_dot = 0
    test_x_double_dot = 0
    for step in range(step_size):
        test_chosen_action = get_new_action(Q_model,test_previous_sample,test_current_state)
        test_current_sample = state_2_sample(test_current_state,test_chosen_action)
        Reward_saver[episode,0] += reward(test_current_state)
        for i in range(10):
            #Get the next state with the current action
            test_next_state,test_theta_double_dot, test_x_double_dot = next_state_testing(test_current_sample, test_theta_double_dot, test_x_double_dot, test_chosen_action)
            test_previous_sample = test_current_sample.copy()
            test_current_sample = state_2_sample(test_next_state,test_chosen_action)
        
        test_current_state = test_next_state.copy()    
    #Print out the quality of the data
    print("Episode number: "+str(episode+1))
    print("Total Reward: "+str(Reward_saver[episode,0]))
    if Reward_saver[episode,0]>-100:
        Q_model.save(path)