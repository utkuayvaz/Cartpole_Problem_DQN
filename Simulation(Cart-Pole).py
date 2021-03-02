import random
from itertools import count
import time
import ffmpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from pathlib import Path


path = Path(__file__).parent/"states.npy"
video_path = Path(__file__).parent/"im.mp4"

fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-0.8,0.8))
line, = ax.plot([], [], lw=2,color="black")
line1, = ax.plot([], [], lw=2,color="black")
line2, = ax.plot([], [], lw=2,color="black")
line3, = ax.plot([], [], lw=2,color="black")
line4, = ax.plot([], [], lw=2,color="black")
line5, = ax.plot([],[],'bo',lw=2)

plt.xlabel('x axis(meter)')
plt.ylabel('y axis(meter)')

def init():
    line.set_data([], [])
    line1.set_data([],[])
    line2.set_data([],[])
    line3.set_data([],[])
    line4.set_data([],[])
    line5.set_data([],[])
    return line,line1,line2,line3,line4,line5

def animate(i):
    State_Data = np.load(path,allow_pickle=True)
    cartpole_x = State_Data[i,0]
    theta_value = State_Data[i,2]

    #Cartpole positions
    left_upper_corner_x = cartpole_x-0.5
    left_upper_corner_y = 0.1

    right_upper_corner_x = cartpole_x+0.5
    right_upper_corner_y = 0.1

    right_down_corner_x = cartpole_x+0.5
    right_down_corner_y = -0.1

    left_down_corner_x = cartpole_x-0.5
    left_down_corner_y = -0.1

    #Pole simulation
    pole_x = np.sin(theta_value)*0.6+cartpole_x
    pole_y = 0.6*np.cos(theta_value)

    #The handle position
    end_of_handle_x = pole_x
    end_of_handle_y = pole_y
    beginning_of_handle_x = cartpole_x
    beginning_of_handle_y = 0

    #Create simulations
    #Cartpole
    upper_bound_cartpole_x = np.linspace(left_upper_corner_x,right_upper_corner_x,1000)
    upper_bound_cartpole_y = np.linspace(left_upper_corner_y,right_upper_corner_y,1000)

    down_bound_cartpole_x = np.linspace(left_down_corner_x,right_down_corner_x,1000)
    down_bound_cartpole_y = np.linspace(left_down_corner_y,left_down_corner_y,1000)

    left_bound_cartpole_x = np.linspace(left_down_corner_x,left_upper_corner_x,1000)
    left_bound_cartpole_y = np.linspace(left_down_corner_y,left_upper_corner_y,1000)

    right_bound_cartpole_x = np.linspace(right_down_corner_x,right_upper_corner_x,1000)
    right_bound_cartpole_y = np.linspace(right_down_corner_y,right_upper_corner_y,1000)

    #Handle
    handle_x = np.linspace(beginning_of_handle_x,end_of_handle_x,1000)
    handle_y = np.linspace(beginning_of_handle_y,end_of_handle_y,1000)

    #Pole
    pole_x = np.linspace(end_of_handle_x,end_of_handle_x,1000)
    pole_y = np.linspace(end_of_handle_y,end_of_handle_y,1000)

    x_data = np.empty((1000,5))
    y_data = np.empty((1000,5))

    x_data[:,0] = upper_bound_cartpole_x
    x_data[:,1] = down_bound_cartpole_x
    x_data[:,2] = left_bound_cartpole_x
    x_data[:,3] = right_bound_cartpole_x
    x_data[:,4] = handle_x

    y_data[:,0] = upper_bound_cartpole_y
    y_data[:,1] = down_bound_cartpole_y
    y_data[:,2] = left_bound_cartpole_y
    y_data[:,3] = right_bound_cartpole_y
    y_data[:,4] = handle_y

    line.set_data(x_data[:,0],y_data[:,0])
    line1.set_data(x_data[:,1],y_data[:,1])
    line2.set_data(x_data[:,2],y_data[:,2])
    line3.set_data(x_data[:,3],y_data[:,3])
    line4.set_data(x_data[:,4],y_data[:,4])
    line5.set_data(pole_x,pole_y)
    return line,line1,line2,line3,line4,line5

anim = FuncAnimation(fig, animate, init_func=init,frames=5000, interval=10)
anim.save(video_path)
print("Save complete")
plt.show()
