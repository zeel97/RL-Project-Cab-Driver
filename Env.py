# Import routines

import numpy as np
import math
import random
from itertools import permutations 

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
max_hours = 24*30

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = list(permutations([0 ,1, 2, 3, 4], 2))
        self.action_space.append([0,0]) #append the [0,0] to the action space
        self.cities = [0,1,2,3,4]
        self.time = np.arange(0,t)
        self.day = np.arange(0,d)
        self.state_space = [[city, time, day] 
                            for city in self.cities  
                            for time in self.time 
                            for day in self.day] 
        random_state = np.random.choice(len(self.state_space))
        self.state_init = self.state_space[random_state]
        self.hours_meter = 0
        self.terminal_state = False

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        """Steps taken:
        1. Apply one hot encoding using numpy to convert numerical value to a one hot encoded vector for location,time and date
        2. Concatenated the 3 together
        """

        #Location to vector
        location = np.array(state[0]).reshape(-1)
        location = np.eye(m)[location]
        
        #Time of the day to vector
        time = np.array(state[1]).reshape(-1)
        time = np.eye(t)[time]
        
        #Day of the week to vector
        day = np.array(state[2]).reshape(-1)
        day = np.eye(d)[day]        
        
        #Concatenate
        state_encod = np.concatenate((location,time,day), axis=None)
        
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        
        # print("Before request count - Hours meter",self.hours_meter)
        
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)


        if requests >15:
            requests =15

        #To get possible actions without [0,0]
        self.action_space.remove([0,0])
        possible_actions_index = random.sample(range(0, len(self.action_space)), requests)
        #Adding [0,0] to possible actions 
        self.action_space.append([0,0])
        possible_actions_index.append(self.action_space.index([0,0]))
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        day = state[2]
        time = state[1]
        current_loc = state[0]
        time_to_pick_up = Time_matrix[current_loc][action[0]][time][day]
        trip_time = Time_matrix[action[0]][action[1]][time][day]
        reward = 0
        if action != [0,0]:
            reward = (R * trip_time) - (C * (trip_time + time_to_pick_up))
        else:
            reward =- C 
        
        return reward



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        #Get the reward
        reward = self.reward_func(state,action,Time_matrix)
        
        trip_time = Time_matrix[action[0]][action[1]][state[1]][state[2]]
        time_to_pick_up = Time_matrix[state[0]][action[0]][state[1]][state[2]]
                        
        next_state = state
        
        #Update the location and calculate the hour of the day
        if action != [0, 0]:
            next_state[0] = action[1]
            next_time = state[1] + int(trip_time) + int(time_to_pick_up) #new time is the trip time + pick up time
            self.hours_meter = self.hours_meter + int(trip_time) + int(time_to_pick_up)
        else:
            # print('Increase time by 1')
            next_time = state[1] + 1 #the location stays the same with time is increased by 1 hour
            self.hours_meter += 1
            
        #Check if it's the next day
        if next_time >= 24:
            next_state[2] += 1
            next_state[1] = next_time - 24
        else:
            next_state[1] = next_time
            
       #Check to reset day of the week
        if next_state[2]>= 7:
            next_state[2] -= 7
        
        
        #Update total time
        if self.hours_meter >= max_hours:
            self.terminal_state = True
            self.hours_meter = 0
        else:
            self.terminal_state = False
        
        return (next_state, self.terminal_state, reward)



    def reset(self):
        return self.action_space, self.state_space, self.state_init
