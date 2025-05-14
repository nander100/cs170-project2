# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached, 
the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import matplotlib.pyplot as plt
import numpy as np
import math


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    v = [0] * NUM_STATES # initialize v = 0
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    # At each iteration, you may need to keep track of pi to perform logging
    threashhold = 1e-4
    for i in range(max_iterations): # itterate across all the episodes
        v_prev = v.copy() # create a a copy of the old v to compare with the new v
        for state in range(NUM_STATES):
            # calcuate the values for each action in the state
            action_values = []
            for action in range(NUM_ACTIONS):
                current_action_value = 0 # stores the value for the current action

                #Calculates the value of the current action
                #t is a list of four-element tuples in the form of (p, s_, r, terminal)
                for prob, s_prime, reward, terminal in TRANSITION_MODEL[state][action]:
                    #IMPORTANT: Bellman equation update
                    current_action_value += prob * (reward + gamma * v_prev[s_prime]) 

                action_values.append(current_action_value) # add the expected value to a list of values

            # update the expected value of the state 
            # IMPORTATNT TO DO BEFORE UPDATING POLICY OR ELSE WILL EXIT AFTER 0 ITTERATIONS
            v[state] = max(action_values)

            # Update policy to use the best action
            pi[state] = action_values.index(max(action_values))
        
        # log the data
        logger.log(i, v, pi)
        if max(abs(v[s] - v_prev[s]) for s in range(NUM_STATES)) < threashhold:
            print(f"Value iteration converged after {i} iterations")
            break

    return pi

def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)
    
    #My code starts here
    # At each iteration, you may need to keep track of pi to perform logging
    threashhold = 1e-4
    # Initialize with a random policy
    pi = [random.randint(0, NUM_ACTIONS-1) for _ in range(NUM_STATES)]
    for i in range(max_iterations): # itterate across all the episodes
        v_prev = v.copy()
        delta = 0 # change in value for each policy itteration
        for state in range(NUM_STATES):
            # get the current state from the policy
            curr_action = pi[state] 
            current_action_value = 0
            #Calculates the value of the current action
            #t is a list of four-element tuples in the form of (p, s_, r, terminal)
            for prob, s_prime, reward, terminal in TRANSITION_MODEL[state][curr_action]:
                #IMPORTANT: Bellman equation update
                current_action_value += prob * (reward + gamma * v_prev[s_prime]) 

            # This takes the highest change because we want the values for all the states to converge.
            v[state] = current_action_value # updates the value for the new action 
            delta = max(delta, abs(v[state] - v_prev[state])) # track the change
        
        # Visualize the initial value and policy
        logger.log(i, v, pi)

        # determines if the expected value converges
        if delta < threashhold:
            break

        # POLICY ITTERATION STAGE
        policy_stable = True # keep track of whether or not the policy is stable
        for state in range(NUM_STATES):
            old_action = pi[state]
            
            # Find best action based on one-step lookahead
            action_values = []
            for action in range(NUM_ACTIONS):
                action_value = 0
                for prob, next_state, reward, terminal in TRANSITION_MODEL[state][action]:
                    action_value += prob * (reward + gamma * v[next_state])
                action_values.append(action_value)
            
            # Update policy to use the best action
            pi[state] = action_values.index(max(action_values))
            
            # Check if policy has changed
            if old_action != pi[state]:
                policy_stable = False
        
        # Log the updated policy
        logger.log(i, v, pi)
        
        # Check if policy has converged
        if policy_stable:
            print(f"Policy converged after {i+1} iterations")
            break
    return pi

import random # Make sure to import the random module
import math   # For math.inf

def q_learning(env, gamma, max_iterations, logger, convergence_tolerance=1e-4):
    """
    Implement Q-learning to return a deterministic policy for all states.
    Includes a convergence check based on the change in the value function.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (total steps) that should be performed;
        the algorithm should terminate when max_iterations is exceeded or convergence is met.
        Consider increasing this if the agent is not converging.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    convergence_tolerance: float, optional
        The tolerance for the infinity norm of the value function difference
        to determine convergence. Default is 1e-4.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    # Initialize value function and policy
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)
    
    # Hyperparameters
    alpha = 0.1        # Learning rate
    epsilon = 1.0      # Exploration rate
    epsilon_decay = 0.999  # Decay rate for epsilon
    min_epsilon = 0.05     # Minimum exploration rate
    
    # Convergence parameters
    convergence_tolerance = 0.1  # Tolerance for convergence
    min_iterations_before_convergence = 1000  # Minimum steps before checking convergence
    check_convergence_every = 100  # Check for convergence every N steps
    
    # Initialize Q-table
    Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    prev_Q = [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]
    
    # Learning loop
    steps = 0
    episodes = 0
    
    while steps < max_iterations:
        episodes += 1
        current_state = env.reset()
        terminal = False
        
        # One episode
        while not terminal and steps < max_iterations:
            steps += 1
            
            # Choose action (epsilon-greedy)
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = pi[current_state]
            
            # Take action
            next_state, reward, terminal, _ = env.step(action)
            
            # Q-learning update
            best_next_q = max(Q[next_state]) if not terminal else 0
            Q[current_state][action] += alpha * (reward + gamma * best_next_q - Q[current_state][action])
            
            # Update policy and value
            best_action = Q[current_state].index(max(Q[current_state]))
            pi[current_state] = best_action
            v[current_state] = Q[current_state][best_action]
            
            # Move to next state
            current_state = next_state
            
            # Decay exploration rate
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            # Periodically log progress
            if steps % check_convergence_every == 0:
                logger.log(steps, v, pi)
                
                # Convergence check (after minimum iterations)
                if steps > min_iterations_before_convergence:
                    # Calculate maximum change in Q-values
                    max_change = 0.0
                    for s in range(NUM_STATES):
                        for a in range(NUM_ACTIONS):
                            max_change = max(max_change, abs(Q[s][a] - prev_Q[s][a]))
                    
                    # copy current Q-values for next comparison
                    for s in range(NUM_STATES):
                        for a in range(NUM_ACTIONS):
                            prev_Q[s][a] = Q[s][a]
                    
                    # Print convergence status
                    logger.log(steps, v, pi)
                    print(f"Step {steps}: max Q-value change = {max_change:.4f}")
                    
                    # Check if converged
                    if max_change < convergence_tolerance:
                        print(f"Converged after {steps} steps, {episodes} episodes")
                        steps = max_iterations  # Force exit from outer loop
                        break
    
    # Final logging
    logger.log(steps, v, pi)
    print(f"Q-learning completed: {episodes} episodes, {steps} steps")
    
    return pi
if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q-Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()