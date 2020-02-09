import numpy as np

def calculate_states_and_M(trajectory):
    """
    caculates the transition matrix and the states array

    Parameters
    ----------
    trajectory : array_like
        trajectory from previous simulation

    Returns
    ----------
    M : array_like
        transition matrix of the trajectory

    states : array_like
        numpy array including the value of the state (-1 or 1) of each 
        simulation step
    """ 

    x = np.copy(trajectory)

    states = np.zeros(len(x))
    # set the first state randomly to -1 or 1
    np.random.seed(777)
    states[0] = np.random.choice([-1,1])

    left_indices   = np.where(x < -1)
    right_indices  = np.where(1 <  x)
    middle_indices = np.where((-1 < x) & (x < 1))

    # set states of left and right populated positions
    states[left_indices]  = -1
    states[right_indices] =  1

    # now loop over the positions between the cores
    # these states are assigned to the state visited before
    for i in middle_indices[0]:
        if i==0:
            continue
        else:
            states[i] = states[i-1]

    # calculate population of the two states
    N_left  = len(np.where(states==-1)[0])
    N_right = len(np.where(states== 1)[0])

    # calculate the transitions
    # calculate difference of states[i] and states[i-1]
    diff = states - np.roll(states, 1, axis=0)
    # diff = -2 corresponds to transition right -> left
    # diff =  2 corresponds to transition left  -> right
    N_left_right = len(np.where(diff== 2)[0])
    N_right_left = len(np.where(diff==-2)[0])

    # calculate the transition probabilities
    p_left_right = N_left_right / N_left
    p_right_left = N_right_left / N_right
    p_left_left   = 1 - p_left_right
    p_right_right = 1 - p_right_left

    # build transition matrix
    M = np.array([[p_left_left,  p_left_right ],
                  [p_right_left, p_right_right]])

    return M, states
