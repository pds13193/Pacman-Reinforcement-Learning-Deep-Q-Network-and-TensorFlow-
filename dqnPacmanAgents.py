# Used code from: INFO 550 Project 4  Homework

from collections import deque    # Importing the deque class for creating a replay memory buffer
from deepQNetwork import *       # Importing the deepQNetwork module containing the DQN class
import numpy as npy              # Importing the NumPy library for numerical operations
from pacman import Directions    # Importing the Directions module from the pacman game

import util                      
import sys                       
import random
import game
import time
import pdb


class PacmanDQN(game.Agent):
    def __init__(self, args):

        """
        Initializes the PacmanDQN agent.

        Args:
            args: Dictionary of arguments specifying the game environment.
        """

        self.Training_Cost_Loss = 0          # cost refers to the discrepancy between the predicted Q-values and the target Q-values
        self.Training_Iteration_Count = 0    # Stores the count of training iterations
        self.Number_Of_Episodes = 0          # Stores the number of episodes
        self.Last_Episode_Score = 0          # Stores the score of the last episode
        self.Last_Step_Reward = 0            # Stores the reward of the last step

        self.Model_Save_Frequency = 1        # Frequency at which the model will be saved 
        self.Epsilon = 1.0                   # Exploration rate (initially set to 1.0)
        self.Final_Epsilon = 0.1             # Final exploration rate

        self.Sample_Experiences_Per_Training = 32    # Number of experiences sampled per training iteration
        self.Pre_Learning = 5000                     # Number of Training steps
        self.Epsilon_Reduction_Interval = 10000      # Interval for reducing epsilon
        self.Replay_Memory_Size = 100000             # Size of the replay memory buffer

        self.Training_QValues = []                   # List to store training Q-values
        self.Current_Time = time.time()
        self.Experience_Memory = deque()

        self.Trained_Model_Save = "Trained"         # File name for saving the trained model
        
        self.Maze_Height = args['height']           # Height of the maze environment
        self.Maze_Width = args['width']             # Width of the maze environment
       

        self.DQN = DQN(self.Maze_Width, self.Maze_Height)                      # Creating an instance of the DQN class
        self.Global_Step = self.DQN.Global_Step.eval(session=self.DQN.Session) # Evaluating the global step count     

        print("Deep Q-Network Agent Initialization") 

    def registerInitialState(self, initialState):
        
        """
        Registers the initial state of the game.

        Args:
        initialState: Initial state of the game.

        """
        
        # Stores the terminal state, last state of the game, previous action taken, status of winning and all are initialized as None
        self.State_Terminal = None                 
        self.State_Last = None                      
        self.Previous_Action = None
        self.Status_Won = True

        # Stores the current reward, reward of the last step, iteration count, cumulative reward for the current episode, score of the last episode and all are initialized to 0
        self.Current_Reward = 0
        self.Last_Step_Reward = 0
        self.Iteration = 0
        self.Episode_Reward = 0
        self.Last_Episode_Score = 0

        self.Number_Of_Episodes = self.Number_Of_Episodes + 1    # Increments the number of episodes by 1                   
        self.State_Current = self.getLayoutState(initialState)   # Fetches the current state of the game layout
         
    def getAction(self, state):
        """
        Determines the action to take based on the current state.

        Args:
            state: Current state of the game.

        Returns:
            Action to be taken.

        """
        
        Action_Legal = state.getLegalActions(0)    # Retrieves the legal actions available in the current state for the agent

        Action_Current = self.chooseAction(state)  # Chooses an action based on the current state

        if Action_Current not in Action_Legal:     # If the chosen action is not legal, set it to STOP
            Action_Current = Directions.STOP

        return Action_Current                      # Returns the action to be taken

    def chooseAction(self, state):
        
        if self.Epsilon < npy.random.rand():       # Checks if a random number is greater than epsilon
            feed_dict = {
                self.DQN.Input: npy.reshape(self.State_Current, (1, self.Maze_Width, self.Maze_Height, 6)),
                self.DQN.Target_QValue: npy.zeros(shape=(1,), dtype=npy.float32),
                self.DQN.Action_Values : npy.full((1, 4), 0),
                self.DQN.Terminal_States: npy.zeros((1,)),
                self.DQN.Reward_Values: npy.zeros((1,))
            }

            self.Predicted_QValues = self.DQN.sess.run(self.DQN.y, feed_dict=feed_dict)[0]  # Computes predicted Q-values using the neural network
            
            self.Training_QValues.append(max(self.Predicted_QValues))                  # Appends the maximum predicted Q-value to the Training_QValues list
            Max_Pred_QValue = npy.amax(self.Predicted_QValues)                         # Finds the maximum predicted Q-value
            Max_QValue_Index = npy.argwhere(Max_Pred_QValue == self.Predicted_QValues) # Finds the indices of maximum predicted Q-values
            #pdb.set_trace()
            if len(Max_QValue_Index) > 0:
                Random_Index = npy.random.randint(0, len(Max_QValue_Index))             # Randomly selects an index from the maximum Q-value indices
                #pdb.set_trace()
                Direction_Random = self.getDirection(Max_QValue_Index[Random_Index][0]) # Gets the corresponding direction for the random index
            else:
                Direction_Random  = self.getDirection(Max_QValue_Index[0][0]) # Gets the direction for the first index
        else:       
            Direction_Random  = self.getDirection(npy.random.randint(0, 4))   # Chooses a random direction
        
        self.Previous_Action = self.getValue(Direction_Random)  # Saves the chosen action as the previous action    
        
        return Direction_Random                   # Returns the chosen direction

    def getDirection(self, Value_Direction):
        """
        Returns the corresponding direction based on the given value.

        Args:
            Value_Direction: Value representing a direction.

        Returns:
            Corresponding direction.

        """
        if Value_Direction == 3:
            return Directions.WEST
        elif Value_Direction == 2:
            return Directions.SOUTH
        elif Value_Direction == 1:
            return Directions.EAST
        else:
            return Directions.NORTH

    def getValue(self, Direction):
        """
        Returns the corresponding value based on the given direction.

        Args:
            Direction: Direction.

        Returns:
            Corresponding value.

        """
        if Directions.WEST == Direction:
            return 3
        elif Directions.SOUTH == Direction:
            return 2
        elif Directions.EAST == Direction:
            return 1
        else:
            return 0
    
    def observationFunction(self, state):
        """
        Pre-processes the observed state.

        Args:
            state: Observed state.

        Returns:
            Pre-processed state.

        """
        
        self.State_Terminal = False     # Sets the terminal state flag to False
        self.processObservation(state)  # Processes the observation

        return state                    # Returns the pre-processed state

    def processObservation(self, state):
        """
        Processes the observed state and performs necessary computations.

        Args:
            state: Observed state.

        """
        if self.Previous_Action is not None:
            
            # Stores the current state as the last state
            self.State_Last = self.State_Current.copy()

            # Updates the current state with the layout state
            self.State_Current = self.getLayoutState(state)
            
            self.Current_Reward = state.getScore()                            # Obtains the current score
            Reward_Obtained = self.Current_Reward - self.Last_Episode_Score   # Calculates the obtained reward
            
            self.Last_Episode_Score = self.Current_Reward # Updates the last episode score

            if Reward_Obtained < -10:
                self.Last_Step_Reward = -500  # Agent eaten by ghost Get eaten
                self.Status_Won = False
            elif Reward_Obtained < 0:
                self.Last_Step_Reward = -1    # Agent punished
            elif Reward_Obtained > 20:
                self.Last_Step_Reward = 50    # Ghost eaten by agent
            elif Reward_Obtained > 0:
                self.Last_Step_Reward = 10    # Food eaten by agent
            
            if(self.State_Terminal and self.Status_Won):
                self.Last_Step_Reward = 100                    # Marks a terminal state with a win, assigns a high positive reward
                
            self.Episode_Reward = self.Episode_Reward + self.Last_Step_Reward  # Accumulates the episode reward

            Last_Experience = (self.State_Last, 
                               float(self.Last_Step_Reward), 
                               self.Previous_Action, 
                               self.State_Current, 
                               self.State_Terminal) # Last experience is stored into memory

            self.Experience_Memory.append(Last_Experience)   # Appends the last experience to the experience memory

            if len(self.Experience_Memory) > self.Replay_Memory_Size:
                self.Experience_Memory.popleft()             # Removes the oldest experience if memory size exceeds the limit

            if(self.Trained_Model_Save):
                if ((self.Training_Iteration_Count % self.Model_Save_Frequency == 0) and (self.Training_Iteration_Count > self.Pre_Learning)):

                    # Below line saves the trained model in memory - Commented becomes it consumes lot of system memory while execution
                    #self.DQN.save_ckpt('Model Saved/model-' + self.Trained_Model_Save + "_" + str(self.Global_Step) + '_' + str(self.Number_Of_Episodes))
                    #print('Saved the trained model in memory')
                    pass

            
            self.performTraining() # Invokes training process

        self.Training_Iteration_Count = self.Training_Iteration_Count + 1       # Updates the training iteration count
        self.Iteration = self.Iteration + 1
        value = float(self.Global_Step)/ float(self.Epsilon_Reduction_Interval) # Calculates the value for epsilon reduction
        self.Epsilon = max(self.Final_Epsilon,1.00 - value)                     # Updates the epsilon value
    
    def final(self, state):
        
        self.State_Terminal = True                 # Marks the state as terminal
        self.Episode_Reward =  self.Episode_Reward + self.Last_Step_Reward     # Accumulates the episode reward
        self.processObservation(state)             # Processes the final state observation
        
        # Print results on console
        sys.stdout.write("# %1d | Number of Steps taken in current episode: %2d | Global Steps: %2d | Elapsed Time: %2f | Current Episode Reward: %2f | Epsilon: %2f " %
                         (self.Number_Of_Episodes,self.Training_Iteration_Count, self.Global_Step, time.time()- self.Current_Time, self.Episode_Reward, self.Epsilon))
        sys.stdout.write("| Q-Value: %2f | Game Won: %r \n" % (max(self.Training_QValues, default=float('nan')), self.Status_Won))
        sys.stdout.flush()

    def performTraining(self):

        """
        Performs the training process using the experience memory.

        """
        
        if (self.Training_Iteration_Count > self.Pre_Learning):
            
            Sample = random.sample(self.Experience_Memory, self.Sample_Experiences_Per_Training)   # Samples experiences from the memory
            Sample_State, Sample_Reward, Sample_Action, Sample_Next_State, Sample_Terminal_State = [], [], [], [], []
            
            # Collects terminal state, next state, action, reward, state from the sample
            Sample_Terminal_State = [i[4] for i in Sample]
            Sample_Next_State = [j[3] for j in Sample]
            Sample_Action = [k[2] for k in Sample]
            Sample_Reward = [l[1] for l in Sample]
            Sample_State = [m[0] for m in Sample]
            
            # Calls the DQN model's train method to update the Q-values
            self.Global_Step, self.Training_Cost_Loss = self.DQN.train(npy.asarray(Sample_State), 
                                                                       self.actionsOneHotEncode(npy.asarray(Sample_Action)),
                                                                       npy.asarray(Sample_Terminal_State), 
                                                                       npy.asarray(Sample_Next_State), 
                                                                       npy.asarray(Sample_Reward))

    def actionsOneHotEncode(self, actions):

        """
        Converts action values into one-hot encoded format.

        Args:
            actions: List of action values.

        Returns:
            One-hot encoded actions.

        """
        
        OneHotEncoded_Actions = 0 * npy.ones((self.Sample_Experiences_Per_Training, 4)) # Initializes an array for one-hot encoded actions

        Actions_Len = len(actions)
        
        for a in range(Actions_Len):     
            b = int(actions[a])              # Gets the action value
            OneHotEncoded_Actions[a][b] = 1  # Assigns 1 to the corresponding action index  

        return OneHotEncoded_Actions   

   
    def getLayoutState(self, state):
        """
        Retrieves the layout state from the given game state.

        Args:
            state: Current game state.

        Returns:
            All_Layouts: Matrix representing the layout state.

        """     
        
        def getLayoutWall(state):
            """
            Retrieves the layout with wall coordinates.

            Args:
                state: Current game state.

            Returns:
                Layout: Layout with wall coordinates marked as 1.

            """
            Data = state.data
            h = Data.layout.height
            w = Data.layout.width

            Wall_Grid = state.data.layout.walls

            Layout = npy.zeros((h, w), dtype=npy.int8)
            
            for x in range(Wall_Grid.height):
                for y in range(Wall_Grid.width):
                    if Wall_Grid[y][x]:
                        Layout_Box = 1
                    else:
                        Layout_Box = 0
                    Layout[-1-x][y] = Layout_Box

            return Layout        # Return layout with coordinates of wall as 1

        def getLayoutCapsules(state):
            
            Data = state.data
            h = Data.layout.height
            w = Data.layout.width

            Layout = npy.zeros((h, w), dtype=npy.int8)

            Wall_Capsules = state.data.layout.capsules

            for c in Wall_Capsules:
                Layout[-1-c[1], c[0]] = 1
                
            return Layout # Returns layout with coordinates of capsule as 1 

        def getLayoutPacman(state):
            
            Data = state.data
            h = Data.layout.height
            w = Data.layout.width

            Layout = npy.zeros((h, w), dtype=npy.int8)

            Agent_States = state.data.agentStates

            for i in Agent_States:
                if i.isPacman:
                    Layout_Box = 1
                    Position = i.configuration.getPosition()
                    Layout[-1-int(Position[1])][int(Position[0])] = Layout_Box

            return Layout # Return layout with coordinates of pacman as 1


        def getLayoutScaredGhost(state):
            
            Data = state.data
            h = Data.layout.height
            w = Data.layout.width

            Layout = npy.zeros((h, w), dtype=npy.int8)
            Agent_States = state.data.agentStates

            for s in Agent_States:
                if not s.isPacman:
                    if s.scaredTimer > 0:
                        Layout_Box = 1
                        Position = s.configuration.getPosition()  
                        Layout[-1-int(Position[1])][int(Position[0])] = Layout_Box

            return Layout # Returns layout with coordinates of scared ghost as 1

        def getLayoutGhost(state):
            
            Data = state.data
            h = Data.layout.height
            w = Data.layout.width

            Layout = npy.zeros((h, w), dtype=npy.int8)

            Agent_States = state.data.agentStates

            for s in Agent_States:
                if not s.isPacman:
                    if not s.scaredTimer > 0:
                        Layout_Box = 1
                        Position = s.configuration.getPosition()
                        Layout[-1-int(Position[1])][int(Position[0])] = Layout_Box

            return Layout # Returns layout with coordinates of ghost as 1

        def getLayoutFood(state):
            
            Data = state.data
            Wall_Grid = Data.food
            
            h = Data.layout.height
            w = Data.layout.width

            
            Layout = npy.zeros((h, w), dtype=npy.int8)
            
            for x in range(Wall_Grid.height):
                for y in range(Wall_Grid.width):
                    if Wall_Grid[x][y]:
                        Layout_Cell = 1
                    else:
                        Layout_Cell = 0     
                    Layout[-1-x][y] = Layout_Cell

            return Layout # Returns layout with coordinates of food as 1

        # Layout matrix is created as a combination of capsule, wall, ghost, pacman, and food layouts

        Maze_Height = self.Maze_Height

        Maze_Width = self.Maze_Width   
        
        All_Layouts = npy.zeros((6, Maze_Height, Maze_Width))
        
        # Stores capsule, food, scared ghost, ghost, pacman and wall layout
        All_Layouts[5] = getLayoutCapsules(state)
        All_Layouts[4] = getLayoutFood(state)
        All_Layouts[3] = getLayoutScaredGhost(state)
        All_Layouts[2] = getLayoutGhost(state)
        All_Layouts[1] = getLayoutPacman(state)
        All_Layouts[0] = getLayoutWall(state)                                              

        All_Layouts = npy.swapaxes(All_Layouts, 0, 2)

        return All_Layouts # Returns wall, ghosts, food, capsules layouts

