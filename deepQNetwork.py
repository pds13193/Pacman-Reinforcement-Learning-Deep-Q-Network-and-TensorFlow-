import numpy as np
import tensorflow.compat.v1 as TFlow

TFlow.disable_eager_execution()      # Disable eager execution for TensorFlow 2 compatibility

import pdb

class DQN:
    def __init__(self, *args):
        
        # Initialize DQN class

        print(TFlow.__version__)

        self.Network = 'Q-Network'
        self.Session = TFlow.compat.v1.Session()
        w = args[0]                 # Width of the input
        h = args[1]                 # Height of the input
        Reward_Discount = 0.5       # Discount factor for future rewards
        Learning_Rate = 0.0002      # Learning rate for the optimizer
        File_Load = None            # File to load model checkpoints from (if any)
        
        # Define placeholders for input tensors
        self.Action_Values = TFlow.placeholder("float", [None, 4], name=self.Network + '_Action_Values')

        self.Target_QValue = TFlow.placeholder('float', [None], name=self.Network + '_QValue')
        self.Input = TFlow.placeholder('float', [None, w, h, 6],name=self.Network + '_Input')
        self.Reward_Values = TFlow.placeholder("float", [None], name=self.Network + '_Reward_Values')
        self.Terminal_States = TFlow.placeholder("float", [None], name=self.Network + '_Terminal_States')

        # Convolutional Layer 1
        # s = size
        # c = Channels
        # f = Filters
        # st = Stride
        
        Layer = 'conv1' ; s = 3 ; c = 6 ; f = 16 ; st = 1   # Layer name
        
        self.Weight_1 = TFlow.Variable(TFlow.random.normal([s,s,c,f], stddev=0.01),name=self.Network + '_'+Layer+'_Weights')
        self.Stride_1 = TFlow.nn.conv2d(self.Input, self.Weight_1, strides=[1, st, st, 1], padding='SAME',name=self.Network + '_'+Layer+'_convs')

        self.Bias_1 = TFlow.Variable(TFlow.constant(0.1, shape=[f]),name=self.Network + '_'+Layer+'_Biases')
        self.Activations_1 = TFlow.nn.relu(TFlow.add(self.Stride_1,self.Bias_1),name=self.Network + '_'+Layer+'_Activations')

        # Convolutional Layer 2
        # s = size
        # c = Channels
        # f = Filters
        # st = Stride
        
        Layer = 'conv2' ; s = 3 ; c = 16 ; f = 32 ; st = 1  # Layer name

        self.Weight_2 = TFlow.Variable(TFlow.random.normal([s,s,c,f], stddev=0.01),name=self.Network + '_'+Layer+'_Weights')
        self.Stride_2 = TFlow.nn.conv2d(self.Activations_1, self.Weight_2, strides=[1, st, st, 1], padding='SAME',name=self.Network + '_'+Layer+'_convs')

        self.Bias_2 = TFlow.Variable(TFlow.constant(0.1, shape=[f]),name=self.Network + '_'+Layer+'_Biases')
        self.Activations_2 = TFlow.nn.relu(TFlow.add(self.Stride_2,self.Bias_2),name=self.Network + '_'+Layer+'_Activations')
        
        Activations_2_Shape = self.Activations_2.get_shape().as_list()        

        # Fully connected Layer 3
        # d = Dimension
        # h = hiddens

        Layer = 'fc3' ; h = 256 ; d = Activations_2_Shape[1]*Activations_2_Shape[2]*Activations_2_Shape[3]                   # Layer name and compute the total number of input features
        self.Activations_2_Flat = TFlow.reshape(self.Activations_2, [-1,d],name=self.Network + '_'+Layer+'_input_flat')
        self.Weight_3 = TFlow.Variable(TFlow.random.normal([d,h], stddev=0.01),name=self.Network + '_'+Layer+'_Weights')
        self.Bias_3 = TFlow.Variable(TFlow.constant(0.1, shape=[h]),name=self.Network + '_'+Layer+'_Biases')
        self.eps = TFlow.add(TFlow.matmul(self.Activations_2_Flat,self.Weight_3),self.Bias_3,name=self.Network + '_'+Layer+'_ips')
        self.Activations_3 = TFlow.nn.relu(self.eps,name=self.Network + '_'+Layer+'_Activations')

        # Fully connected Layer 4
        # d = Dimension
        # h = hiddens
        
        Layer = 'fc4' ; h = 4 ; d = 256          # Layer name, Number of output units, Number of input units
        self.Weight_4 = TFlow.Variable(TFlow.random.normal([d,h], stddev=0.01),name=self.Network + '_'+Layer+'_Weights')
        self.Bias_4 = TFlow.Variable(TFlow.constant(0.1, shape=[h]),name=self.Network + '_'+Layer+'_Biases')
        self.value = TFlow.add(TFlow.matmul(self.Activations_3,self.Weight_4),self.Bias_4,name=self.Network + '_'+Layer+'_Outputs')

        # Q-Value, Optimizer and Reward
        Reward_Discount = TFlow.constant(Reward_Discount)
        self.Value1 = TFlow.add(self.Reward_Values, TFlow.multiply(1.0-self.Terminal_States, TFlow.multiply(Reward_Discount, self.Target_QValue)))
        self.Predicted_QValues = TFlow.reduce_sum(TFlow.multiply(self.value,self.Action_Values), axis=1)
        self.Reward = TFlow.reduce_sum(TFlow.pow(TFlow.subtract(self.Value1, self.Predicted_QValues), 2))
        
        # Global step variable for tracking training progress
        if (File_Load == None):
            self.Global_Step = TFlow.Variable(0, name='global_step', trainable=False)
        elif (File_Load != None):
            self.Global_Step = TFlow.Variable(int(File_Load.split('_')[-1]),name='Global_Step', trainable=False)

        # Model saver for saving and restoring checkpoints
        self.Model_Save = TFlow.train.Saver(max_to_keep=0)

        # Initialize TensorFlow session and variables
        self.Session.run(TFlow.global_variables_initializer())

        # Define the optimizer for training
        self.Train_Optimal = TFlow.train.AdamOptimizer(Learning_Rate).minimize(self.Reward, global_step=self.Global_Step)
        
        # Restore model from checkpoint if specified
        if File_Load != None:
            self.Model_Save.restore(self.Session,File_Load)
            print('Checkpoint is Loaded...')
            
        
    def train(self,Sample_State,Sample_Action,Sample_Terminal,Sample_Next_State,Sample_Reward):

        """
        Trains the model using the provided samples.

        Args:
        Sample_State: Array of input states for training.
        Sample_Action: Array of corresponding actions.
        Sample_Terminal: Array indicating whether each state is terminal or not.
        Sample_Next_State: Array of next states.
        Sample_Reward: Array of corresponding rewards.

        Returns:
        Count: Number of training steps performed.
        Reward: Value of the training reward.
        """
        
        # Compute Q-values for the next states
        feed_dict={self.Input: Sample_Next_State, self.Target_QValue: np.zeros(Sample_Next_State.shape[0]), self.Action_Values: Sample_Action, self.Terminal_States:Sample_Terminal, self.Reward_Values: Sample_Reward}
        
        QValue_Time = self.Session.run(self.value,feed_dict=feed_dict)
        QValue_Time = np.amax(QValue_Time, axis=1)
        
        # Update Q-values using the current states and the computed Q-values for the next states
        feed_dict={self.Input: Sample_State, self.Target_QValue: QValue_Time, self.Action_Values: Sample_Action, self.Terminal_States:Sample_Terminal, self.Reward_Values: Sample_Reward}
        _,Count,Reward = self.Session.run([self.Train_Optimal, self.Step_Global,self.Reward],feed_dict=feed_dict)

        return Count, Reward

    def save_ckpt(self,filename):
        """
        Saves the current model checkpoint to a file.

        Args:
        filename: Name of the file to save the checkpoint.
        """
        self.Model_Save.save(self.Session, filename)

