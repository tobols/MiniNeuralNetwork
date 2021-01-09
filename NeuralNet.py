import numpy as np
import matplotlib.pyplot as plt

class NeuralNet():
    '''
    A two layer neural network
    '''

    def __init__(self, layers=[13,8,1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.data = None
        self.truth = None


    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(34) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])     # weight-vector 13:8
        self.params['b1'] = np.random.randn(self.layers[1],)                    # bias-vector   13

        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2])      # weight-vector 8:1
        self.params['b2'] = np.random.randn(self.layers[2],)                    # bias-vector   1


    def relu(self, Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        return np.maximum(0,Z)


    def sigmoid(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1.0/(1.0+np.exp(-Z))


    def entropy_loss(self, y, yhat):
        '''
        Calculate the average loss with respect to all the inputs, so we
        can correct inaccurate predictions.
        '''
        nsample = len(y)
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss


    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''
        Z1 = self.data.dot(self.params['W1']) + self.params['b1']               # Calculate weighted sum
        A1 = self.relu(Z1)                                                      # Activation (is weighted sum above 0?)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']                      # Calculate weighted sum between first and second layer
        result = self.sigmoid(Z2)                                               # Compute output function
        loss = self.entropy_loss(self.truth,result)                             # Calculate loss between predicted and actual truth

        # Save calculated parameters     
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return result, loss
        

    def back_propagation(self,result):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        def dRelu(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        
        dl_wrt_yhat = -(np.divide(self.truth,result) - np.divide((1 - self.truth),(1-result)))
        dl_wrt_sig = result * (1-result)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.data.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)
        
        # Update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2


    def train(self, input_dataset, truth):
        '''
        Trains the neural network using the specified data and labels.
        '''
        print("Training...")
        self.data = input_dataset
        self.truth = truth
        self.init_weights() # Initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)


    def predict(self, data):
        '''
        Predicts on a test data
        '''
        Z1 = data.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)


    def accuracy(self, truth, result):
        '''
        Calculates the accutacy between the predicted value and the truth labels
        '''
        acc = int(sum(truth == result) / len(truth) * 100)
        return acc


    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Förändring")
        plt.title("Dålighetsindikator")
        plt.show()

