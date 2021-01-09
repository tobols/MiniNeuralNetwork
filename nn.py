import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ReadData import heart_df
from NeuralNet import NeuralNet


#convert input to numpy arrays
X = heart_df.drop(columns=['heart_disease'])
y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)


#standardize the dataset
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)


nn = NeuralNet(layers=[13,8,1], learning_rate=0.001, iterations=500) # create the NN model
nn.train(Xtrain, ytrain) #train the model

nn.plot_loss()

train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

print("Accuracy on training dataset is: {}%".format(nn.accuracy(ytrain, train_pred)))
print("Accuracy on testing dataset is: {}%".format(nn.accuracy(ytest, test_pred)))
