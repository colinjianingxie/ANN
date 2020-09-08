from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import pickle
import random
from scipy.special import expit	# Used safe sigmoid calculation library
import matplotlib.pyplot as plt


class NeuralNetwork:

	def __init__(self, nodes_array, activation='sigmoid'):

		self.layers = len(nodes_array)
		self.nodes_array = nodes_array
		self.activation = activation

		self.weights = []
		self.biases = []
		self.losses = []


	def fit(self, X, y, alpha, t, lambda_val):

		for iteration in range(t):

			num_rows = np.shape(X)[0]


			# FORWARD PROPAGATION
			total_loss = 0
			for i in range(len(X)):

				input_row = X[i]
				a_arr = [input_row]			# add input into the first layer

				a, a_arr = self.feed_forward(input_row, a_arr)

				# First output error
				total_loss += np.sum(self.calc_cross(a_arr[-1], y[i], lambda_val))
				deriv_last_activation_layer = a_arr[-1] * (1 - a_arr[-1]) # softmax * (1 - softmax)
				cost_gradient = self.calc_cross_entropy_gradient(a_arr[-1], y[i], lambda_val)
				back_delta = cost_gradient * deriv_last_activation_layer
	
				curr_delta = back_delta # 3,1

				for last_layer in range(self.layers-1, 0, -1):
					self.biases[last_layer-1] -= alpha * curr_delta
					update_weights = np.dot(a_arr[last_layer-1].reshape(-1,1),curr_delta.reshape(-1,1).T)
					self.weights[last_layer-1] -= alpha * update_weights
					derive_sigmoid = a_arr[last_layer-1] * (1-a_arr[last_layer-1])
					weight_delta = np.dot(self.weights[last_layer-1],curr_delta)

					curr_delta = np.multiply(weight_delta, derive_sigmoid)
					
			self.losses.append(total_loss/num_rows)
			print("Loss for iteration %d: %.7f" % (iteration, total_loss/num_rows))


	def feed_forward(self, row, a_arr):


		#a_arr.append(row)
		for layer in range(self.layers-2): #2
			row = np.dot(self.weights[layer].T, row) + self.biases[layer]
			row = sigmoid(row)
			a_arr.append(row)

		# do softmax for last layer
		row = np.dot(self.weights[-1].T, row) + self.biases[-1]
		row = softmax(row)

		a_arr.append(row)
		
		return (row, a_arr)
		

	def feed_forward_test(self, row):

		for layer in range(self.layers-2): #2
			row = np.dot(self.weights[layer].T, row) + self.biases[layer]
			row = sigmoid(row)

		row = np.dot(self.weights[-1].T, row) + self.biases[-1]
		row = softmax(row)

		return row


	def predict(self, X, y):
		num_rows = np.shape(X)[0]

		accuracy_count = 0
		for i in range(num_rows):

			
			test_row = X[i]
		
			test_result = np.argmax(y[i])
			prediction = np.argmax(self.feed_forward_test(test_row))
			if(prediction == test_result):
				accuracy_count += 1
			
		return round(100.0 * accuracy_count/num_rows, 4)



	def set_weights(self, weights):
		'''
		weights = [w1, b1, w2, b2, ...]
		'''

		for i in range(len(weights)):
			if i % 2 == 0:
				self.weights.append(weights[i])
			else:
				self.biases.append(weights[i])


	def get_weights(self):
		print("\nMY Weights are: ")
		for weight in self.weights:
			print(weight)

		print("\nMY Biases are: ")
		for bias in self.biases:
			print(bias)
		print("---")

	def calc_cross(self, predictions, labels, lambda_val):
		return -(labels * np.log(predictions))

	# Use this for regularization
	def calc_cross_entropy(self, predictions, labels, lambda_val):
		return -(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)) + 0.5 * lambda_val * (self.weights[-1]**2).sum()

	def calc_cross_entropy_gradient(self, predictions, labels, lambda_val):
		
		return (-labels/predictions + (1-labels)/(1-predictions))
		#return (-labels/predictions + (1-labels)/(1-predictions)) + lambda_val * (self.weights[-1]).sum() # for the regularization

def softmax(x):
	
	#K = 10
	#return np.exp(x + np.log(K))/(np.sum(np.exp(x + np.log(K))))
	return np.exp(x)/(np.sum(np.exp(x)))

def get_random_weight(n_in, n_out):
	low = -np.sqrt(6/(n_in + n_out))
	high = np.sqrt(6/(n_in + n_out))
	return random.uniform(low, high)

def get_xavier_weights_bias(n_in, n_out):

	weight_arr = np.zeros((n_in, n_out))
	bias_arr = np.zeros(n_out)
	for i in range(n_in):
		for j in range(n_out):
			weight_arr[i][j] = get_random_weight(n_in, n_out)


	return weight_arr, bias_arr

def derivative_softmax(x):

	return softmax(x) * (1 - softmax(x))
	#return (np.exp(x)/np.sum(np.exp(x)) - 1 )

def sigmoid(x):
	'''Note I used SKLearns Sigmoid, since it was numerically safer to calculate than the one I have commented'''
	return expit(x)
	#return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))




if __name__ == "__main__":


	  
	'''
		GETTING MNIST
	'''

	df = pd.read_csv('train.csv')

	train_df = df.copy()	# make a copy of the train_df


	labels = train_df['label']
	del train_df['label']
	train_features = train_df.copy()
	dummy_labels = pd.get_dummies(labels, dtype=int).values
	train_features = train_features.values

	'''

	NOTE: num_train_values + num_test_values < number of rows in train.csv

	I did not shuffle the dataset, I wasn't sure what to submit...
	However, I could have shuffled for the purpose of true SGD

	Edit the values below: num_train_values, num_test_values

	Training set will be from rows [0...num_train_values]
	Testing set will be from rows [num_train_values...num_train_values + num_test_values]

	'''
	num_train_values = 30000		
	num_test_values = 3000

	training = train_features[0:num_train_values]
	test_features = train_features[num_train_values: num_train_values + num_test_values]

	train_labels = dummy_labels[0: num_train_values]
	test_labels = dummy_labels[num_train_values: num_train_values + num_test_values]



	'''
		CREATING NEURAL NETOWRK
	'''


	# manually create the hidden layer

	num_start_nodes = 784
	num_hidden_nodes = 500

	num_end_nodes = 10

	w1, b1 = get_xavier_weights_bias(num_start_nodes, num_hidden_nodes)	
	w2, b2 = get_xavier_weights_bias(num_hidden_nodes, num_end_nodes)	

	w = [w1,b1,w2,b2]	# Weight array for setting the NN

	node_arr = np.array([num_start_nodes, num_hidden_nodes, num_end_nodes])


	NN = NeuralNetwork(node_arr, 'sigmoid')
	NN.set_weights(w)



	alpha = 0.001
	lambda_val = 0.001
	t = 10

	
	print("About to starting fitting for...%d epochs " % (t))
	NN.fit(training, train_labels, alpha, t, lambda_val)
	print("Finished training!")
	print("Now testing for accuracy on the test set of dataset...")
	acc = NN.predict(test_features, test_labels)
	print("Achieved accuracy of: %.2f percent" % (acc))

	x_arr = np.arange(t)
	loss_arr = np.array(NN.losses)
	title_name = "C.E Loss [" + str(t) + " epochs], " + str(num_hidden_nodes) + " nodes in 1 hidden layer\n (" + str(acc) + " percent acc)"


	plt.plot(x_arr, loss_arr, '.b-')
	plt.xlabel('Number of epochs')
	plt.ylabel('Cross Entropy Loss')
	plt.title(title_name)
	plt.savefig('final_losses', format='pdf')
	plt.show()






