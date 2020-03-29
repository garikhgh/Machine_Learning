import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


def print_result(y, pred_, cost):
	fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
	sns.scatterplot(range(len(y)), y, color='red', ax=ax[0])
	sns.scatterplot(range(len(pred_)), pred_, ax=ax[0])
	sns.lineplot(range(len(cost)), cost, ax=ax[1])
	ax[0].legend(['Actual Data ', 'Prediction '])
	plt.show()




# # f - house prices
# f = x1*w1 + x2*w2 + x3*w3 + x4*w4 + epsilon
# # where x1 - floor number, x2 - square meter, x3 - the district with 5 distinct values just for the example afterwards let it be the distance from the center, x4 - old/new
# # w1,w2,w3 and w4 are the weights that should be calibrated
w1 = 0.8
w2 = 2.2
w3 = 1.2
w4 = 0.9

weights = pd.Series([1.8, 2.2, 1.2, 1.9])
# # suppose we have observed some values of f represented by x1_,x2_,x3_,x4_ and epsilon_
# # initialize the values for the features and the error
# # ------------ your code goes here
size = 1000
x1_ = np.random.randint(low = 1, high = 5, size = size)	
x2_ = np.random.randint(low = 15, high = 60, size = size)
x3_ = np.random.randint(low = 1, high = 5, size = size)
x4_ = np.random.randint(low = 1, high = 2, size = size)
epsilon_ = np.random.normal(loc = 0, scale = 1, size = size)

# # end of the code ------------

# # get the sample dataset as a pandas dataframe using this values
# ------------ your code goes here
#//////////////////////////////////////////////////////////////////////////////////////
house_dict = {'floor_number':x1_, 'square meter':x2_,'distance': x3_, 'old/new': x4_}

house = pd.DataFrame(house_dict)

house['target'] = np.dot(house,weights) + epsilon_
y = house.pop('target')

#/////////////////////////////////////////////////////////////////////////////////////
#Normal Equation
new_weights = np.linalg.inv(house.T.dot(house)).dot(house.T).dot(y)

new_house = house.copy(deep = True)
new_house['pred'] = np.dot(new_house, new_weights)

new_house['target'] = y 
print("Accuracy is :",r2_score(new_house['target'], new_house['pred']))
print(new_weights,'\n', weights)
#/////////////////////////////////////////////////////////////////////////////////////
# end of the code ------------

# g - is the function that we want to learn from our sample that has a general structure and loss function
# ------------------------------------- specify the general structure
# the exact type of true function


# g = x1_*w1_ + x2_*w2_ + x3_*w3_ + x4_*w4_  # model 1
# # oversimplify the model
# g = x1_*w1_ + x3_*w3_ + x4_*w4_ # model 2
# g = x1_*w1_ + x4_*w4_ # model 3
# # overcomplicate the model
# x1_sq = x1**2
# x2_sq = x2**2
# g = x1_*w1_ + x2_*w2_ + x3_*w3_ + x4_*w4_ + x1_sq*w5_ # model 4
# g = x1_*w1_ + x2_*w2_ + x3_*w3_ + x4_*w4_ + x1_sq*w5_ + x2_sq_*w6_ # model 5

# specify the loss function - rmse
# ------------ your code goes here
cols = ['full_sq', 'life_sq', 'floor', 'big_church_count_5000', 'church_count_5000',
		'leisure_count_5000', 'sport_count_5000', 'market_count_500', 'price_doc']

path = 'C:\\Users\\user\\applied statistics and data scince\\ML\\data\\Sber_b_house\\train.csv'
df = pd.read_csv(path, usecols=cols, nrows=1000)

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns = cols)


y = df.pop('price_doc')
X = df.copy()


# exit()
#///////////////////////////////////////////////////////////////////////////////////////////////
eta = 0.0000005
n_epochs = 500
m = size
stop_ = 1e+3

cost = []
def gradient_descent(X, y, eta, n_epochs, m, stop_, cost):

	initial_weights = np.zeros(X.shape[1]) - 200
	print(initial_weights)
	for epoch in range(n_epochs):

		gradients = 2/m * X.T.dot(X.dot(initial_weights) - y)
		initial_weights = initial_weights - eta * gradients
		
		errors = y - X.dot(initial_weights)
		loss = np.sqrt((errors**2).sum() / 2*m) 
		cost.append(loss)
		
		if loss < stop_:
			exit()
			return 	initial_weights, cost, epoch
			
	return 	initial_weights, cost, epoch

initial_weights_, cost, epoch = gradient_descent(X, y, eta, n_epochs, m, stop_, cost)

print("Here is the output of the gradients\n ",initial_weights_) 
print('Iteration was stoped at :', epoch)


pred_ = X.dot(initial_weights_)


print("Accuracy is :",r2_score(y, pred_))

print_result(y, pred_,cost)


#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Stochastick Gradient Descent
path = 'C:\\Users\\user\\applied statistics and data scince\\ML\\data\\Sber_b_house\\train.csv'
df = pd.read_csv(path, usecols=cols, nrows=1000)

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns = cols)


y = df.pop('price_doc')
X = df.copy()


n_epochs = 100
t0, t1 = 5, 5000000000



def learning_schedule(t):
	return t0/(t + t1)
def mse(x,y, theta):
	xw = x.dot(theta)
	return np.sqrt(((xw - y)**2).sum() / (2*len(y)))



def stochastic_gradient_descent(X, y, n_epochs):
	theta = np.random.randn(X.shape[1])

	loss = []
	m = 200
	eta = 0.00000000001
	for epoch in range(n_epochs):
		for i in range(m):
			random_index = np.random.randint(m)
			xi = X[random_index:random_index+1]
			yi = y[random_index:random_index+1]
			gradients = 2 * xi.T.dot(xi.dot(theta) - yi)

			eta = learning_schedule(epoch * m + i)
			theta = theta - eta * gradients

		cost = mse(X, y, theta)
		loss.append(cost)
		print("Epoch--", epoch, 'ETA--', eta, 'RMSE', cost)

	return theta, loss

new_theta, loss = stochastic_gradient_descent(X, y, n_epochs)
pred__ = X.dot(initial_weights_)


print("Accuracy is :",r2_score(y, pred__))

print_result(y, pred__, loss)


# end of the code ------------

# define the gradient descent optimizator for multivariate linear regression problem
# ------------ your code goes here



# end of the code ------------

# train the models, for each of them you should get the model's rmse and parameter values (compare them with the true parameter values)
# ------------ your code goes here
# model 1
# model 2
# model 3
# model 4
# model 5
# end of the code ------------

# we will discuss the final, model validation part, during the upcoming workshops