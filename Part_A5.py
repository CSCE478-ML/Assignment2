import itertools as it
import numpy as np
import functools as ft
import random as rand

class Linear_Regression:
    #steps to implement Batch Gradient Descent: Implementation. Regularization in this case: none but model evaluation requires all three
    #1. Choose the degree of polynomial (for polynomial regression), then add polynomial & bias terms in X 
    #2. Standardize the features
    #3. Choose the hyperparameters: η & λ learning rate= n, 
    #4. Initialize θ (all zeros)
    #5. Choose a stopping criterion by setting a tolerance parameter that stops iteration when the difference between
    # the two consecutive J(MSE)’s goes below the tolerance value. J is cost function. Loss/error function L
    #6. Iteratively apply the update rule until the stopping criterion is met. This is all to choose theta as to minimize J(theta). theta_hat is the one with the line above it.
    #m is size of training dataset, X.shape[0]
    def __init__(self):
        self=self

    def fit(self, X, Y, learning_rate=0.01, epochs=1000, tol=None, regularizer=None, lambd=0.0, **kwargs):
        stop=0
        #1. Choose the degree of polynomial (for polynomial regression), then add polynomial & bias terms in X 
        #Degree of polynomial : 1, this is linear, not polynomial. Bias terms is : in weight vector theta_hat,  
        # contains prameters for the model (one parameter for each feature and one for bias) theta hat is 1d column vector
        #print(X)
        m=X.shape[0]
        features= X.T


        #bias in terms of X
        #Bias is the ||theta_hat||
        bias=[[1]]
        #print(Y)
        theta_hat=np.concatenate((features,bias),axis=1)
        theta_hat=theta_hat.T
        #print(theta_hat)
        #2. Standardize the features
        X = (X - np.mean(X))/np.std(X)
        Y = (Y - np.mean(Y))/np.std(Y)
        #3. Choose the hyperparameters: η & λ
        #n is learning_rate, lambda is lambd
        #4. Initialize θ (all zeros)
        theta=np.zeros(m)
        #5. Choose a stopping criterion by setting a tolerance parameter that stops iteration when the difference between
        # the two consecutive J(MSE)’s goes below the tolerance value
        #Yes I am aware that steps 5 and 6 are to be in the same loop, I'm still figuring out the algorithm. 
        #Currently figuring out where the heck bias is derived from in terms of X and....
        if(tol == None):
            stop=epochs
        for i in range(epochs):
            #Calc J(MSE)
            if(regularizer!=None):
                error=1
                previous_error=0     
                if(regularizer=="l2"):
                    
                    J=(1/(2*m))*((Y-np.dot(X,theta))**2)+ ((lambd/(2*m))*bias)
                    #bias= sum(from j=1 to d of theta_j^2), where theta_j = theta_j - learning_rate*(Partial derivative of theta_j)*J(theta)
                    error=1
                if(regularizer=="l1"):
                     #difference between l2 and l1 is in the update function, where the regularizer actually makes a difference. That should
                     #be in this loop
                    J=(1/(2*m))*((Y-np.dot(X,theta))**2)+ ((lambd/(2*m))*bias)
                    error=1
                    previous_error=0                 

                if(tol !=None):
                    if(error> previous_error - tol):
                        stop=i
                        break
                    else:
                        stop=epochs
            else:
                i=epochs
                stop=epochs

        #6. Iteratively apply the update rule until the stopping criterion is met


        #regularized cost function:

        for i in range(stop):

            a= theta
            d=(np.dot(X[:,0],theta)-Y)[:,0]
            b= np.dot(X.T[0],d)        
            c=0  #third part is based on type of regularization.
            if regularizer=="l2":
                c= ((learning_rate*lambd*theta)/m)
            if regularizer=="l1": 
                c= ((learning_rate*lambd)/m)*sign(theta)
            theta = a - b - c
        print(theta)
        print(stop)

        self.theta=theta_hat

        return self
    def predict(self, X):
        #1D array of predictions for each row in X.
        #The 1D array should be designed as a column vector
        predictions=X
        return predictions


# In[ ]:




