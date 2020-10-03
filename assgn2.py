import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import functools

# 1 polynomialFeatures(X, degree) function

def polynomialFeatures(x,degree = 2):
    x_t = x.T
    features = np.ones(len(x))
    for degree in range(1, degree + 1):
        for items in combinations_with_replacement(x_t, degree):
            features = np.vstack((features,functools.reduce(lambda x, y: x * y, items)))
    return features.T



# 2 mse(Y_true, Y_pred) function

def mse(Y_true, Y_pred):
    E = np.array(Y_true).reshape(-1,1) - np.array(Y_pred).reshape(-1,1)
    mse = 1/np.array(Y_true).shape[0] * (E.T.dot(E))
    return mse[(0,0)]




# 3 learning_curve function

def k_partition(data,kfold):
    return np.array_split(data,kfold)

def kfold_error(model, X, Y, k_fold, learning_rate = 0.01, epochs = 1000, 
                tol = None, regularizer = None, lambd = 0.0, **kwargs):
    if(Y.shape == (Y.shape[0],)):
        Y = np.expand_dims(Y,axis=1)
    dataset = np.concatenate([X,Y],axis=1)

    k_part = k_partition(dataset, k_fold)   # using the function_1 k_partition
    
    error_training = []
    error_validation = []

    for idx,val in enumerate(k_part):
        validation_Y = val[:,-1]
        validation_X = val[:,:-1]
        train = np.concatenate(np.delete(k_part,idx,0))
        train_Y = train[:,-1]
        train_X = train[:,:-1]          
    
        # with sklearn Linearregression to test the entire function
        # replace it when our modeling function is done.
        #reg = model().fit(train_X, train_Y)
        #pr_train_Y = reg.predict(train_X)
        #pr_validation_Y = reg.predict(validation_X)
        #mse_train_Y = mse(train_Y, pr_train_Y)
        #mse_validation_Y = mse(validation_Y, pr_validation_Y)
    
#         # using our modeling function
        model = Linear_Regression()
        model.fit(train_X,train_Y,learning_rate=learning_rate,epochs=epochs,regularizer=regularizer,lambd= lambd, tol=tol)
        pr_train_Y = model.predict(train_X)
        pr_validation_Y = model.predict(validation_X)
        mse_train_Y = mse(train_Y, pr_train_Y)
        mse_validation_Y = mse(validation_Y, pr_validation_Y)

        error_training.append(mse_train_Y)
        error_validation.append(mse_validation_Y)

    # return the average mse for the training and the validation fold.
    return np.array(error_training).mean(), np.array(error_validation).mean()


def learning_curve(model, X, Y, cv, train_size = 1, learning_rate = 0.01, 
                   epochs = 1000, tol = None, regularizer = None, lambd = 0.0, **kwargs):
    
    
    if type(train_size) == int and train_size > 1: 
        train_size_abs = train_size
    elif train_size > 0 and train_size <= 1:
        train_size_abs = int(train_size * X.shape[0])
    else:
        print(f'unaceptable train_size of {train_size}')
        
    if X.shape[0] % train_size_abs != 0:
        t = X.shape[0] // train_size_abs + 1
    else:
        t = X.shape[0] // train_size_abs
    
    t0 = 1
    num_samples_list = []
    rmse_training_list = []
    rmse_validation_list = []

    while t0 <= t:
        i = t0 * train_size_abs
        if i >= X.shape[0]: i = X.shape[0]
        index = np.arange(i)
        X_split = (X[index])
        Y_split = (Y[index])
        
        # using the function_2 to calculate the mse, and then rmse using k-fold based on varying training samples.
        rmse_tr, rmse_va = np.sqrt(kfold_error(model, X_split, Y_split, cv, learning_rate = learning_rate, 
                                               epochs = epochs, tol = tol, regularizer = regularizer, lambd = lambd))
        num_samples_list.append(i)
        rmse_training_list.append(rmse_tr)
        rmse_validation_list.append(rmse_va)
        t0 += 1

    result = {'num_samples':pd.Series(num_samples_list), 
              'train_scores':pd.Series(rmse_training_list), 'val_score':pd.Series(rmse_validation_list)}
    learning_curve_elements = pd.DataFrame(result)
    
    return np.array(learning_curve_elements["train_scores"]), \
            np.array(learning_curve_elements["val_score"]), np.array(learning_curve_elements["num_samples"]), \
            learning_curve_elements



# 4 plot_polynomial_model_complexity function

def plot_polynomial_model_complexity(model, X, Y, cv, maxPolynomialDegree, 
                                     learning_rate=0.01, epochs=1000, tol=None, regularizer=None, lambd=0.0, **kwargs):
    
    poly_list = []
    rmse_training_list = []
    rmse_validation_list = []
    
    for i in range(1, maxPolynomialDegree+1):
        
        # Here should be replaced with our #1 function polynomialFeatures(X, degree)
        X_poly = polynomialFeatures(X, i)
        Y_poly = polynomialFeatures(X, i)
        print(f'X shape = {X_poly.shape} #### learning_rate = {learning_rate}   ##### Degree = {i} ## epoch:{epochs}')
        
        rmse_tr, rmse_va = np.sqrt(kfold_error(model, X_poly, Y_poly, cv, learning_rate = learning_rate, 
                                               epochs = epochs, tol = tol, regularizer = regularizer, lambd = lambd))
        learning_rate *= 0.1
        poly_list.append(i)
        rmse_training_list.append(rmse_tr)
        rmse_validation_list.append(rmse_va)
        
    # plot the RMSE using the list above.
    plt.figure(figsize = (10, 10))
    
    # linewidth and fontsize
    lw = 2
    fontsize = 20
    
    # plot curves
    plt.plot(poly_list, rmse_training_list, 'o-', color='green', lw = lw, label = "Training set") 
    plt.plot(poly_list, rmse_validation_list, 'o-', color='red', lw = lw, label = "Validation set") 
            
    # add title, xlabel, ylabel, and legend. 
    plt.title('RMSE curve', fontsize = fontsize)
    plt.xlabel('Polynomial Degree', fontsize = fontsize)
    plt.ylabel('RMSE', fontsize = fontsize)
    plt.legend(loc="best", fontsize = fontsize)
    plt.xticks(np.arange(1, i+1, 1))
    
    plt.show()
    

class Linear_Regression:
    def __init__(self):
        pass   
    
    def fit(self, X, Y, learning_rate=0.01, epochs=1000, tol=None, regularizer=None,lambd=0.0,**kwargs):
        epch_counter = 0
        m,n = X.shape
        self.theta = np.zeros(n) # Initialize theta to be 0 of size n
        if(tol != None):
            self.error = -tol
        else:
            self.error = -1e5
        while epch_counter < epochs:
            epch_counter += 1 #count the epochs
            theta = self.theta
            error = self.error
            
            a = np.dot(X.T,(X @ theta - Y)) # gradient of loss wrt parameter  
            if(regularizer == 'l2'):
                self.theta = theta - ((a * learning_rate) / m) - (learning_rate * lambd*theta)/m
            elif(regularizer == 'l1'):
                self.theta = theta - ((a * learning_rate) / m) - (learning_rate * lambd* np.sign(theta))/m
            else:
                self.theta = theta - ((a * learning_rate) / m)

            self.error = mse(Y,X @ self.theta)
            if(tol != None):
                
                if np.abs(self.error - error) < tol:
                    break
        #print(f'Epochs needed: {epch_counter}')
                                  
    def predict(self,X):
        return X @ self.theta

################# Add learning rate scheduler (Geron chapter 4)
t0,t1 = 5 , 50

def learning_schedule(t):
    return t0 / (t - t1)

class SGD:
    def __init__(self):
        pass   
    
    def fit(self, X, Y, learning_rate=0.01, epochs=1000, tol=None, regularizer=None,lambd=0.0,**kwargs):
        epch_counter = 0
        m,n = X.shape
        self.theta = np.zeros(n) # Initialize theta to be 0 of size n
        self.error = -tol
        while epch_counter < epochs:
            epch_counter += 1 #count the epochs
            for i in range(m):
                learning_rate = learning_schedule(epochs * m + i)
                theta = self.theta
                error = self.error
                a = np.dot(X[i].T , (X[i] @ self.theta - Y[i]))
                if(regularizer == 'l2'):
                    self.theta = theta - (a * learning_rate) - (learning_rate * lambd*theta)           
                elif(regularizer == 'l1'):
                    self.theta = theta - (a * learning_rate) - (learning_rate * lambd* np.sign(theta))
                else:
                    self.theta = theta - (a * learning_rate)                
                self.error = mse(Y,X @ self.theta)
                
                if np.abs(self.error - error) < tol:
                    break
        #print(f'Epochs needed: {epch_counter}')
                                  
    def predict(self,X):
        return X @ self.theta

def sFold(folds,data,labels,model,error_fuction,**model_args):
#def sFold(folds,data,labels,model,error_fuction,**model_args):
    if(labels.shape == (labels.shape[0],)):
        labels = np.expand_dims(labels,axis=1)
    dataset = np.concatenate([data,labels],axis=1)
    s_part = s_partition(dataset,folds)
    pred_y = []
    true_y = []
    for idx,val in enumerate(s_part):
        test_y = val[:,-1]
        #test_y = np.expand_dims(test_y, axis=1)
        test = val[:,:-1]
        train = np.concatenate(np.delete(s_part,idx,0))
        label = train[:,-1]
        train = train[:,:-1]        
        model.fit(train,label,**model_args)       
        pred = model.predict(test)
        pred_y.append(pred)
        true_y.append(test_y)
    pred_y = np.concatenate(pred_y)
    true_y = np.concatenate(true_y)

    avg_error = error_fuction(pred_y,true_y).mean()   
    result = {'Expected labels':true_y, 'Predicted labels': pred_y,'Average error':avg_error }
    return result


#helper
def s_partition(x,s):
    return np.array_split(x,3)


