# Function to define the weights and bias
def init_params():
    W1=np.random.rand(10,784)-0.5 # Generates a 10x784 array of random numbers between -0.5 and 0.5
    b1=np.random.rand(10,1)-0.5 
    W2=np.random.rand(10,10)-0.5
    b2=np.random.rand(10,1)-0.5
    return W1,b1,W2,b2

# Define Rectified Linear Unit
def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1,b1,W2,b2,X):
    Z1=W1.dot(X)+b1 # Value for first layer
    A1=ReLU(Z1) # Activation function for first layer
    
    Z2=W2.dot(A1)+b2 # Value for second layer
    A2=softmax(Z2) # Activation function for second layer

    return Z1,A1,Z2,A2

# Convert output values as a single-column matrix with 10 rows, each corresponding to the possible digits 0-9
def one_hot(Y):
    one_hot_Y=np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size),Y]=1 # Match where the 1s are at
    one_hot_Y=one_hot_Y.T # Transpose to get each column as an example
    return one_hot_Y

# Derivative of ReLU function
def deriv_ReLU(Z):
    return Z > 0

# Backpropagation to compute gradients of loss with respect to the weights and biases 
def back_prop(Z1,A1,Z2,A2,W2,X,Y):
    m=len(Y)
    one_hot_Y=one_hot(Y)
    dZ2=A2-one_hot_Y
    dW2=1 / m * dZ2.dot(A1.T) # Divide by m to find the average
    db2=1 / m * np.sum(dZ2)
    dZ1=W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1=1 / m * dZ1.dot(X.T)
    db1=1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Use gradient descent to optimise the values of weights and biases to improve predictions
def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha): # alpha is user-defined learning rate
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2,0) # Finds the class with the highest number

def get_accuracy(predictions, Y):
    return np.sum(predictions==Y)/Y.size

# Find the derivative and slowly move towards the point of minimum gradient which means cost is a minimum
def gradient_descent(X,Y, iterations,alpha):
    W1,b1,W2,b2=init_params()
    accuracy=[]
    for i in range(iterations):
        Z1,A1,Z2,A2=forward_prop(W1,b1,W2,b2,X)
        dW1, db1, dW2, db2=back_prop(Z1,A1,Z2,A2,W2,X,Y)
        W1,b1,W2,b2=update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        accuracy.append(get_accuracy(get_predictions(A2), Y))
        if i%10 ==0: # Print accuracy for every 10 iterations
            print("Iteration: ", i)
            print("Accuracy: ", accuracy[-1])
    predictions=get_predictions(A2)
    return W1,b1,W2,b2,accuracy, predictions
