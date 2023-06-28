# Define Xavier initialization function
def xavier_init(n_in, n_out):
    var=1.0/(n_in + n_out) # Calculate variance
    stdev=np.sqrt(var) # Calculate standard deviation
    weights=np.random.normal(loc=0.0,scale=stdev, size=(n_in,n_out)) # Generate random values from a Gaussian distribution with a mean of 0 and S.D of stdev
    return weights

# Combine all functions into a single Dense Neural Network (DNN) class with Xavier Initialization
class DNN3:
    
    def __init__(self, epochs=100,lr=0.01):
        self.epochs=epochs
        self.lr=lr
        
         # Initalization parameters with Xavier Initialization
        self.params={
            'W1': xavier_init(128,784), # 128x784
            'b1': xavier_init(128,1), # 128x1
            'W2': xavier_init(64,128), # 64x128
            'b2': xavier_init(64,1), # 64x1
            'W3': xavier_init(10,64), # 10x64
            'b3': xavier_init(10,1) #10x1
        }
        
        self.change={}
        
    def ReLU(self,Z):
        return np.maximum(Z,0)

    def softmax(self,Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
    
    def deriv_ReLU(self,Z):
        return Z > 0
    
    def one_hot(self,Y):
        one_hot_Y=np.zeros((Y.size, Y.max()+1))
        one_hot_Y[np.arange(Y.size),Y]=1 # Match where the 1s are at
        one_hot_Y=one_hot_Y.T # Transpose to get each column as an example
        return one_hot_Y
    
    def forward_prop(self, X):
        params=self.params
        params['X']=X # 784x1, input layer
        
        params['Z1']=params['W1'].dot(params['X'])+params['b1']
        params['A1']=self.ReLU(params['Z1']) # First layer

        params['Z2']=params['W2'].dot(params['A1'])+params['b2']
        params['A2']=self.ReLU(params['Z2']) # Second layer

        params['Z3']=params['W3'].dot(params['A2'])+params['b3']
        params['A3']=self.softmax(params['Z3']) # Third layer
        
        return params['A3']

    def back_prop(self,Y,output):
        params=self.params
        params['Y']=self.one_hot(Y)
        
        m=len(Y)
        change=self.change

        dZ3=output-params['Y']
        change['W3']=1 / m * dZ3.dot(params['A2'].T)
        change['b3']=1 / m * np.sum(dZ3)
        
        dZ2=params['W3'].T.dot(dZ3) * self.deriv_ReLU(params['Z2'])
        change['W2']=1 / m * dZ2.dot(params['A1'].T)
        change['b2']=1 / m * np.sum(dZ2)
        
        dZ1=params['W2'].T.dot(dZ2) * self.deriv_ReLU(params['Z1'])
        change['W1']=1 / m * dZ1.dot(params['X'].T)
        change['b1']=1 / m * np.sum(dZ1)
        
        return change
    
    def update_params(self,change):
        params=self.params
        for key,val in change.items():
            params[key]-=self.lr*val
            
    def get_predictions(self,X):
        return np.argmax(X,0)
        
    def get_accuracy(self,predictions, Y):
        return np.sum(predictions==Y)/Y.size
     
    # continue with train function
    def train(self,X,Y):
        accuracy=[]
        change=self.change
        for i in range(1,self.epochs+1): # Repeat forward and backward propagation and updates weights and biases after each iteration
            output=self.forward_prop(X)
            change=self.back_prop(Y,output)
            self.update_params(self.change)
            accuracy.append(self.get_accuracy(self.get_predictions(output),Y))
            if i%50==0: 
                print("Iteration: ",i)
                print("Accuracy: ", accuracy[-1])
        predictions=self.get_predictions(output)
        return accuracy,predictions
    
    def predict(self,X):
        output=self.forward_prop(X)
        predictions=self.get_predictions(output)
        return predictions
        
    
