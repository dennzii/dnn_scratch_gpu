import os
import cupy as cp


class dnn_model:

    def __init__(self,layer_dims,lr=10e-4):
        
        self.lr = lr
        self.layer_dims = layer_dims
        self.parameters = self.init_parameters(layer_dims)
        self.LOG_EPSILON = 10e-10

    def init_parameters(self,layer_dims):
        
        parameters = {} 
        L = len(layer_dims)

        for l in range(1,L):#ilkten başlar çünkü ilk layer input
            parameters['W'+str(l)] = cp.random.randn(layer_dims[l], layer_dims[l-1]) * cp.sqrt(2/layer_dims[l-1])
            parameters['b'+str(l)] = cp.zeros((layer_dims[l],1))

        return parameters
    
    #Aktivasyon fonksiyonları
    def sigmoid(self,Z):

        A = 1 / (1+cp.exp(-Z))
    
        activation_cache = Z
        return A, activation_cache

    def relu(self,Z):
        A = cp.maximum(0,Z)
        activation_cache = A

        return A, activation_cache

    def forward_linear(self,A_prev,W,b):
        Z = cp.dot(W,A_prev) + b
        linear_cache = (A_prev,W,b)

        return Z, linear_cache
    
    def forward_activation(self,A_prev,W,b,activation):
        
        Z, linear_cache = self.forward_linear(A_prev,W,b)

        if activation == 'sigmoid':
            A, activation_cache = self.sigmoid(Z)
        elif activation == 'relu':
            A, activation_cache = self.relu(Z)

        return A, (linear_cache,activation_cache)

    def forward_prop(self,X):

        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1,L):#Son katman dışında olan katmanlar
            A_prev = A

            W = self.parameters['W'+str(l)]
            b = self.parameters['b'+str(l)]

            A, cache = self.forward_activation(A_prev,W,b,activation='relu')

            caches.append(cache)
        #Son katmanın forward prop'u
        WL = self.parameters['W'+str(L)]
        bL = self.parameters['b'+str(L)]

        AL, cache = self.forward_activation(A,WL,bL,activation='sigmoid')

        caches.append(cache)#katman sayısı kadar cachemiz olacak katmanın cachesi için caches[l-1] yapmak lazım

        return AL,caches
    
    def compute_loss(self,Y,AL):#
        AL = cp.clip(AL,self.LOG_EPSILON,1-self.LOG_EPSILON)# 0 logaritma olmasın diye clip yapıyoruz
        loss = -cp.mean(Y * cp.log(AL) + (1 - Y) * cp.log(1 - AL))


        loss = cp.squeeze(loss)

        return loss
    
    def sigmoid_derivative(self,Z):
        A,_ = self.sigmoid(Z)

        der = A*(1-A)

        return der
    
    def relu_derivative(self,Z):
        der = cp.where(Z>0,1,0)

        return der
    
    #eğer dZ'yi bilirsek dA[l-1] db[l] ve dW[l]'yi hesaplayabiliriz
    def linear_backward(self,dZ,cache):
        
        A_prev, W, b = cache
        m = A_prev.shape[1]  # batch size
        dW = cp.dot(dZ, A_prev.T) / m            # shape: (n_l, n_l-1)
        db = cp.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = cp.dot(W.T,dZ)

        return dA_prev, dW, db 
    
    def linear_activation_backward(self,dA,cache,activation):
        linear_cache, activation_cache = cache
        Z = activation_cache
        

        if activation == 'sigmoid':
            dZ = dA*self.sigmoid_derivative(Z)
        elif activation == 'relu':
            dZ = dA*self.relu_derivative(Z)
        
        dA_prev, dW, db = self.linear_backward(dZ,linear_cache)

        return dA_prev,dW,db
    

    def model_backprop(self,AL,Y,caches):
        
        grads = {}
        L = len(self.parameters)//2
        AL = cp.clip(AL,self.LOG_EPSILON,1-self.LOG_EPSILON)# logaritmanın içi 0 olmasına karşın bir önlem

        #output layerınının türevi alınarak başlanır.
        dAL = - (cp.divide(Y, AL) - cp.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,
                                                                                                      current_cache,"sigmoid")
        dA = grads["dA" + str(L - 1)]   
        for l in reversed(range(1,L)):
            current_cache = caches[l-1]  # dikkat: caches 0-indexli
            dA, dW, db = self.linear_activation_backward(dA, current_cache, "relu")
            grads['dA' + str(l - 1)] = dA
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

        return grads
    
    def update_parameters(self,grads):
        L = len(self.parameters)//2
        for l in range(1,L+1):
            self.parameters['W'+str(l)] = self.parameters['W'+str(l)] - self.lr * grads['dW'+str(l)]
            self.parameters['b'+str(l)] = self.parameters['b'+str(l)] - self.lr * grads['db'+str(l)]


    def get_mini_batches(self,X,Y,batch_size = 32):
        mini_batches = []

        m = X.shape[1]

        permutation = list(cp.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        num_complete_mini_batches = shuffled_X.shape[1] // batch_size

        for i in range(num_complete_mini_batches):
            batch_X = shuffled_X[:,i * batch_size:(i+1)*batch_size]
            batch_Y = shuffled_Y[:,i* batch_size: (i+1)*batch_size]

            mini_batch = (batch_X,batch_Y)

            mini_batches.append(mini_batch)

        return mini_batches
    

    def learning_rate_decay(self,epoch,initial_learning_rate,decay_rate):
        return initial_learning_rate / (1 + decay_rate * epoch)
    
    def train(self,X,Y,epochs,batch_size):
        costs = []
        for epoch in range(epochs):
            self.lr = self.learning_rate_decay(epoch,self.lr,decay_rate=0.0001)
            print(f"LR:{self.lr}")
            mini_batches = self.get_mini_batches(X,Y,batch_size)
            print(f"Epoch:{epoch+1}")
            cost = 0
            for batch in mini_batches:

                batch_X,batch_Y = batch

                AL,cache = self.forward_prop(batch_X)

                grads = self.model_backprop(AL,batch_Y,cache)

                self.update_parameters(grads)

                batch_cost = self.compute_loss(batch_Y,AL)
                cost += batch_cost
                
            avg_cost = cost / len(mini_batches)
            costs.append(avg_cost)
            print(f"Loss:{avg_cost}")
        return costs
    
    def predict(self,X,threshold=0.5):
        A,_ = self.forward_prop(X)
        return (A > threshold)
    
