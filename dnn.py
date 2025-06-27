import os
import cupy as cp
import pickle
import cv2
import numpy as np

class dnn_model:

    def __init__(self,layer_dims,lr=10e-4,lambd=0.01):
        
        self.lr = lr # learning rate 
        self.layer_dims = layer_dims #katman boyutları
        self.parameters = self.init_parameters(layer_dims)
        self.LOG_EPSILON = 10e-10 # loss fonksiyonunun içi 0 olmasın diye kırpma işlemi için gereken çok küçük değer
        self.lambd = lambd # L2 regularizasyon katsayısı

    def init_parameters(self,layer_dims):
        """
        Parametrelerin HE başlatması
        """
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
        cross_entropy = -cp.mean(Y * cp.log(AL) + (1 - Y) * cp.log(1 - AL))

        if self.lambd > 0: # L2 regularizasyonu eklenmişse loss'a L2 kısmı eklenir.
            L2_sum = sum(cp.sum(cp.square(self.parameters['W'+str(l)])) for l in range(1, len(self.layer_dims)))
            L2_term = (self.lambd / (2 * Y.shape[1])) * L2_sum
            loss = cross_entropy + L2_term
        else:
            loss = cross_entropy

        loss = cp.squeeze(cross_entropy)

        return loss
    
    #Aktivasyon fonksiyonlarının türevleri
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
        dW = (cp.dot(dZ, A_prev.T) / m) + (self.lambd / m) * W          # shape: (n_l, n_l-1)
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
    
    #Parametrelerin güncellenmesi
    def update_parameters(self,grads):
        L = len(self.parameters)//2
        for l in range(1,L+1):
            self.parameters['W'+str(l)] = self.parameters['W'+str(l)] - self.lr * grads['dW'+str(l)]
            self.parameters['b'+str(l)] = self.parameters['b'+str(l)] - self.lr * grads['db'+str(l)]

    # mini-batchleri organize eden fonksiyon
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
    
    def get_val_acc(self,x_val,y_val):
        preds = self.predict(x_val)
        acc = cp.sum(preds==y_val) /x_val.shape[1]

        return acc


    def train(self,X,Y,X_val,Y_val,epochs,batch_size):
        logs = []
        best_parameters = None
        best_parameters_loss = 9999999
        for epoch in range(epochs):
            self.lr = self.learning_rate_decay(epoch,self.lr,decay_rate=0.0003)
            mini_batches = self.get_mini_batches(X,Y,batch_size)
            print(f"Epoch:{epoch+1}")
            loss = 0
            for batch in mini_batches:

                batch_X,batch_Y = batch

                AL,cache = self.forward_prop(batch_X)

                grads = self.model_backprop(AL,batch_Y,cache)

                self.update_parameters(grads)

                batch_loss = self.compute_loss(batch_Y,AL)
                loss += batch_loss
                
            avg_loss = loss / len(mini_batches)
            val_acc = self.get_val_acc(X_val,Y_val)
            val_loss = self.compute_loss(Y_val,self.predict(X_val))

           
            log = (avg_loss,val_loss,val_acc)
            if val_loss < best_parameters_loss:
                best_parameters = self.parameters
                best_parameters_log = log
                best_parameters_loss = val_loss

            logs.append(log)
            print(f"Loss:{avg_loss} Val loss:{val_loss} Val acc:{val_acc}")
            
        print("Saving Model..")
        self.save_model(best_parameters)
        return logs,best_parameters_log,best_parameters
    
    def predict(self,X,threshold=0.5):
        A,_ = self.forward_prop(X)
        return (A > threshold)
    
    def predict_no_threshold(self,X):
        #NMS algoritması için gereken eşik değersiz prediction fonksiyonu
        A,_ = self.forward_prop(X)
        return A
    
    def save_model(self,parameters):
        """
        parametreleri ilk önce cupy'dan numpy'a çevirir ardından
        npz dosyası olarak kaydeder.
        """
        # 1) CuPy → NumPy
        cpu_params = { k: cp.asnumpy(v) for k, v in parameters.items() }
        # 2) sıkıştırılmış NPZ dosyası olarak kaydet
        np.savez_compressed(f"best.npz", **cpu_params)
        print(f"Model parameters saved to best.npz")

    def load_model(self, pth):
        """
        .npz dosyasını yükler ve içindekileri CuPy dizilerine dönüştürür.
        """
        # NPZ’yi oku 
        npz = np.load(pth)
        # her bir parametreyi CuPy’ye al
        self.parameters = {
            k: cp.asarray(npz[k], dtype=cp.float32)
            for k in npz.files
        }

        


    def sliding_window(self,image, window_size=(196, 196), step_size=32):

        window_width = int(window_size[0]) if isinstance(window_size[0], (int, float)) else int(window_size[0].item())
        window_height = int(window_size[1]) if isinstance(window_size[1], (int, float)) else int(window_size[1].item())
        
        for x in range(0, image.shape[1] - window_width, step_size):
            for y in range(0, image.shape[0] - window_height, step_size):
                yield (x, y, image[y:y + window_height, x:x + window_width])




    def object_detection_on_video(self, video_pth,
                              window_size=(164,164),
                              step_size=25,
                              detect_threshold=0.5):
        cap = cv2.VideoCapture(video_pth)

        # sliding_window fonksiyonu da parametre alacak şekilde değişmeli:
        # def sliding_window(self, image, window_size, step_size):

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # kareyi 640px genişlik yap, orantıyı koru
            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * new_w / w)
            frame = cv2.resize(frame, (new_w, new_h))

            scores = []
            boxes  = []
            win_w, win_h = window_size

            # 1) Kaydırmalı pencere
            for x, y, window in self.sliding_window(frame, window_size, step_size):
                # ağın istediği input boyutuna küçült, görüntüyü griye çevir, flatten & normalize
                small = cv2.resize(window, (64,64))
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                flat  = gray.flatten().astype(np.float32) / 255.0
                X     = cp.asarray(flat).reshape(-1,1)

                # tahmin & thresh kontrolü
                score = float(self.predict_no_threshold(X))
                if score < detect_threshold:
                    continue

                
                scores.append(score)
                
                boxes.append((x, y, x + win_w, y + win_h))

            # 5) NMS
            selected = self.non_max_suppression(boxes, scores)

            # 6) çizim
            for idx in selected:
                x1, y1, x2, y2 = boxes[idx]
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    def intersection_over_union(self,box1, box2):
        # kesişim kutusunun koordinatlarını hesapla
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        
        union_area = box1_area + box2_area - inter_area
        
        # iou değerini döndürür
        return inter_area / union_area if union_area != 0 else 0

    import cupy as cp

    def non_max_suppression(self, boxes, scores, threshold=0.3):
        """
        boxes: (x1,y1,x2,y2)
        """
        
        scores_cp = cp.asarray(scores, dtype=cp.float32)
        # azalan sırada indeksleri al
        indices = scores_cp.argsort()[::-1]  # cupy.ndarray

        selected_boxes = []
       
        while len(indices) > 0:
           
            current = int(indices[0])
            selected_boxes.append(current)

            remaining = []
            for idx in indices[1:]:
                idx = int(idx)
                if self.intersection_over_union(boxes[current], boxes[idx]) < threshold:
                    remaining.append(idx)
            # kalanları tekrar cupy.ndarray’e çevir
            if remaining:
                indices = cp.asarray(remaining, dtype=cp.int32)
            else:
                break

        return selected_boxes
