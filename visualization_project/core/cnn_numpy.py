import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

# -----------------------------
# 하이퍼파라미터
# -----------------------------
IMG_H, IMG_W, IN_CHANNELS = 64, 64, 1
OUTPUT_SIZE = 111
LEARNING_RATE = 1e-3
EPOCHS = 30
BATCH_SIZE = 64

# -----------------------------
# Utils
# -----------------------------
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    if y_true.ndim>1: y_true=np.argmax(y_true,1)
    if y_pred.ndim>1: y_pred=np.argmax(y_pred,1)
    return np.mean(y_true==y_pred)

def iterate_minibatches(X, Y, batch_size=64, shuffle=True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i:i+batch_size]
        if j.size:
            yield X[j], Y[j]

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        # params: 리스트 [W1, b1, W2, b2, ...] 형태 (numpy array)
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t))
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * (g * g)
            p[:] -= lr_t * m / (np.sqrt(v) + self.eps)


# -----------------------------
# Conv utils (im2col)
# -----------------------------
def get_im2col_indices(x_shape, field_h, field_w, padding, stride):
    N,C,H,W = x_shape
    out_h = (H+2*padding-field_h)//stride+1
    out_w = (W+2*padding-field_w)//stride+1
    i0 = np.repeat(np.arange(field_h), field_w)
    i0 = np.tile(i0,C)
    i1 = stride*np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(field_w), field_h)
    j0 = np.tile(j0,C)
    j1 = stride*np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1,1)+i1.reshape(1,-1)
    j = j0.reshape(-1,1)+j1.reshape(1,-1)
    k = np.repeat(np.arange(C), field_h*field_w).reshape(-1,1)
    return (k,i,j,out_h,out_w)

def im2col_indices(x, field_h, field_w, padding, stride):
    N,C,H,W = x.shape
    x_padded = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    k,i,j,out_h,out_w = get_im2col_indices(x.shape,field_h,field_w,padding,stride)
    cols = x_padded[:,k,i,j]
    cols = cols.transpose(1,2,0).reshape(C*field_h*field_w,-1)
    return cols,out_h,out_w

def col2im_indices(cols, x_shape, field_h, field_w, padding, stride, out_h, out_w):
    N,C,H,W = x_shape
    x_padded = np.zeros((N,C,H+2*padding,W+2*padding),dtype=cols.dtype)
    k,i,j,_,_ = get_im2col_indices(x_shape,field_h,field_w,padding,stride)
    cols_reshaped = cols.reshape(C*field_h*field_w,out_h*out_w,N)
    cols_reshaped = cols_reshaped.transpose(2,0,1)
    np.add.at(x_padded,(slice(None),k,i,j),cols_reshaped)
    if padding==0: return x_padded
    return x_padded[:,:,padding:-padding,padding:-padding]

# -----------------------------
# Layers
# -----------------------------
class ReLU:
    def forward(self,x):
        self.mask=(x>0).astype(np.float32); return x*self.mask
    def backward(self,grad): return grad*self.mask

class Flatten:
    def forward(self,x): self.in_shape=x.shape; return x.reshape(x.shape[0],-1)
    def backward(self,grad): return grad.reshape(self.in_shape)

class Linear:
    def __init__(self,in_dim,out_dim):
        self.W=np.random.randn(in_dim,out_dim).astype(np.float32)*np.sqrt(2.0/in_dim)
        self.b=np.zeros((1,out_dim),dtype=np.float32)
    def forward(self,x): self.x=x; return x@self.W+self.b
    def backward(self, grad_out):
        dW = self.x.T @ grad_out
        db = np.sum(grad_out, axis=0, keepdims=True)
        dx = grad_out @ self.W.T
        # 업데이트는 여기서 하지 않음 (Adam이 처리)
        return dx, dW, db

class Conv2D:
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        self.C_in=in_channels; self.C_out=out_channels; self.K=kernel_size
        self.stride=stride; self.padding=padding
        self.W=np.random.randn(out_channels,in_channels,self.K,self.K).astype(np.float32)*np.sqrt(2.0/(in_channels*self.K*self.K))
        self.b=np.zeros((out_channels,1),dtype=np.float32)
    def forward(self,x):
        self.x_shape=x.shape
        self.cols,self.out_h,self.out_w=im2col_indices(x,self.K,self.K,self.padding,self.stride)
        self.W_col=self.W.reshape(self.C_out,-1)
        out=self.W_col@self.cols+self.b
        N=x.shape[0]
        return out.reshape(self.C_out,self.out_h,self.out_w,N).transpose(3,0,1,2)
    def backward(self, grad_out):
        grad_r = grad_out.transpose(1,2,3,0).reshape(self.C_out, -1)
        dW_col = grad_r @ self.cols.T
        dW = dW_col.reshape(self.W.shape)
        db = np.sum(grad_r, axis=1, keepdims=True)
        dcols = self.W_col.T @ grad_r
        dx = col2im_indices(dcols, self.x_shape, self.K, self.K,
                            self.padding, self.stride, self.out_h, self.out_w)
        # 업데이트는 여기서 하지 않음
        return dx, dW, db

class MaxPool2D:
    def __init__(self,pool_size=2,stride=2):
        self.P=pool_size; self.stride=stride; assert self.P==self.stride
    def forward(self,x):
        self.x_shape=x.shape; N,C,H,W=x.shape; P=self.P
        x_resh=x.reshape(N,C,H//P,P,W//P,P)
        self.flat=x_resh.reshape(N,C,H//P,W//P,P*P)
        self.argmax=np.argmax(self.flat,axis=4)
        return self.flat.max(axis=4)
    def backward(self,grad):
        N,C,H,W=self.x_shape; P=self.P
        grad_flat=np.zeros_like(self.flat)
        n,c,i,j=np.indices(self.argmax.shape)
        grad_flat[n,c,i,j,self.argmax]=grad
        return grad_flat.reshape(N,C,H//P,P,W//P,P).reshape(N,C,H,W)

# -----------------------------
# CNN Network
# -----------------------------
class CNNNetwork:
    def __init__(self,learning_rate=LEARNING_RATE):
        self.lr=learning_rate
        self.conv1=Conv2D(1,16,3,1,1); self.relu1=ReLU(); self.pool1=MaxPool2D()
        self.conv2=Conv2D(16,32,3,1,1); self.relu2=ReLU(); self.pool2=MaxPool2D()
        self.flat=Flatten(); self.fc1=Linear(32*(IMG_H//4)*(IMG_W//4),256); self.relu3=ReLU()
        self.fc2=Linear(256,OUTPUT_SIZE)

    def forward(self,x,training=True):
        x=x.reshape(-1,1,IMG_H,IMG_W)
        x=self.conv1.forward(x); x=self.relu1.forward(x); x=self.pool1.forward(x)
        x=self.conv2.forward(x); x=self.relu2.forward(x); x=self.pool2.forward(x)
        x=self.flat.forward(x); x=self.fc1.forward(x); x=self.relu3.forward(x)
        logits=self.fc2.forward(x)
        probs=softmax(logits)
        self.last_logits=logits; self.last_probs=probs
        return probs

    def backward(self, y_true):
        N = y_true.shape[0]
        dlogits = (self.last_probs - y_true) / N

        dx, dW_fc2, db_fc2 = self.fc2.backward(dlogits)
        dx = self.relu3.backward(dx)
        dx, dW_fc1, db_fc1 = self.fc1.backward(dx)
        dx = self.flat.backward(dx)
        dx = self.pool2.backward(dx)
        dx = self.relu2.backward(dx)
        dx, dW_conv2, db_conv2 = self.conv2.backward(dx)
        dx = self.pool1.backward(dx)
        dx = self.relu1.backward(dx)
        dx, dW_conv1, db_conv1 = self.conv1.backward(dx)

        grads = [dW_conv1, db_conv1, dW_conv2, db_conv2, dW_fc1, db_fc1, dW_fc2, db_fc2]
        return grads


    def params(self):
        return [self.conv1.W, self.conv1.b,
                self.conv2.W, self.conv2.b,
                self.fc1.W,  self.fc1.b,
                self.fc2.W,  self.fc2.b]

    def train(self, X, Y, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE,
              weight_decay=0.0, lr_decay_epochs=(10, 20), lr_decay_gamma=0.5):
        self.lr = lr
        opt = Adam(self.params(), lr=self.lr)

        for epoch in range(epochs):
            total_loss, total_count, correct = 0.0, 0, 0

            # ✅ 셔플 미니배치
            for xb, yb in iterate_minibatches(X, Y, batch_size, shuffle=True):
                out = self.forward(xb, training=True)
                loss = cross_entropy(yb, out)

                # (옵션) L2 가중치감쇠
                if weight_decay > 0:
                    l2 = 0.0
                    for p in self.params():
                        l2 += np.sum(p * p)
                    loss = loss + 0.5 * weight_decay * l2 / len(xb)

                total_loss += loss * len(xb)
                total_count += len(xb)
                pred = np.argmax(out, axis=1); true = np.argmax(yb, axis=1)
                correct += np.sum(pred == true)

                # 역전파 → 기울기 모음
                grads = self.backward(yb)

                # (옵션) L2의 기울기 추가
                if weight_decay > 0:
                    for g, p in zip(grads, self.params()):
                        g += weight_decay * p

                # Adam step
                opt.step(grads)

            avg_loss = total_loss / max(total_count, 1)
            acc = 100.0 * correct / max(total_count, 1)
            print(f"[CNN] Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}  Acc={acc:.2f}%")

            # 간단 LR decay
            if (epoch + 1) in lr_decay_epochs:
                opt.lr *= lr_decay_gamma
                print(f"  ↘ LR decayed to {opt.lr:.5f}")

    def evaluate(self,X,Y):
        out=self.forward(X,training=False); loss=cross_entropy(Y,out); acc=accuracy(Y,out)
        print(f"[CNN] Test Loss={loss:.4f}, Acc={acc*100:.2f}%"); return loss,acc

    def save(self,filepath="saved_cnn.pkl"):
        params={"conv1_W":self.conv1.W,"conv1_b":self.conv1.b,
                "conv2_W":self.conv2.W,"conv2_b":self.conv2.b,
                "fc1_W":self.fc1.W,"fc1_b":self.fc1.b,
                "fc2_W":self.fc2.W,"fc2_b":self.fc2.b}
        with open(filepath,"wb") as f: pickle.dump(params,f)
        print(f"✅ CNN saved to {filepath}")

    def load(self,filepath="saved_cnn.pkl"):
        with open(filepath,"rb") as f: p=pickle.load(f)
        self.conv1.W,self.conv1.b=p["conv1_W"],p["conv1_b"]
        self.conv2.W,self.conv2.b=p["conv2_W"],p["conv2_b"]
        self.fc1.W,self.fc1.b=p["fc1_W"],p["fc1_b"]
        self.fc2.W,self.fc2.b=p["fc2_W"],p["fc2_b"]
        print(f"✅ CNN loaded from {filepath}")
