import cupy as cp
from dnn import dnn_model
from load_images import load_images_from_disk
import matplotlib.pyplot as plt
# XOR tabanlı temel veri noktaları

import matplotlib.pyplot as plt

def to_float_list(lst):
    return [float(x.get()) if hasattr(x, 'get') else float(x) for x in lst]

def plot_training_logs(logs):
    # logs = [(avg_loss, val_loss, val_acc), ...]
    loss_log = to_float_list([log[0] for log in logs])
    val_loss_log = to_float_list([log[1] for log in logs])
    val_acc_log = to_float_list([log[2] for log in logs])
    epochs = list(range(1, len(logs) + 1))

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_log, label='Training Loss')
    plt.plot(epochs, val_loss_log, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc_log, label='Validation Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


IMAGE_SHAPE = 64
INPUT_LAYER_SIZE = IMAGE_SHAPE * IMAGE_SHAPE

X,Y = load_images_from_disk(IMAGE_SHAPE)

X_tr = X[:,500:]
Y_tr = Y[:,500:]

X_val = X[:,:500]
Y_val = Y[:,:500]

print(cp.sum(Y_val,axis=1))

layer_dims = [INPUT_LAYER_SIZE, 256, 32, 1]
model = dnn_model(layer_dims, lr=0.07,lambd=0.3)

# Eğitim
logs,best_parameters_log,best_parameters = model.train(X_tr, Y_tr,X_val,Y_val, epochs=150, batch_size=32)

print(best_parameters_log)

# Örnek çağrı
plot_training_logs(logs)

# Tahmin
preds = model.predict(X)
print("Tahminler (ilk 10):", preds[:, :10])
print("Gerçek Y (ilk 10):", Y[:, :10])



