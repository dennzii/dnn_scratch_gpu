import cupy as cp
from dnn import dnn_model
from load_images import load_images_from_disk
# XOR tabanlı temel veri noktaları

IMAGE_SHAPE = 64
INPUT_LAYER_SIZE = IMAGE_SHAPE * IMAGE_SHAPE

X,Y = load_images_from_disk(IMAGE_SHAPE)


layer_dims = [INPUT_LAYER_SIZE, 256, 64, 1]
model = dnn_model(layer_dims, lr=0.1)

# Eğitim
model.train(X, Y, epochs=150, batch_size=256)

# Tahmin
preds = model.predict(X)
print("Tahminler (ilk 10):", preds[:, :10])
print("Gerçek Y (ilk 10):", Y[:, :10])
