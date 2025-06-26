from dnn import dnn_model

IMAGE_SHAPE = 64
INPUT_LAYER_SIZE = IMAGE_SHAPE * IMAGE_SHAPE

layer_dims = [INPUT_LAYER_SIZE, 256, 32, 1]
model = dnn_model(layer_dims, lr=0.07,lambd=0.1)

model.load_model("best.npz")

model.object_detection_on_video("video.mp4")
