import cv2 as cv
import cupy as cp
import glob
from tqdm import tqdm

def load_images_from_disk(pixel):
    X = []
    Y = []
    
    print("preprocessing images vehicles")
    for file in tqdm(glob.iglob('vehicles/*')):
        img = cv.imread(file)
        if img is None:
            print(f"Warning: could not read {file}, skipping.")
            continue
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img_gray = cv.resize(img_gray, (pixel, pixel))  # uncomment if needed
        res = cp.asarray(img_gray).flatten() / 255.0
        X.append(res)
        Y.append(1)

    print("preprocessing images non-vehicles")
    for file in tqdm(glob.iglob('non-vehicles/*')):
        img = cv.imread(file)
        if img is None:
            print(f"Warning: could not read {file}, skipping.")
            continue
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img_gray = cv.resize(img_gray, (pixel, pixel))  # uncomment if needed
        res = cp.asarray(img_gray).flatten() / 255.0
        X.append(res)
        Y.append(0)

    if len(X) == 0:
        raise ValueError("No images loaded. Please check your folders.")

    # Liste -> Cupy array
    X = cp.stack(X).T  # shape: (features, samples)
    Y = cp.asarray(Y).reshape((1, X.shape[1]))

    # Shuffle
    permutation = cp.random.permutation(X.shape[1])
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    return shuffled_X, shuffled_Y


def resize_and_flatten(img, image_shape):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (image_shape, image_shape))
    img = cp.asarray(img).flatten() / 255.0
    img = img.reshape((img.shape[0], 1))  # shape: (features, 1)
    return img
