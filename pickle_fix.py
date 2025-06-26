import pickle
import cupy as cp
import numpy as np

# 1. Orijinal dosyayı aç
with open('parameters_best.pickle', 'rb') as f:
    params = pickle.load(f)

# 2. Her değeri numpy array’e çevir
fixed = {}
for k, v in params.items():
    if isinstance(v, cp.ndarray):
        fixed[k] = cp.asnumpy(v)
    else:
        fixed[k] = np.array(v)

# 3. Yeni dosyaya dump et
with open('parameters_best_fixed.pickle', 'wb') as f:
    pickle.dump(fixed, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Yenilenen pickle kaydedildi: parameters_best_fixed.pickle")
