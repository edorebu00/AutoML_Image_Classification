from tensorflow.keras.datasets import cifar10
import autokeras as ak
import numpy as np
import os 

# 1. Carichiamo il dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. Controlliamo la forma dei dati
print("Forma x_train:", x_train.shape)
print("Forma y_train:", y_train.shape)
print("Forma x_test:", x_test.shape)
print("Forma y_test:", y_test.shape)

# 3. Normalizziamo i pixel nell'intervallo [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 4. Rimuoviamo la dimensione inutile dalle etichette
y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("Prime 10 etichette:", y_train[:10])

print("\nInformazioni sul dataset")
print("------------------------")
print("Tipo x_train:", x_train.dtype)
print("Valore minimo:", x_train.min())
print("Valore massimo:", x_train.max())

print("\nNumero classi:", len(np.unique(y_train)))
print("Classi:", np.unique(y_train))


np.random.seed(42)

indici = np.random.choice(
    len(x_train),
    size=10000,
    replace=False
)

x_train_small = x_train[:10000]
y_train_small = y_train[:10000]

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=1
)

clf.fit(
    x_train_small,
    y_train_small,
    epochs=3,
    validation_split=0.2
)

accuracy = clf.evaluate(
    x_test,
    y_test
)

print("Accuratezza", accuracy)

best_model = clf.export_model()

project_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(project_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)
best_model.save(os.path.join(save_dir, "Best_Model_AK.keras"))


