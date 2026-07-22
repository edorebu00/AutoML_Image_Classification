import os
import csv
import time
from collections import Counter

import numpy as np
import tensorflow as tf
import autokeras as ak
import matplotlib.pyplot as plt


class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

start_time = time.time()

project_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    project_dir,
    "saved_models",
    "Best_Model_AK.keras"
)

images_dir = os.path.join(
    project_dir,
    "images"
)

csv_path = os.path.join(
    project_dir,
    "risultati.csv"
)

if not os.path.exists(model_path):
    print(f"Modello non trovato: {model_path}")
    raise SystemExit(1)

if not os.path.isdir(images_dir):
    print(f"Cartella delle immagini non trovata: {images_dir}")
    raise SystemExit(1)

model = tf.keras.models.load_model(
    model_path,
    custom_objects=ak.CUSTOM_OBJECTS
)

print("Modello caricato correttamente!")

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

image_names = sorted([
    file_name
    for file_name in os.listdir(images_dir)
    if file_name.lower().endswith(valid_extensions)
])

print(f"\nTrovate {len(image_names)} immagini:")

for file_name in image_names:
    print(f"- {file_name}")

images = []
image_arrays = []
valid_image_names = []
skipped_images = 0

for image_name in image_names:
    image_path = os.path.join(
        images_dir,
        image_name
    )

    try:
        image = tf.keras.utils.load_img(
            image_path,
            target_size=(32, 32)
        )

        image_array = tf.keras.utils.img_to_array(image)
        image_array = image_array.astype("float32") / 255.0

        images.append(image)
        image_arrays.append(image_array)
        valid_image_names.append(image_name)

    except Exception as error:
        print(
            f"Errore durante il caricamento di "
            f"{image_name}: {error}"
        )
        skipped_images += 1

if len(image_arrays) == 0:
    print("Nessuna immagine valida trovata.")
    raise SystemExit(1)

image_batch = np.stack(image_arrays)

print(f"\nForma del batch: {image_batch.shape}")

predictions = model.predict(
    image_batch,
    verbose=1
)

predicted_classes = []

with open(
    csv_path,
    mode="w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "Image",
        "Prediction",
        "Confidence (%)",
        "Second prediction",
        "Second confidence (%)",
        "Third prediction",
        "Third confidence (%)"
    ])

    for image_name, image, prediction in zip(
        valid_image_names,
        images,
        predictions
    ):
        top_3 = np.argsort(prediction)[-3:][::-1]

        predicted_index = top_3[0]
        predicted_class = class_names[predicted_index]
        confidence = prediction[predicted_index] * 100

        second_index = top_3[1]
        second_class = class_names[second_index]
        second_confidence = prediction[second_index] * 100

        third_index = top_3[2]
        third_class = class_names[third_index]
        third_confidence = prediction[third_index] * 100

        predicted_classes.append(predicted_class)

        writer.writerow([
            image_name,
            predicted_class,
            round(confidence, 2),
            second_class,
            round(second_confidence, 2),
            third_class,
            round(third_confidence, 2)
        ])

        print("\n------------------------------")
        print(f"Immagine: {image_name}")
        print(f"Classe prevista: {predicted_class}")
        print(f"Confidenza: {confidence:.2f}%")

        print("\nTop 3:")

        for index in top_3:
            class_name = class_names[index]
            class_confidence = prediction[index] * 100

            print(
                f"{class_name} -> "
                f"{class_confidence:.2f}%"
            )

        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(
            f"{predicted_class} "
            f"({confidence:.2f}%)"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

execution_time = time.time() - start_time
class_distribution = Counter(predicted_classes)

print("\n==============================")
print("RIEPILOGO FINALE")
print("==============================")
print(f"Immagini trovate: {len(image_names)}")
print(f"Immagini analizzate: {len(valid_image_names)}")
print(f"Immagini scartate: {skipped_images}")
print(f"Tempo totale: {execution_time:.2f} secondi")

print("\nDistribuzione delle classi:")

for class_name, count in class_distribution.most_common():
    print(f"- {class_name}: {count}")

print("\nRisultati salvati in:")
print(csv_path)