## Model

The trained model is intentionally excluded from this repository to keep it lightweight.

Before running predictions, generate the model by executing:

```bash
python train_autokeras.py
```

This will create the following directory automatically:

```
saved_models/
└── Best_Model_AK.keras
```

Once the model has been generated, you can run:

```bash
python predict.py
```