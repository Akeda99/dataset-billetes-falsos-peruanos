<p align="right">
  <a href="#english">English</a> · <a href="#español">Español</a>
</p>

---

## Español

# Dataset — Clasificación de Billetes Peruanos Falsos

Dataset de imágenes para detección y clasificación de billetes peruanos falsos.  
16 clases · 2,320+ imágenes · Arquitecturas CNN, CNN-LSTM y CNN-GRU evaluadas.

### Clases (16 en total)

Cada denominación tiene 4 variantes: Anverso/Reverso × Nuevo/Viejo

| Denominación | Clases |
|---|---|
| S/ 10 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |
| S/ 20 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |
| S/ 50 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |
| S/ 100 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |

### Estructura

```
dataset-billetes-falsos-peruanos/
├── Billete10_Anverso_Nuevo_Falso/   ┐
├── ...                              │  Imágenes aumentadas (~2,000)
├── Billete100_Reverso_Viejo_Falso/  ┘
├── dataset_reales/                  ← Imágenes originales (320)
├── Ray.ipynb                        ← Notebook de entrenamiento
├── augment_billetes.py              ← Script de data augmentation
└── README.md
```

### Data Augmentation

```bash
python augment_billetes.py
```

Técnicas aplicadas: rotaciones, variaciones de brillo/contraste, flips, zoom y recortes.

### Modelos evaluados

| Arquitectura | Descripción |
|---|---|
| **CNN** | Red convolucional base |
| **CNN-LSTM** | CNN + memoria secuencial LSTM |
| **CNN-GRU** | CNN + memoria secuencial GRU |

### Stack

Python · TensorFlow · Keras · OpenCV · NumPy

---

## English

# Dataset — Peruvian Counterfeit Banknote Classification

Image dataset for detection and classification of counterfeit Peruvian banknotes.  
16 classes · 2,320+ images · CNN, CNN-LSTM, and CNN-GRU architectures evaluated.

### Classes (16 total)

Each denomination has 4 variants: Obverse/Reverse × New/Old

| Denomination | Classes |
|---|---|
| S/ 10 | Obverse New, Obverse Old, Reverse New, Reverse Old |
| S/ 20 | Obverse New, Obverse Old, Reverse New, Reverse Old |
| S/ 50 | Obverse New, Obverse Old, Reverse New, Reverse Old |
| S/ 100 | Obverse New, Obverse Old, Reverse New, Reverse Old |

### Structure

```
dataset-billetes-falsos-peruanos/
├── Billete10_Anverso_Nuevo_Falso/   ┐
├── ...                              │  Augmented images (~2,000)
├── Billete100_Reverso_Viejo_Falso/  ┘
├── dataset_reales/                  ← Original images (320)
├── Ray.ipynb                        ← Training notebook
├── augment_billetes.py              ← Data augmentation script
└── README.md
```

### Data Augmentation

```bash
python augment_billetes.py
```

Techniques: random rotations, brightness/contrast variation, flips, zoom, and cropping.

### Evaluated models

| Architecture | Description |
|---|---|
| **CNN** | Base convolutional network |
| **CNN-LSTM** | CNN + LSTM sequential memory |
| **CNN-GRU** | CNN + GRU sequential memory |

### Stack

Python · TensorFlow · Keras · OpenCV · NumPy

---

© 2026 Ray Cardenas. All rights reserved.
