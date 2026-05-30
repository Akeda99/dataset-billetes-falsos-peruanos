# Dataset — Clasificación de Billetes Peruanos Falsos

Dataset de imágenes para detección y clasificación de billetes peruanos falsos.  
16 clases · 2,320+ imágenes · Arquitecturas CNN, CNN-LSTM y CNN-GRU evaluadas.

---

## Clases (16 en total)

Cada denominación tiene 4 variantes: Anverso/Reverso × Nuevo/Viejo

| Denominación | Clases |
|---|---|
| S/ 10 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |
| S/ 20 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |
| S/ 50 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |
| S/ 100 | Anverso Nuevo, Anverso Viejo, Reverso Nuevo, Reverso Viejo |

---

## Estructura del repositorio

```
dataset-billetes-falsos-peruanos/
├── Billete10_Anverso_Nuevo_Falso/   ┐
├── Billete10_Anverso_Viejo_Falso/   │
├── Billete10_Reverso_Nuevo_Falso/   │  Imágenes aumentadas
├── ...                              │  (~2,000 imágenes)
├── Billete100_Reverso_Viejo_Falso/  ┘
│
├── dataset_reales/                  ← Imágenes originales (320)
│   ├── Billete10_Anverso_Nuevo_Falso/
│   ├── ...
│   └── Billete100_Reverso_Viejo_Falso/
│
├── augment_billetes.py              ← Script de data augmentation
└── README.md
```

---

## Data Augmentation

Las imágenes aumentadas se generaron con `augment_billetes.py` aplicando:
- Rotaciones aleatorias
- Variaciones de brillo y contraste
- Flips horizontales/verticales
- Zoom y recortes

```bash
python augment_billetes.py
```

---

## Modelos evaluados

| Arquitectura | Descripción |
|---|---|
| **CNN** | Red convolucional base |
| **CNN-LSTM** | CNN + memoria secuencial LSTM |
| **CNN-GRU** | CNN + memoria secuencial GRU |

---

## Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy

---

## Contexto

Dataset recopilado y procesado en Arequipa, Perú, para investigación en detección de billetes falsos mediante visión computacional.

---

© 2026 Ray Cardenas. Todos los derechos reservados.
