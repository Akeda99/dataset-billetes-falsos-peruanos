<p align="right">
  <a href="#english">English</a> · <a href="#español">Español</a>
</p>

---

## Español

# Clasificación de Billetes Peruanos Falsificados

Sistema de identificación de billetes peruanos falsos mediante **Procesamiento Digital de Imágenes (PDI)** y **Deep Learning**.  
16 clases · 9,054 imágenes · 47 técnicas PDI · CNN, CNN-LSTM y CNN-GRU comparadas.

---

### Dataset

| Conjunto | Imágenes |
|---|---|
| Entrenamiento | 6,568 |
| Validación | 2,486 |
| **Total** | **9,054** |

**16 categorías** — 4 denominaciones × 2 caras × 2 versiones (Nuevo / Antiguo):

| Denominación | Personaje (Nuevo) | Personaje (Antiguo) | Color dominante |
|---|---|---|---|
| S/ 10 | Chabuca Granda | José A. Quiñones | Verde |
| S/ 20 | José M. Arguedas | Raúl Porras Barrenechea | Ocre amarillo |
| S/ 50 | María Rostworowski | Abraham Valdelomar | Rojo rosáceo |
| S/ 100 | Pedro Paulet Mostajo | Jorge Basadre Grohmann | Azul acero |

---

### Características de seguridad analizadas

Los billetes genuinos peruanos incluyen los siguientes elementos que el modelo aprende a identificar:

- **Marca de agua multitonal** con el rostro del personaje
- **Hilo de seguridad** que cambia de color según el ángulo (ej. violeta → bronce en S/10)
- **Tinta OVI** (Optically Variable Ink): cambia de color al inclinar el billete
- **Microimpresión** — texto microscópico ilegible sin lupa
- **Registro perfecto** — coincidencia exacta frente/reverso al trasluz
- **Imagen latente** — número visible solo en cierto ángulo
- **Relieve táctil** — sentido al tacto en las figuras principales
- **Fibrillas UV** — visibles solo con luz ultravioleta (rojo, azul, verde)

---

### Procesamiento Digital de Imágenes — 47 técnicas implementadas

#### Conversión de espacios de color
Escala de grises · HSV · LAB · YCrCb · Canales R, G, B individuales · H, S, V · L, A, B

#### Filtros de suavizado (reducción de ruido)
Filtro de media (k=5, k=9) · Gaussian blur (k=5, k=9) · Filtro de mediana · Filtro bilateral · Non-Local Means

#### Mejora de contraste
Ecualización de histograma · CLAHE · Corrección gamma (0.5, 1.5, 2.0) · Estiramiento de contraste por percentiles

#### Detección de bordes *(clave para detectar falsificaciones)*
Canny (50-150, 100-200) · Sobel (magnitud, X, Y) · Laplaciano · Prewitt · Scharr · Roberts Cross · LoG

#### Nitidez y realce
Unsharp masking · High-Boost filtering

#### Segmentación y binarización
Umbralización simple (t=127) · Otsu automático · Umbral adaptativo gaussiano · Umbral adaptativo media

#### Operaciones morfológicas *(análisis de textura)*
Erosión · Dilatación · Apertura · Cierre · Gradiente morfológico · Top Hat · Black Hat

#### Detección de características
Detección de contornos · Harris corners · ORB (500 puntos) · Hough lines

---

### Cómo las técnicas PDI detectan billetes falsos

| Técnica | Qué revela en un billete falso |
|---|---|
| **Análisis de color** | Desviaciones del color característico por denominación |
| **Detección de bordes (Canny, Sobel)** | Bordes borrosos o irregulares vs. la precisión de los genuinos |
| **Textura (entropía, gradiente)** | Diferencias en la calidad de impresión y patrones de seguridad |
| **Histogramas RGB** | Distribución de color distinta a la del billete auténtico |
| **Morfología** | Irregularidades en el relieve y microimpresión |

**Ejemplo — características de textura del S/ 100 Antiguo:**
- Media: 91.46 · Desviación estándar: 70.88 · Entropía: 6.67 · Energía de gradiente: 49.33

---

### Arquitecturas de modelos

#### CNN — Red Convolucional Base
**685,744 parámetros** (2.62 MB)

```
Bloque 1: Conv2D(32) + BN + Conv2D(32) + BN + MaxPool + Dropout(0.25)
Bloque 2: Conv2D(64) + BN + Conv2D(64) + BN + MaxPool + Dropout(0.25)
Bloque 3: Conv2D(128) + BN + Conv2D(128) + BN + MaxPool + Dropout(0.25)
Bloque 4: Conv2D(256) + BN + GlobalAveragePooling
Clasificador: Dense(256) + Dropout(0.5) + Dense(128) + Dropout(0.3) + Dense(16, softmax)
```
Optimizador: Adam (lr=0.0001) · Pérdida: categorical_crossentropy

#### CNN-LSTM — Híbrido con Memoria Secuencial
**244,560 parámetros** (955 KB)

```
Extractor CNN: Conv2D(32) + Conv2D(64) + Conv2D(128) → Reshape(28×28, 128)
LSTM Bidireccional 1: 64 unidades, return_sequences=True + Dropout(0.3)
LSTM Bidireccional 2: 32 unidades, return_sequences=False + Dropout(0.3)
Clasificador: Dense(128) + Dropout(0.5) + Dense(16, softmax)
```
*La CNN extrae patrones espaciales; el LSTM bidireccional captura secuencias de patrones en la imagen.*

#### CNN-GRU — Híbrido más Eficiente
**210,128 parámetros** (820 KB)

Misma estructura que CNN-LSTM pero con capas GRU en lugar de LSTM.  
*Mayor eficiencia de parámetros manteniendo capacidad de modelado secuencial.*

---

### Configuración de entrenamiento

| Parámetro | Valor |
|---|---|
| Tamaño de imagen | 224 × 224 px |
| Batch size | 32 |
| Épocas máx. | 20 |
| Early stopping | paciencia = 7 |
| ReduceLROnPlateau | factor = 0.5, paciencia = 4, min_lr = 1e-7 |

**Augmentación de datos:**
- Rotación: ±10° · Desplazamiento: 10% · Shear: 10% · Zoom: 10% · Brillo: [0.8, 1.2]
- ⚠️ Sin flip horizontal — invertiría el billete e invalidaría la clasificación

---

### Estructura del repositorio

```
dataset-billetes-falsos-peruanos/
├── Billete10_Anverso_Nuevo_Falso/   ┐
├── ...                              │  Imágenes aumentadas (~2,000)
├── Billete100_Reverso_Viejo_Falso/  ┘
├── dataset_reales/                  ← 320 imágenes originales
├── Ray.ipynb                        ← Notebook completo de entrenamiento y PDI
├── augment_billetes.py              ← Script de data augmentation
└── README.md
```

### Stack

Python · TensorFlow 2.19 · OpenCV 4.13 · NumPy 2.0 · Scikit-learn · Matplotlib · GPU

---

---

## English

# Peruvian Counterfeit Banknote Classification

Counterfeit Peruvian banknote detection system using **Digital Image Processing (DIP)** and **Deep Learning**.  
16 classes · 9,054 images · 47 DIP techniques · CNN, CNN-LSTM, and CNN-GRU compared.

---

### Dataset

| Split | Images |
|---|---|
| Training | 6,568 |
| Validation | 2,486 |
| **Total** | **9,054** |

**16 categories** — 4 denominations × 2 sides × 2 versions (New / Old):

| Denomination | Character (New) | Character (Old) | Dominant color |
|---|---|---|---|
| S/ 10 | Chabuca Granda | José A. Quiñones | Green |
| S/ 20 | José M. Arguedas | Raúl Porras Barrenechea | Yellow ochre |
| S/ 50 | María Rostworowski | Abraham Valdelomar | Rosy red |
| S/ 100 | Pedro Paulet Mostajo | Jorge Basadre Grohmann | Steel blue |

---

### Security features analyzed

Genuine Peruvian banknotes include the following elements that the model learns to identify:

- **Multitonal watermark** with the character's face
- **Color-changing security thread** (e.g. violet → bronze on S/10)
- **OVI ink** (Optically Variable Ink): changes color when tilted
- **Microprinting** — microscopic text invisible to the naked eye
- **Perfect register** — exact front/back alignment when held to light
- **Latent image** — denomination number visible only at a specific angle
- **Tactile relief** — felt by touch on the main figures
- **UV fibrils** — visible only under ultraviolet light (red, blue, green)

---

### Digital Image Processing — 47 techniques implemented

#### Color space conversion
Grayscale · HSV · LAB · YCrCb · Individual R, G, B channels · H, S, V · L, A, B

#### Smoothing filters (noise reduction)
Mean filter (k=5, k=9) · Gaussian blur (k=5, k=9) · Median filter · Bilateral filter · Non-Local Means

#### Contrast enhancement
Histogram equalization · CLAHE · Gamma correction (0.5, 1.5, 2.0) · Percentile contrast stretching

#### Edge detection *(key to detecting counterfeits)*
Canny (50-150, 100-200) · Sobel (magnitude, X, Y) · Laplacian · Prewitt · Scharr · Roberts Cross · LoG

#### Sharpening & enhancement
Unsharp masking · High-Boost filtering

#### Segmentation & binarization
Simple thresholding (t=127) · Otsu automatic · Gaussian adaptive · Mean adaptive

#### Morphological operations *(texture analysis)*
Erosion · Dilation · Opening · Closing · Morphological gradient · Top Hat · Black Hat

#### Feature detection
Contour detection · Harris corners · ORB (500 keypoints) · Hough lines

---

### How DIP techniques detect counterfeit bills

| Technique | What it reveals in a fake bill |
|---|---|
| **Color analysis** | Deviation from the denomination's characteristic color |
| **Edge detection (Canny, Sobel)** | Blurry or irregular edges vs. the precision of genuine bills |
| **Texture (entropy, gradient)** | Print quality differences and security pattern irregularities |
| **RGB histograms** | Color distribution different from the authentic bill |
| **Morphology** | Irregularities in relief and microprinting |

**Example — S/ 100 Old texture characteristics:**
- Mean: 91.46 · Std: 70.88 · Entropy: 6.67 · Gradient energy: 49.33

---

### Model architectures

#### CNN — Base Convolutional Network
**685,744 parameters** (2.62 MB)

```
Block 1: Conv2D(32) + BN + Conv2D(32) + BN + MaxPool + Dropout(0.25)
Block 2: Conv2D(64) + BN + Conv2D(64) + BN + MaxPool + Dropout(0.25)
Block 3: Conv2D(128) + BN + Conv2D(128) + BN + MaxPool + Dropout(0.25)
Block 4: Conv2D(256) + BN + GlobalAveragePooling
Classifier: Dense(256) + Dropout(0.5) + Dense(128) + Dropout(0.3) + Dense(16, softmax)
```
Optimizer: Adam (lr=0.0001) · Loss: categorical_crossentropy

#### CNN-LSTM — Hybrid with Sequential Memory
**244,560 parameters** (955 KB)

```
CNN extractor: Conv2D(32) + Conv2D(64) + Conv2D(128) → Reshape(28×28, 128)
Bidirectional LSTM 1: 64 units, return_sequences=True + Dropout(0.3)
Bidirectional LSTM 2: 32 units, return_sequences=False + Dropout(0.3)
Classifier: Dense(128) + Dropout(0.5) + Dense(16, softmax)
```
*CNN extracts spatial features; bidirectional LSTM captures sequential patterns across the image.*

#### CNN-GRU — More Efficient Hybrid
**210,128 parameters** (820 KB)

Same structure as CNN-LSTM but with GRU layers instead of LSTM.  
*More parameter-efficient while maintaining sequential modeling capacity.*

---

### Training configuration

| Parameter | Value |
|---|---|
| Image size | 224 × 224 px |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | patience = 7 |
| ReduceLROnPlateau | factor = 0.5, patience = 4, min_lr = 1e-7 |

**Data augmentation:**
- Rotation: ±10° · Shift: 10% · Shear: 10% · Zoom: 10% · Brightness: [0.8, 1.2]
- ⚠️ No horizontal flip — would reverse the bill and invalidate classification

---

### Repository structure

```
dataset-billetes-falsos-peruanos/
├── Billete10_Anverso_Nuevo_Falso/   ┐
├── ...                              │  Augmented images (~2,000)
├── Billete100_Reverso_Viejo_Falso/  ┘
├── dataset_reales/                  ← 320 original images
├── Ray.ipynb                        ← Full training and DIP notebook
├── augment_billetes.py              ← Data augmentation script
└── README.md
```

### Stack

Python · TensorFlow 2.19 · OpenCV 4.13 · NumPy 2.0 · Scikit-learn · Matplotlib · GPU

---

© 2026 Ray Cardenas. All rights reserved.
