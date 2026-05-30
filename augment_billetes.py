import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path

SRC_DIR = Path(r"C:\Users\rayke\Desktop\dataset_falsos_reales")
DST_DIR = Path(r"C:\Users\rayke\Desktop\dataset_falsos_aumentado")

ORIGINALS = 20
TARGET    = 125
SYNTHETIC = TARGET - ORIGINALS  # 105


# ── Transformaciones ──────────────────────────────────────────────────────────

def brightness_contrast(img):
    alpha = random.uniform(0.75, 1.30)   # contraste
    beta  = random.uniform(-35, 35)      # brillo
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

def gaussian_blur(img):
    ksize = random.choice([3, 3, 5])     # leve: mayormente k=3
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def gaussian_noise(img):
    std   = random.uniform(2, 18)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def rotation(img):
    angle = random.uniform(-15, 15)
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def zoom(img):
    scale    = random.uniform(0.85, 1.15)
    h, w     = img.shape[:2]
    new_h    = int(h * scale)
    new_w    = int(w * scale)
    resized  = cv2.resize(img, (new_w, new_h))

    if scale >= 1.0:                     # zoom in → recortar al centro
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        return resized[y0:y0 + h, x0:x0 + w]
    else:                                # zoom out → rellenar bordes
        py = (h - new_h) // 2
        px = (w - new_w) // 2
        return cv2.copyMakeBorder(
            resized,
            py, h - new_h - py,
            px, w - new_w - px,
            borderType=cv2.BORDER_REFLECT_101,
        )

def perspective(img):
    h, w      = img.shape[:2]
    shift     = int(min(h, w) * 0.04)
    def r():  return random.randint(0, shift)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [r(), r()],
        [w - r(), r()],
        [r(), h - r()],
        [w - r(), h - r()],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def saturation_reduction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.45, 1.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def yellow_tint(img):
    strength = random.uniform(5, 28)
    overlay  = np.zeros_like(img, dtype=np.float32)
    overlay[:, :, 2] =  strength          # R ↑
    overlay[:, :, 1] =  strength * 0.75   # G ↑ leve
    overlay[:, :, 0] = -strength * 0.35   # B ↓
    return np.clip(img.astype(np.float32) + overlay, 0, 255).astype(np.uint8)

def sharpness_reduction(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 1.2)
    alpha   = random.uniform(0.35, 0.70)  # peso del original (menos = más suave)
    return cv2.addWeighted(img, alpha, blurred, 1 - alpha, 0)


# ── Pipeline de augmentación ─────────────────────────────────────────────────

TRANSFORMS = [
    (rotation,             0.85),
    (zoom,                 0.75),
    (perspective,          0.60),
    (brightness_contrast,  0.90),
    (gaussian_blur,        0.55),
    (gaussian_noise,       0.75),
    (saturation_reduction, 0.65),
    (yellow_tint,          0.50),
    (sharpness_reduction,  0.45),
]

def augment(img):
    for fn, prob in TRANSFORMS:
        if random.random() < prob:
            img = fn(img)
    return img


# ── Procesamiento por carpeta ─────────────────────────────────────────────────

def process_folder(src: Path, dst: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    originals  = [p for p in src.iterdir() if p.suffix in extensions]

    # Copiar originales
    for p in originals:
        shutil.copy2(p, dst / p.name)

    # Cargar en memoria
    imgs = [cv2.imread(str(p)) for p in originals]
    imgs = [i for i in imgs if i is not None]

    # Generar sintéticas
    generated = 0
    for i in range(SYNTHETIC):
        src_img = random.choice(imgs)
        aug     = augment(src_img.copy())
        out     = dst / f"aug_{i + 1:04d}.jpg"
        cv2.imwrite(str(out), aug, [cv2.IMWRITE_JPEG_QUALITY, 93])
        generated += 1

    total = len([p for p in dst.iterdir() if p.suffix in extensions])
    return {"orig": len(originals), "sint": generated, "total": total}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    if not SRC_DIR.exists():
        print(f"\nERROR: No se encontró la ruta {SRC_DIR}")
        return

    folders = sorted([f for f in SRC_DIR.iterdir() if f.is_dir()])
    if not folders:
        print("No se encontraron subcarpetas en el dataset.")
        return

    print(f"\n{'='*62}")
    print(f"   DATA AUGMENTATION - Billetes Peruanos")
    print(f"{'='*62}")
    print(f"   Fuente : {SRC_DIR}")
    print(f"   Destino: {DST_DIR}")
    print(f"   Carpetas: {len(folders)}  |  Objetivo: {TARGET} imgs/carpeta")
    print(f"{'='*62}\n")

    results = []
    for idx, folder in enumerate(folders, 1):
        label = f"[{idx:2d}/{len(folders)}]"
        print(f"  {label} {folder.name:<35}", end="", flush=True)
        info = process_folder(folder, DST_DIR / folder.name)
        results.append({"name": folder.name, **info})
        print(f"OK  {info['total']:>3} imgs")

    # ── Tabla final ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"   RESUMEN FINAL")
    print(f"{'='*62}")
    print(f"   {'Carpeta':<33} {'Orig':>5} {'Sint':>6} {'Total':>6}")
    print(f"   {'-'*33} {'-'*5} {'-'*6} {'-'*6}")

    sum_o = sum_s = sum_t = 0
    for r in results:
        print(f"   {r['name']:<33} {r['orig']:>5} {r['sint']:>6} {r['total']:>6}")
        sum_o += r["orig"]
        sum_s += r["sint"]
        sum_t += r["total"]

    print(f"   {'-'*33} {'-'*5} {'-'*6} {'-'*6}")
    print(f"   {'TOTAL':<33} {sum_o:>5} {sum_s:>6} {sum_t:>6}")
    print(f"{'='*62}")
    print(f"\n   Dataset aumentado guardado en:\n   {DST_DIR}\n")


if __name__ == "__main__":
    main()
