import argparse
import os
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
from docx import Document
from docx.shared import Pt
from tqdm import tqdm
from ultralytics import YOLO
import torch

# Input clear images
INPUT_DIR = os.path.join("input_clear", "images")

# Output structure
OUT_ROOT = "output_dataset"
CLEAR_IMG_OUT = os.path.join(OUT_ROOT, "images", "clear")
CLEAR_LBL_OUT = os.path.join(OUT_ROOT, "labels", "clear")
SYN_IMG_OUT = os.path.join(OUT_ROOT, "images", "rain_lowlight")
SYN_LBL_OUT = os.path.join(OUT_ROOT, "labels", "rain_lowlight")
VIS_CLEAR_OUT = os.path.join(OUT_ROOT, "visualizations", "clear")
VIS_SYN_OUT = os.path.join(OUT_ROOT, "visualizations", "rain_lowlight")
META_OUT = os.path.join(OUT_ROOT, "metadata", "rain_lowlight_metadata.docx")

# Required classes
CUSTOM_CLASS_ORDER = ["person", "car", "traffic_light", "bicycle", "bus"]

# COCO -> custom ids
COCO_TO_CUSTOM = {
    0: 0,  # person
    2: 1,  # car
    9: 2,  # traffic light
    1: 3,  # bicycle
    5: 4,  # bus
}

# Severity parameterization ranges (from rubric section 5.2)
# Randomized per (image, severity) using a fixed seed for exact regeneration.
SEVERITIES = {
    "low": {
        "label": "Light",
        "rain_intensity_range": (0.15, 0.30),   # +50% streak density
        "streak_length_range": (10, 25),        # px
        "brightness_scale_range": (0.75, 0.90), # lower -> darker
        "gamma_range": (1.1, 1.3),              # higher -> darker midtones
        "thickness": 1,
        "blur": 2,
        "haze": 0.04,
        "contrast_drop": 0.06,
        "vignette": 0.08,
        "bloom": 0.025,
        "droplets": 34,
        "angle_deg_range": (-12.0, 12.0),
        "noise_level": 0.008,
        "ca_shift": 1,
        "rain_sharpness": 1.35,
        "desaturation": 0.06,
        "black_crush": 0.10,
        "ground_wetness": 0.12,
        "clarity_boost": 1.00,
        "rain_visibility_boost": 1.55,
    },
    "mid": {
        "label": "Medium",
        "rain_intensity_range": (0.30, 0.525),  # +50% streak density
        "streak_length_range": (20, 40),
        "brightness_scale_range": (0.55, 0.75),
        "gamma_range": (1.3, 1.6),
        "thickness": 1,
        "blur": 3,
        "haze": 0.08,
        "contrast_drop": 0.14,
        "vignette": 0.12,
        "bloom": 0.05,
        "droplets": 56,
        "angle_deg_range": (-16.0, 16.0),
        "noise_level": 0.013,
        "ca_shift": 1,
        "rain_sharpness": 1.45,
        "desaturation": 0.10,
        "black_crush": 0.16,
        "ground_wetness": 0.18,
        "clarity_boost": 1.03,
        "rain_visibility_boost": 1.40,
    },
    "high": {
        "label": "Heavy",
        "rain_intensity_range": (0.825, 1.00),  # +50% streak density (capped)
        "streak_length_range": (30, 60),
        "brightness_scale_range": (0.35, 0.55),
        "gamma_range": (1.6, 2.0),
        "thickness": 2,
        "blur": 7,
        "haze": 0.13,
        "contrast_drop": 0.18,
        "vignette": 0.18,
        "bloom": 0.07,
        "droplets": 78,
        "angle_deg_range": (-20.0, 20.0),
        "noise_level": 0.018,
        "ca_shift": 2,
        "rain_sharpness": 1.35,
        "desaturation": 0.14,
        "black_crush": 0.22,
        "ground_wetness": 0.24,
        "clarity_boost": 1.08,
        "rain_visibility_boost": 1.15,
    },
}

# YOLO config
YOLO_MODEL_WEIGHTS = os.environ.get("YOLO_MODEL_WEIGHTS", "yolov8n.pt")
CONF_THRES = 0.25
IOU_THRES = 0.45
YOLO_BATCH = 32
YOLO_IMGSZ = 640
YOLO_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Preview box drawing
BOX_COLOR = (0, 255, 0)  # bright green (BGR)


def resolve_yolo_weights():
    candidates = [
        YOLO_MODEL_WEIGHTS,
        "/Users/varunbalaji/Desktop/synth/yolov8n.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return YOLO_MODEL_WEIGHTS


def ensure_dirs(render_previews: bool):
    os.makedirs(CLEAR_IMG_OUT, exist_ok=True)
    os.makedirs(CLEAR_LBL_OUT, exist_ok=True)
    os.makedirs(SYN_IMG_OUT, exist_ok=True)
    os.makedirs(SYN_LBL_OUT, exist_ok=True)
    if render_previews:
        os.makedirs(VIS_CLEAR_OUT, exist_ok=True)
        os.makedirs(VIS_SYN_OUT, exist_ok=True)
    os.makedirs(os.path.dirname(META_OUT), exist_ok=True)


def write_classes_txt():
    path = os.path.join(OUT_ROOT, "classes.txt")
    with open(path, "w", encoding="utf-8") as f:
        for c in CUSTOM_CLASS_ORDER:
            f.write(c + "\n")


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def xyxy_to_yolo_line(x1, y1, x2, y2, w, h, cls_id_custom):
    x1 = clamp(x1, 0, w - 1)
    x2 = clamp(x2, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    y2 = clamp(y2, 0, h - 1)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return f"{cls_id_custom} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def parse_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                boxes.append((cls_id, cx, cy, bw, bh))
            except ValueError:
                continue
    return boxes


def draw_boxes_from_yolo(image_path, label_path, out_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    boxes = parse_yolo_labels(label_path)
    vis = img.copy()

    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)

        x1 = clamp(x1, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            continue

        cls_name = CUSTOM_CLASS_ORDER[cls_id] if 0 <= cls_id < len(CUSTOM_CLASS_ORDER) else f"class_{cls_id}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2, lineType=cv2.LINE_AA)

        text = cls_name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        ty1 = max(0, y1 - th - 8)
        ty2 = max(th + 8, y1)
        tx2 = min(w - 1, x1 + tw + 8)
        cv2.rectangle(vis, (x1, ty1), (tx2, ty2), BOX_COLOR, -1)
        cv2.putText(vis, text, (x1 + 4, ty2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return True


def make_rain_overlay_rgba(width, height, seed, cfg):
    rng = np.random.default_rng(seed)
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    # 3 rain depth layers for realism
    layers = [
        (0.25, 0.75, 0.9, 0.8),
        (0.50, 1.00, 1.0, 1.0),
        (0.25, 1.35, 1.4, 1.2),
    ]

    base_streaks = int(cfg["streaks"])
    base_length = float(cfg["length"])
    base_thick = int(cfg["thickness"])

    base_angle_deg = float(cfg.get("angle_deg", 0.0))
    for frac, len_scale, alpha_scale, thick_scale in layers:
        streaks = int(base_streaks * frac)
        length = max(6, int(base_length * len_scale))
        thickness = max(1, int(round(base_thick * thick_scale)))

        # Small layer jitter around deterministic base angle to mimic wind fluctuation.
        angle = np.deg2rad(base_angle_deg + rng.uniform(-4.0, 4.0))
        dx = int(np.cos(angle) * length)
        dy = int(np.sin(angle) * length) + length

        for _ in range(streaks):
            x = int(rng.integers(0, width))
            y = int(rng.integers(-height // 2, height))
            x2 = x + dx
            y2 = y + dy
            alpha = int(rng.integers(80, 190) * alpha_scale)
            alpha = int(clamp(alpha, 40, 240))
            color = (255, 255, 255, alpha)
            cv2.line(overlay, (x, y), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)

    # Lens droplets + occasional streak trails from droplet slip on windshield.
    droplet_count = int(cfg.get("droplets", 0))
    if droplet_count > 0:
        for _ in range(droplet_count):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            r = int(rng.integers(4, 16))
            a = int(rng.integers(24, 88))
            cv2.circle(overlay, (x, y), r, (220, 220, 220, a), -1, lineType=cv2.LINE_AA)
            # small highlight ring to mimic refractive droplet edge
            cv2.circle(overlay, (x, y), max(2, r - 2), (255, 255, 255, min(120, a + 22)), 1, lineType=cv2.LINE_AA)
            if rng.random() < 0.35:
                trail_len = int(rng.integers(8, 38))
                tx = int(x + rng.integers(-2, 3))
                ty = int(min(height - 1, y + trail_len))
                ta = int(max(18, a - 20))
                cv2.line(overlay, (x, y), (tx, ty), (220, 220, 220, ta), thickness=max(1, r // 6), lineType=cv2.LINE_AA)

    blur = int(cfg["blur"])
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        overlay = cv2.GaussianBlur(overlay, (k, k), 0)

    # Slightly brighten rain so streaks remain visible under low-light.
    overlay[:, :, :3] = (overlay[:, :, :3] * 0.94).astype(np.uint8)
    return overlay


def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx * xx + yy * yy)
    mask = 1.0 - strength * np.clip(rr, 0.0, 1.0)
    mask = np.clip(mask, 0.6, 1.0)
    return img * mask[:, :, None]


def apply_depth_haze(comp, base_haze):
    # Heuristic depth map from vertical position: farther near top, closer near bottom.
    h = comp.shape[0]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    depth_haze = base_haze * (1.25 - 0.85 * y)
    haze_color = np.array([0.62, 0.66, 0.72], dtype=np.float32)[None, None, :]
    return comp * (1.0 - depth_haze) + haze_color * depth_haze


def add_wet_road_reflection(comp, strength):
    if strength <= 0:
        return comp
    h, w = comp.shape[:2]
    start = int(h * 0.58)
    if start >= h - 2:
        return comp
    bottom = comp[start:, :, :]
    flip = cv2.flip(bottom, 0)
    # Smooth and darken reflection layer.
    refl = cv2.GaussianBlur(flip, (0, 0), 6.0)
    refl *= 0.75
    y = np.linspace(0.0, 1.0, bottom.shape[0], dtype=np.float32)[:, None, None]
    fade = (1.0 - y) ** 1.8
    comp[start:, :, :] = np.clip(bottom * (1.0 - strength * fade) + refl * (strength * fade), 0.0, 1.0)
    return comp


def add_ground_wetness(comp, strength):
    """
    Adds subtle puddle/sheen appearance in lower image region to make rain feel grounded.
    """
    if strength <= 0:
        return comp
    h, w = comp.shape[:2]
    start = int(h * 0.55)
    if start >= h - 2:
        return comp

    roi = comp[start:, :, :].copy()
    # Smooth "water layer" + mild brightness lift for specular wet look
    water = cv2.GaussianBlur(roi, (0, 0), 5.0)
    water = np.clip(water * 1.05, 0.0, 1.0)

    # Vertical fade so strongest near bottom, weaker upward
    y = np.linspace(0.0, 1.0, roi.shape[0], dtype=np.float32)[:, None, None]
    fade = y ** 1.5
    blend = strength * (0.35 + 0.65 * fade)
    comp[start:, :, :] = np.clip(roi * (1.0 - blend) + water * blend, 0.0, 1.0)
    return comp


def add_sensor_artifacts(comp, seed, noise_level, ca_shift):
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    out = comp.copy()
    if noise_level > 0:
        noise = rng.normal(0.0, noise_level, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    # Mild chromatic aberration by channel shifts.
    s = int(max(0, ca_shift))
    if s > 0:
        b, g, r = cv2.split((out * 255.0).astype(np.uint8))
        r = np.roll(r, s, axis=1)
        b = np.roll(b, -s, axis=1)
        out = cv2.merge([b, g, r]).astype(np.float32) / 255.0
    return out


def tonal_lowlight_curve(comp, desaturation, black_crush):
    # Film-like low-light tone behavior (not just brightness): darker shadows, muted colors.
    y = 0.114 * comp[:, :, 0] + 0.587 * comp[:, :, 1] + 0.299 * comp[:, :, 2]
    shadow = np.clip((0.45 - y) / 0.45, 0.0, 1.0)[:, :, None]
    comp = comp * (1.0 - black_crush * shadow)

    gray = np.repeat(y[:, :, None], 3, axis=2)
    comp = comp * (1.0 - desaturation) + gray * desaturation
    return np.clip(comp, 0.0, 1.0)


def run_realistic_composite(clear_bgr, rain_rgba, cfg):
    clear = clear_bgr.astype(np.float32) / 255.0
    rain_rgb = rain_rgba[:, :, :3].astype(np.float32) / 255.0
    rain_a = rain_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rain_a = np.clip(rain_a * float(cfg["rain_alpha"]) * float(cfg.get("rain_visibility_boost", 1.0)), 0.0, 1.0)

    comp = clear * (1.0 - rain_a) + rain_rgb * rain_a

    # brightness + gamma from severity parameterization table
    comp = comp * float(cfg["brightness_scale"])
    comp = np.power(np.clip(comp, 0.0, 1.0), float(cfg["gamma"]))

    # depth-varying haze/fog veil
    haze = float(cfg["haze"])
    comp = apply_depth_haze(comp, haze)

    # lower local contrast in rain
    contrast_drop = float(cfg["contrast_drop"])
    comp = (comp - 0.5) * (1.0 - contrast_drop) + 0.5

    # cooler cast
    comp[:, :, 0] *= 1.03  # blue
    comp[:, :, 2] *= 0.95  # red

    # bloom around bright areas
    bloom_strength = float(cfg["bloom"])
    if bloom_strength > 0:
        luminance = (0.2126 * comp[:, :, 2] + 0.7152 * comp[:, :, 1] + 0.0722 * comp[:, :, 0])
        bright = np.clip((luminance - 0.55) / 0.45, 0.0, 1.0)
        bright3 = np.repeat(bright[:, :, None], 3, axis=2)
        glow = cv2.GaussianBlur(comp * bright3, (0, 0), 3.0)
        comp = comp + glow * bloom_strength

    # Sharpen rain presence a bit so streaks stay perceptible after darkening.
    rain_sharpness = float(cfg.get("rain_sharpness", 1.0))
    if rain_sharpness > 1.0:
        blur = cv2.GaussianBlur(comp, (0, 0), 1.1)
        comp = np.clip(comp + (comp - blur) * (rain_sharpness - 1.0), 0.0, 1.0)

    comp = tonal_lowlight_curve(
        comp,
        desaturation=float(cfg.get("desaturation", 0.0)),
        black_crush=float(cfg.get("black_crush", 0.0)),
    )

    comp = add_wet_road_reflection(comp, strength=float(cfg["rain_intensity"]) * 0.24)
    comp = add_ground_wetness(comp, strength=float(cfg.get("ground_wetness", 0.0)))
    comp = apply_vignette(comp, float(cfg["vignette"]))
    comp = add_sensor_artifacts(
        comp,
        seed=int(cfg["seed"]),
        noise_level=float(cfg.get("noise_level", 0.0)),
        ca_shift=int(cfg.get("ca_shift", 0)),
    )

    # Slightly lift clarity/brightness (used mainly for high severity request).
    clarity = float(cfg.get("clarity_boost", 1.0))
    if clarity > 1.0:
        comp = np.clip(comp * clarity, 0.0, 1.0)

    comp = np.clip(comp, 0.0, 1.0)
    return (comp * 255.0 + 0.5).astype(np.uint8)


def stable_seed(stem, severity):
    return zlib.crc32(f"{stem}|{severity}".encode("utf-8")) & 0xFFFFFFFF


def sample_severity_params(base_cfg, seed, width, height):
    """
    Deterministically sample severity parameters per (image, severity) using fixed seed.
    """
    rng = np.random.default_rng(seed)
    rain_intensity = float(rng.uniform(*base_cfg["rain_intensity_range"]))
    streak_length = int(rng.integers(base_cfg["streak_length_range"][0], base_cfg["streak_length_range"][1] + 1))
    brightness_scale = float(rng.uniform(*base_cfg["brightness_scale_range"]))
    gamma = float(rng.uniform(*base_cfg["gamma_range"]))
    angle_deg = float(rng.uniform(*base_cfg.get("angle_deg_range", (-10.0, 10.0))))

    # Convert density-like rain_intensity into streak count, scaled by resolution.
    mp_scale = (float(width) * float(height)) / 1_000_000.0
    streaks = int(max(80, round(rain_intensity * 1400.0 * mp_scale)))

    # Map intensity to blend alpha so heavier rain looks stronger.
    rain_alpha = float(clamp(0.15 + 0.70 * rain_intensity, 0.10, 0.85))

    cfg = dict(base_cfg)
    cfg["rain_intensity"] = rain_intensity
    cfg["length"] = streak_length
    cfg["streaks"] = streaks
    cfg["brightness_scale"] = brightness_scale
    cfg["gamma"] = gamma
    cfg["rain_alpha"] = rain_alpha
    cfg["angle_deg"] = angle_deg
    cfg["seed"] = int(seed)
    return cfg


def create_metadata_doc(entries, out_docx):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    doc.add_heading("Combined Condition Dataset Metadata: Rain + Low Light", level=1)
    doc.add_paragraph("This document lists sampled parameter settings and random seeds for each generated image.")
    doc.add_paragraph("All random parameters are generated using a fixed per-image seed for exact regeneration.")
    doc.add_paragraph("Weather type: Rain + Low light (combined adverse condition).")

    table = doc.add_table(rows=1, cols=13)
    hdr = table.rows[0].cells
    hdr[0].text = "image"
    hdr[1].text = "severity"
    hdr[2].text = "severity_label"
    hdr[3].text = "weather_type"
    hdr[4].text = "rain_intensity"
    hdr[5].text = "rain_alpha"
    hdr[6].text = "rain_streaks"
    hdr[7].text = "rain_length_px"
    hdr[8].text = "brightness_scale"
    hdr[9].text = "gamma"
    hdr[10].text = "rain_angle_deg"
    hdr[11].text = "noise_level"
    hdr[12].text = "seed"

    for e in entries:
        row = table.add_row().cells
        row[0].text = e["image"]
        row[1].text = e["severity"]
        row[2].text = e["severity_label"]
        row[3].text = "rain+lowlight"
        row[4].text = f"{e['rain_intensity']:.4f}"
        row[5].text = f"{e['rain_alpha']:.4f}"
        row[6].text = str(e["streaks"])
        row[7].text = str(e["length"])
        row[8].text = f"{e['brightness_scale']:.4f}"
        row[9].text = f"{e['gamma']:.4f}"
        row[10].text = f"{e['rain_angle_deg']:.4f}"
        row[11].text = f"{e['noise_level']:.4f}"
        row[12].text = str(e["seed"])

    os.makedirs(os.path.dirname(out_docx), exist_ok=True)
    doc.save(out_docx)


def get_sorted_input_files():
    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)]
    files.sort()
    return files


def run_yolo_annotations(files, model, render_previews, draw_on_main_images):
    selected_stems = []

    # speed: batch inference instead of per-image inference
    paths = [os.path.join(INPUT_DIR, f) for f in files]
    for i in tqdm(range(0, len(paths), YOLO_BATCH), desc="YOLO detect"):
        batch_paths = paths[i:i + YOLO_BATCH]
        results = model.predict(
            source=batch_paths,
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=YOLO_IMGSZ,
            batch=min(YOLO_BATCH, len(batch_paths)),
            device=YOLO_DEVICE,
            verbose=False,
        )

        for in_path, r in zip(batch_paths, results):
            fname = os.path.basename(in_path)
            stem = os.path.splitext(fname)[0]
            selected_stems.append(stem)

            h, w = r.orig_shape
            img = r.orig_img

            out_clear_img = os.path.join(CLEAR_IMG_OUT, stem + ".jpg")
            out_clear_lbl = os.path.join(CLEAR_LBL_OUT, stem + ".txt")

            yolo_lines = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), coco_id in zip(boxes, clss):
                    if coco_id not in COCO_TO_CUSTOM:
                        continue
                    custom_id = COCO_TO_CUSTOM[coco_id]
                    line = xyxy_to_yolo_line(float(x1), float(y1), float(x2), float(y2), w, h, custom_id)
                    if line:
                        yolo_lines.append(line)

            with open(out_clear_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            cv2.imwrite(out_clear_img, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if render_previews:
                out_clear_vis = os.path.join(VIS_CLEAR_OUT, stem + ".jpg")
                draw_boxes_from_yolo(out_clear_img, out_clear_lbl, out_clear_vis)
            if draw_on_main_images:
                draw_boxes_from_yolo(out_clear_img, out_clear_lbl, out_clear_img)

    return selected_stems


def synthesize_one(stem, render_previews, draw_on_main_images):
    clear_img_path = os.path.join(CLEAR_IMG_OUT, stem + ".jpg")
    clear_lbl_path = os.path.join(CLEAR_LBL_OUT, stem + ".txt")

    clear_bgr = cv2.imread(clear_img_path, cv2.IMREAD_COLOR)
    if clear_bgr is None or not os.path.exists(clear_lbl_path):
        return []

    h, w = clear_bgr.shape[:2]
    entries = []

    for sev, cfg in SEVERITIES.items():
        out_img_dir = os.path.join(SYN_IMG_OUT, sev)
        out_lbl_dir = os.path.join(SYN_LBL_OUT, sev)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        seed = stable_seed(stem, sev)
        sampled_cfg = sample_severity_params(cfg, seed, w, h)
        out_img_path = os.path.join(out_img_dir, stem + ".jpg")
        out_lbl_path = os.path.join(out_lbl_dir, stem + ".txt")

        # Resume-friendly behavior: skip expensive synthesis if both outputs already exist.
        if not (os.path.exists(out_img_path) and os.path.exists(out_lbl_path)):
            overlay = make_rain_overlay_rgba(w, h, seed, sampled_cfg)
            synth = run_realistic_composite(clear_bgr, overlay, sampled_cfg)
            cv2.imwrite(out_img_path, synth, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            shutil.copyfile(clear_lbl_path, out_lbl_path)

        if render_previews:
            out_vis_path = os.path.join(VIS_SYN_OUT, sev, stem + ".jpg")
            draw_boxes_from_yolo(out_img_path, out_lbl_path, out_vis_path)
        if draw_on_main_images:
            draw_boxes_from_yolo(out_img_path, out_lbl_path, out_img_path)

        entries.append({
            "image": f"rain_lowlight/{sev}/{stem}.jpg",
            "severity": sev,
            "severity_label": sampled_cfg["label"],
            "rain_intensity": sampled_cfg["rain_intensity"],
            "rain_alpha": sampled_cfg["rain_alpha"],
            "streaks": sampled_cfg["streaks"],
            "length": sampled_cfg["length"],
            "brightness_scale": sampled_cfg["brightness_scale"],
            "gamma": sampled_cfg["gamma"],
            "rain_angle_deg": sampled_cfg["angle_deg"],
            "noise_level": sampled_cfg.get("noise_level", 0.0),
            "seed": seed,
        })

    return entries


def parse_args():
    p = argparse.ArgumentParser(description="Fast YOLO annotation + realistic rain/lowlight synthesis")
    p.add_argument("--start-index", type=int, default=1, help="1-based start index")
    p.add_argument("--end-index", type=int, default=0, help="1-based inclusive end index; 0 means end")
    p.add_argument("--stage", choices=["all", "yolo", "synth"], default="all")
    p.add_argument("--synth-workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    p.add_argument("--render-box-previews", action="store_true", help="Write boxed preview images to visualizations/")
    p.add_argument("--draw-boxes-on-main-images", action="store_true", help="Burn boxes directly into images/* outputs")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs(render_previews=args.render_box_previews)
    write_classes_txt()

    files = get_sorted_input_files()
    total_files = len(files)
    if total_files == 0:
        print("No input images found in", INPUT_DIR)
        return

    start_idx = max(1, int(args.start_index))
    end_idx = int(args.end_index) if int(args.end_index) > 0 else total_files
    end_idx = min(end_idx, total_files)
    if start_idx > end_idx:
        raise ValueError(f"Invalid range: start_index={start_idx}, end_index={end_idx}")

    files = files[start_idx - 1:end_idx]
    print(f"Processing range {start_idx}-{end_idx} of {total_files} total images.")

    stems = [os.path.splitext(f)[0] for f in files]

    # 1) YOLO annotation stage for all selected images
    if args.stage in ("all", "yolo"):
        print("\n[1/3] Running batched YOLO detection and writing YOLO labels...")
        model = YOLO(resolve_yolo_weights())
        stems = run_yolo_annotations(
            files=files,
            model=model,
            render_previews=args.render_box_previews,
            draw_on_main_images=args.draw_boxes_on_main_images,
        )
    else:
        print("\n[1/3] Skipped YOLO stage (--stage synth).")

    if args.stage == "yolo":
        print("\n[2/3] Skipped synthesis (--stage yolo).")
        print("[3/3] Skipped metadata (--stage yolo).")
        print("\nDONE ✅")
        print("Clear images:", CLEAR_IMG_OUT)
        print("Clear labels:", CLEAR_LBL_OUT)
        print("classes.txt:", os.path.join(OUT_ROOT, "classes.txt"))
        return

    # 2) Weather synthesis stage
    print("\n[2/3] Running parallel rain+lowlight synthesis (low/mid/high)...")
    metadata_entries = []
    max_workers = max(1, int(args.synth_workers))
    try:
        executor_cm = ProcessPoolExecutor(max_workers=max_workers)
        use_chunksize = True
    except PermissionError:
        # Some sandboxed macOS environments block semaphore limits used by ProcessPool.
        executor_cm = ThreadPoolExecutor(max_workers=max_workers)
        use_chunksize = False

    with executor_cm as ex:
        if use_chunksize:
            it = ex.map(
                synthesize_one,
                stems,
                [args.render_box_previews] * len(stems),
                [args.draw_boxes_on_main_images] * len(stems),
                chunksize=8,
            )
        else:
            it = ex.map(
                synthesize_one,
                stems,
                [args.render_box_previews] * len(stems),
                [args.draw_boxes_on_main_images] * len(stems),
            )
        for entries in tqdm(it, total=len(stems), desc="Synthesize"):
            metadata_entries.extend(entries)

    # 3) Metadata
    print("\n[3/3] Writing metadata Word document...")
    create_metadata_doc(metadata_entries, META_OUT)

    print("\nDONE ✅")
    print("Clear images:", CLEAR_IMG_OUT)
    print("Clear labels:", CLEAR_LBL_OUT)
    print("Synthetic images:", SYN_IMG_OUT)
    print("Synthetic labels:", SYN_LBL_OUT)
    print("Metadata:", META_OUT)
    print("classes.txt:", os.path.join(OUT_ROOT, "classes.txt"))


if __name__ == "__main__":
    main()
