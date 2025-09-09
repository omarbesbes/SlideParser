"""
Automated color-distinctification pipeline.
Dependencies: opencv-python, numpy, Pillow
pip install opencv-python numpy Pillow
"""

import cv2
import numpy as np
from PIL import Image
import os

def read_image(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    # If image has alpha, drop it for processing
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    return bgr

def save_bgr(path, bgr):
    cv2.imwrite(path, bgr)

def to_lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

def sample_pixels(bgr, max_samples=10000):
    h,w,_ = bgr.shape
    flat = bgr.reshape(-1,3)
    if len(flat) > max_samples:
        idx = np.random.choice(len(flat), max_samples, replace=False)
        flat = flat[idx]
    lab = cv2.cvtColor(flat.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    return lab

def kmeans_centers_lab(lab_pixels, k=6, attempts=3):
    # lab_pixels: Nx3 float32
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    lab_pixels32 = lab_pixels.astype(np.float32)
    _, labels, centers = cv2.kmeans(lab_pixels32, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return centers # kx3

def min_pairwise_lab_dist(centers):
    # centers: kx3
    if centers.shape[0] < 2:
        return 0.0
    dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            d = np.linalg.norm(centers[i]-centers[j])
            dists.append(d)
    return float(np.min(dists)) if dists else 0.0

# Transformations
def boost_saturation(bgr, scale=1.6):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def hue_shift(bgr, degrees=15):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # OpenCV H range is [0,179]
    shift = degrees * (179/360.0)
    hsv[...,0] = (hsv[...,0] + shift) % 179
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def clahe_on_l_channel(bgr, clip_limit=3.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def posterize(bgr, bits=4):
    # Reduce color depth to increase separability
    shift = 8 - bits
    out = np.right_shift(bgr, shift)
    out = np.left_shift(out, shift)
    return out

def remap_to_palette(bgr, palette_rgb=None):
    """
    Map every pixel to nearest color in palette (in Lab space) to get highly distinct colors.
    palette_rgb: list of RGB tuples (0-255)
    """
    if palette_rgb is None:
        # a compact high-contrast palette (R,G,B)
        palette_rgb = [
            (0,0,0), (255,255,255), (255,0,0), (0,255,0),
            (0,0,255), (255,255,0), (255,0,255), (0,255,255)
        ]
    palette_bgr = np.array([(c[2], c[1], c[0]) for c in palette_rgb], dtype=np.uint8)
    lab_palette = cv2.cvtColor(palette_bgr.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    h,w,_ = bgr.shape
    flat = bgr.reshape(-1,3)
    lab_flat = cv2.cvtColor(flat.reshape(-1,1,3), cv2.COLOR_BGR2LAB).reshape(-1,3).astype(np.float32)
    # compute nearest palette color for each pixel (brute force)
    dists = np.linalg.norm(lab_flat[:,None,:] - lab_palette[None,:,:], axis=2) # N x P
    idx = np.argmin(dists, axis=1)
    mapped = palette_bgr[idx].reshape(h,w,3)
    return mapped

# Pipeline wrapper
def make_variants_and_score(bgr, k=6, sample_pixels_count=5000):
    variants = {}
    variants['orig'] = bgr
    variants['sat_up'] = boost_saturation(bgr, scale=1.6)
    variants['hue_shift_pos'] = hue_shift(bgr, degrees=18)
    variants['hue_shift_neg'] = hue_shift(bgr, degrees=-18)
    variants['clahe'] = clahe_on_l_channel(bgr, clip_limit=3.0)
    variants['posterize'] = posterize(bgr, bits=4)
    variants['palette_remap'] = remap_to_palette(bgr)

    scores = {}
    centers_dict = {}
    for name, img in variants.items():
        lab_sample = sample_pixels(img, max_samples=sample_pixels_count)
        try:
            centers = kmeans_centers_lab(lab_sample, k=k)
            score = min_pairwise_lab_dist(centers)
        except Exception as e:
            # fallback small sample/k
            centers = kmeans_centers_lab(lab_sample, k=min(k,3))
            score = min_pairwise_lab_dist(centers)
        scores[name] = score
        centers_dict[name] = centers

    # sort variants by score (descending)
    sorted_names = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    return variants, scores, sorted_names, centers_dict

def pick_best_variant_and_save(inpath, out_dir='out_variants', k=6):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    bgr = read_image(inpath)
    variants, scores, sorted_names, centers = make_variants_and_score(bgr, k=k)
    # Save top 3 variants and return best
    results = []
    for i,name in enumerate(sorted_names[:3]):
        outpath = os.path.join(out_dir, f"{i+1:02d}_{name}.png")
        save_bgr(outpath, variants[name])
        results.append((name, scores[name], outpath))
    # Also return full ranking
    return results, scores, sorted_names

# Example usage:
if __name__ == "__main__":
    inpath = "input_chart.png"
    best_results, scores, ranking = pick_best_variant_and_save(inpath, out_dir='variants_out', k=6)
    print("Ranking:", ranking)
    print("Scores:", scores)
    print("Top saved files:", best_results)