"""
WHT-Based Dual Color Image Watermarking Algorithm
Implements: "A Novel Dual Color Image Watermarking Algorithm Using Walsh–Hadamard Transform
with Difference-Based Embedding Positions" (Symmetry 2026, 18, 65)
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim_func


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BLOCK_SIZE = 4          # 4×4 blocks
BITS_PER_BLOCK = 4      # 4 watermark bits per block
T = 8                   # Quantization step (optimal per paper)
MU = 4.0                # Logistic map control parameter
X0 = 0.398              # Logistic map initial value

# 4×4 Hadamard matrix (Eq. 3 in paper)
H4 = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1]
], dtype=np.float64)


# ─────────────────────────────────────────────
# Walsh-Hadamard Transform
# ─────────────────────────────────────────────

def wht(block):
    """Apply WHT: F = (1/N) * H4 * block  (Eq. 5)"""
    return (1.0 / BLOCK_SIZE) * H4 @ block.astype(np.float64)


def iwht(F):
    """Apply inverse WHT: f = H4 * F  (Eq. 6)"""
    return H4 @ F


# ─────────────────────────────────────────────
# Logistic Chaotic Encryption
# ─────────────────────────────────────────────

def logistic_sequence(length, mu=MU, x0=X0):
    """Generate chaotic sequence via logistic map (Eq. 12)"""
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = mu * x * (1 - x)
        seq[i] = x
    return seq


def encrypt_bits(bits, mu=MU, x0=X0):
    """Scramble watermark bits using logistic chaotic map"""
    n = len(bits)
    seq = logistic_sequence(n, mu, x0)
    indices = np.argsort(seq)          # permutation from chaotic sequence
    encrypted = np.zeros(n, dtype=np.uint8)
    for new_pos, old_pos in enumerate(indices):
        encrypted[new_pos] = bits[old_pos]
    return encrypted, indices


def decrypt_bits(encrypted_bits, indices):
    """Inverse permutation to recover original bits"""
    n = len(encrypted_bits)
    bits = np.zeros(n, dtype=np.uint8)
    for new_pos, old_pos in enumerate(indices):
        bits[old_pos] = encrypted_bits[new_pos]
    return bits


# ─────────────────────────────────────────────
# Entropy-Based Block Selection
# ─────────────────────────────────────────────

def visual_entropy(block):
    """Visual entropy E1 = -Σ(pk * log(pk))  (Eq. 10)"""
    flat = block.flatten().astype(np.float64)
    # Build histogram over pixel value range
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    pk = hist / hist.sum()
    pk = pk[pk > 0]
    return -np.sum(pk * np.log(pk + 1e-12))


def edge_entropy(block):
    """Edge entropy E2 = Σ(pk * exp(1 - pk))  (Eq. 11)"""
    flat = block.flatten().astype(np.float64)
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    pk = hist / hist.sum()
    pk = pk[pk > 0]
    return np.sum(pk * np.exp(1 - pk))


def block_score(block):
    """BlockScore = E1 + E2"""
    return visual_entropy(block) + edge_entropy(block)


def get_blocks(channel, block_size=BLOCK_SIZE):
    """Divide channel into non-overlapping 4×4 blocks, return list of (row, col, block)"""
    h, w = channel.shape
    blocks = []
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            blk = channel[r:r+block_size, c:c+block_size].copy()
            blocks.append((r, c, blk))
    return blocks


def select_blocks(channel, n_blocks_needed):
    """Select n_blocks_needed lowest-entropy blocks (Section 3.1.2)"""
    blocks = get_blocks(channel)
    scored = [(block_score(blk), r, c) for (r, c, blk) in blocks]
    scored.sort(key=lambda x: x[0])   # ascending entropy
    selected = [(r, c) for (_, r, c) in scored[:n_blocks_needed]]
    return selected


# ─────────────────────────────────────────────
# Difference-Based Coefficient Pair Selection
# ─────────────────────────────────────────────

def get_coefficient_pairs(F):
    """
    Divide a 4×4 WHT coefficient matrix into 8 horizontal 1×2 pairs.
    Returns list of (row, col1, col2) for each pair.
    Section 3.1.3 / Figure 1.
    """
    pairs = []
    for row in range(BLOCK_SIZE):
        for col in range(0, BLOCK_SIZE, 2):
            pairs.append((row, col, col + 1))
    return pairs   # 8 pairs total


def select_best_pairs(F, n=4):
    """Select the 4 pairs with smallest absolute coefficient difference"""
    pairs = get_coefficient_pairs(F)
    diffs = [(abs(F[r, c1] - F[r, c2]), r, c1, c2) for (r, c1, c2) in pairs]
    diffs.sort(key=lambda x: x[0])
    return [(r, c1, c2) for (_, r, c1, c2) in diffs[:n]]


# ─────────────────────────────────────────────
# Watermark Embedding (Algorithm 1)
# ─────────────────────────────────────────────

def embed_bits_in_block(F, bits_4, T=T):
    """
    Embed 4 watermark bits into a WHT coefficient block.
    Modifies the 4 selected coefficient pairs in-place.
    Returns modified F and the positions used.
    """
    positions = select_best_pairs(F, n=4)
    F_mod = F.copy()
    for k, (r, c1, c2) in enumerate(positions):
        a, b = F_mod[r, c1], F_mod[r, c2]
        avg = (a + b) / 2.0
        diff = a - b
        wbit = bits_4[k]
        if wbit == 1:
            if diff <= T:
                F_mod[r, c1] = avg + T / 2
                F_mod[r, c2] = avg - T / 2
        else:  # wbit == 0
            if -diff <= T:
                F_mod[r, c1] = avg - T / 2
                F_mod[r, c2] = avg + T / 2
    return F_mod, positions


def embed_channel(channel, bits, block_positions):
    """Embed watermark bits into one channel at given block positions"""
    ch = channel.astype(np.float64)
    embedding_map = {}   # (r,c) -> list of pair positions used

    bit_idx = 0
    for (r, c) in block_positions:
        if bit_idx + BITS_PER_BLOCK > len(bits):
            break
        block = ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
        F = wht(block)
        bits_4 = bits[bit_idx:bit_idx + BITS_PER_BLOCK]
        F_mod, positions = embed_bits_in_block(F, bits_4, T)
        block_rec = iwht(F_mod)
        ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = block_rec
        embedding_map[(r, c)] = positions
        bit_idx += BITS_PER_BLOCK

    ch = np.clip(ch, 0, 255).astype(np.uint8)
    return ch, embedding_map


# ─────────────────────────────────────────────
# Watermark Extraction (Algorithm 2)
# ─────────────────────────────────────────────

def extract_bits_from_block(F, positions):
    """Extract watermark bits by comparing coefficient pairs (Algorithm 2)"""
    bits = []
    for (r, c1, c2) in positions:
        if F[r, c1] > F[r, c2]:
            bits.append(1)
        else:
            bits.append(0)
    return bits


def extract_channel(channel, block_positions, embedding_map):
    """Extract bits from one channel using stored embedding positions"""
    ch = channel.astype(np.float64)
    bits = []
    for (r, c) in block_positions:
        if (r, c) not in embedding_map:
            continue
        positions = embedding_map[(r, c)]
        block = ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
        F = wht(block)
        bits.extend(extract_bits_from_block(F, positions))
    return np.array(bits, dtype=np.uint8)


# ─────────────────────────────────────────────
# Image ↔ Bits Conversion
# ─────────────────────────────────────────────

def image_to_bits(img_array):
    """Convert image array to flat binary array (MSB first per byte)"""
    flat = img_array.flatten()
    bits = np.unpackbits(flat.astype(np.uint8))
    return bits


def bits_to_image(bits, shape):
    """Convert flat binary array back to image"""
    # Ensure correct length
    total = shape[0] * shape[1] * (shape[2] if len(shape) == 3 else 1) * 8
    bits = bits[:total]
    if len(bits) < total:
        bits = np.concatenate([bits, np.zeros(total - len(bits), dtype=np.uint8)])
    flat = np.packbits(bits.astype(np.uint8))
    return flat.reshape(shape)


# ─────────────────────────────────────────────
# Main Embed / Extract API
# ─────────────────────────────────────────────

def embed_watermark(cover_img, watermark_img):
    """
    Full embedding pipeline (Section 3.1).
    Returns watermarked image and metadata needed for extraction.
    """
    # Resize watermark to paper spec (32×32) if needed
    wm_resized = cv2.resize(watermark_img, (32, 32), interpolation=cv2.INTER_AREA)

    # Split channels
    cover_r, cover_g, cover_b = cv2.split(cover_img)
    wm_r, wm_g, wm_b = cv2.split(wm_resized)

    # Total bits needed per channel
    wm_bits_per_channel = 32 * 32 * 8
    n_blocks_needed = wm_bits_per_channel // BITS_PER_BLOCK  # blocks per channel

    metadata = {'wm_shape': wm_resized.shape, 'channels': {}}

    result_channels = []
    for ch_name, cover_ch, wm_ch in [('R', cover_r, wm_r),
                                      ('G', cover_g, wm_g),
                                      ('B', cover_b, wm_b)]:
        # Step 1: Select low-entropy blocks
        block_positions = select_blocks(cover_ch, n_blocks_needed)

        # Step 2: Convert watermark channel to bits
        wm_bits = image_to_bits(wm_ch)

        # Step 3: Logistic chaotic encryption
        encrypted_bits, perm_indices = encrypt_bits(wm_bits)

        # Step 4: Embed
        watermarked_ch, embedding_map = embed_channel(cover_ch, encrypted_bits, block_positions)

        result_channels.append(watermarked_ch)
        metadata['channels'][ch_name] = {
            'block_positions': block_positions,
            'embedding_map': embedding_map,
            'perm_indices': perm_indices,
        }

    watermarked = cv2.merge(result_channels)
    return watermarked, metadata


def extract_watermark(watermarked_img, metadata):
    """
    Full extraction pipeline (Section 3.2).
    Returns extracted watermark image.
    """
    wm_shape = metadata['wm_shape']
    wm_bits_per_channel = 32 * 32 * 8

    wm_r, wm_g, wm_b = cv2.split(watermarked_img)
    extracted_channels = []

    for ch_name, wm_ch in [('R', wm_r), ('G', wm_g), ('B', wm_b)]:
        ch_meta = metadata['channels'][ch_name]
        block_positions = ch_meta['block_positions']
        embedding_map = ch_meta['embedding_map']
        perm_indices = ch_meta['perm_indices']

        # Extract encrypted bits
        encrypted_bits = extract_channel(wm_ch, block_positions, embedding_map)

        # Ensure correct length
        encrypted_bits = encrypted_bits[:wm_bits_per_channel]
        if len(encrypted_bits) < wm_bits_per_channel:
            encrypted_bits = np.concatenate([
                encrypted_bits,
                np.zeros(wm_bits_per_channel - len(encrypted_bits), dtype=np.uint8)
            ])

        # Decrypt using inverse logistic permutation
        original_bits = decrypt_bits(encrypted_bits, perm_indices)

        # Convert bits back to image channel
        ch_img = bits_to_image(original_bits, (32, 32))
        extracted_channels.append(ch_img)

    extracted_wm = cv2.merge(extracted_channels)
    return extracted_wm


# ─────────────────────────────────────────────
# Metrics (Section 4)
# ─────────────────────────────────────────────

def compute_psnr(original, watermarked):
    """PSNR averaged across R,G,B channels (Eq. 13-14)"""
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    channel_psnr = []
    for i in range(3):
        mse = np.mean((original[:, :, i] - watermarked[:, :, i]) ** 2)
        if mse == 0:
            channel_psnr.append(100.0)
        else:
            max_val = np.max(original[:, :, i]) ** 2
            channel_psnr.append(10 * np.log10(max_val / mse))
    return np.mean(channel_psnr)


def compute_ssim(original, watermarked):
    """SSIM between original and watermarked image (Eq. 15)"""
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    wm_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    score = ssim_func(orig_gray, wm_gray, data_range=255)
    return float(score)


def compute_nc(original_wm, extracted_wm):
    """Normalized Correlation between watermarks (Eq. 16)"""
    w = original_wm.astype(np.float64)
    w_prime = extracted_wm.astype(np.float64)
    numerator = np.sum(w * w_prime)
    denominator = np.sqrt(np.sum(w ** 2)) * np.sqrt(np.sum(w_prime ** 2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_ber(original_wm, extracted_wm):
    """Bit Error Rate between original and extracted watermark (Eq. 17)"""
    orig_bits = image_to_bits(original_wm)
    ext_bits = image_to_bits(extracted_wm)
    min_len = min(len(orig_bits), len(ext_bits))
    errors = np.sum(orig_bits[:min_len] != ext_bits[:min_len])
    return float(errors / min_len)


# ─────────────────────────────────────────────
# Attack Simulation
# ─────────────────────────────────────────────

def apply_attack(img, attack_type, **kwargs):
    """Apply various attacks to a watermarked image"""
    result = img.copy()

    if attack_type == 'gaussian_noise':
        var = kwargs.get('variance', 0.01)
        noise = np.random.normal(0, var * 255, img.shape).astype(np.float64)
        result = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    elif attack_type == 'jpeg_compression':
        quality = kwargs.get('quality', 50)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, enc = cv2.imencode('.jpg', img, encode_params)
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    elif attack_type == 'cropping':
        percent = kwargs.get('percent', 0.1)
        h, w = img.shape[:2]
        crop_h = int(h * percent)
        crop_w = int(w * percent)
        result = img.copy()
        result[h - crop_h:, :] = 0
        result[:, w - crop_w:] = 0

    elif attack_type == 'rotation':
        angle = kwargs.get('angle', 5)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img, M, (w, h))

    elif attack_type == 'scaling':
        scale = kwargs.get('scale', 0.9)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        result = cv2.resize(scaled, (w, h))

    return result


# ─────────────────────────────────────────────
# Utility: Image encode/decode helpers
# ─────────────────────────────────────────────

def numpy_to_base64(img_array):
    """Convert numpy BGR image to base64 PNG string"""
    success, buffer = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_numpy(b64_str):
    """Convert base64 image string to numpy BGR array"""
    img_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def file_bytes_to_numpy(file_bytes):
    """Convert uploaded file bytes to numpy BGR array"""
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def prepare_cover_image(img, target_size=256):
    """Resize cover image to 256×256 as per paper"""
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
