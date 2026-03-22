import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import librosa
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from PIL import Image
from scipy.ndimage import label as nd_label
from torchvision import transforms


# Model + classes are fixed per user requirement.
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "bat_best.pth"
CLASSES_PATH = SCRIPT_DIR / "21_species.json"

# Inference constants mirrored from 20th_march_testing_code.py.
ACOUSTIC_DIM = 4
FEAT_DIM = 1280
ATT_DIM = 512
N_HEADS = 8
CHUNK_DUR_SEC = 1.0
MODEL_THRESHOLD_DEFAULT = 0.21

# CV gate defaults
CV_ENERGY_PERCENTILE = 72
CV_MIN_ACTIVE_RATIO = 0.04
CV_MIN_FREQ_RANGE_KHZ = 1.5
CV_MIN_COMPONENT_AREA = 80
CV_MIN_CONTOUR_LENGTH = 12
CV_FLATNESS_MAX = 0.92
FREQ_MIN_KHZ = 10.0
FREQ_MAX_KHZ = 250.0

# Denoise defaults
PCEN_GAIN = 0.2
PCEN_BIAS = 9
PCEN_POWER = 0.3
PCEN_TC = 1.0
SUB_PERCENTILE = 30
DB_MIN = -55

SPECIES_DATA = {
    "Chaerephon plicatus": (218.0, 207.2, 0.999, 0.706),
    "Hipposideros armiger": (140.8, 67.2, 0.997, 0.808),
    "Hipposideros ater": (214.4, 128.4, 1.000, 0.934),
    "Hipposideros durgadasi": (230.8, 137.8, 0.998, 0.766),
    "Hipposideros galeritus": (151.6, 140.0, 0.955, 0.750),
    "Hipposideros hypophyllus": (216.2, 205.4, 0.984, 0.755),
    "Hipposideros lankadiva": (215.6, 203.6, 0.997, 0.765),
    "Hipposideros pomona": (161.6, 66.6, 0.985, 0.806),
    "Hipposideros speoris": (193.4, 182.8, 0.992, 0.775),
    "Lyroderma lyra Megaderma lyra": (222.4, 210.0, 1.000, 0.761),
    "Megaderma spasma": (71.0, 60.6, 1.000, 0.839),
    "Pipistrellus ceylonicus": (73.0, 62.4, 1.000, 0.767),
    "Pipistrellus coromandra": (73.6, 63.2, 0.997, 0.764),
    "Pipistrellus tenuis": (66.6, 55.8, 0.988, 0.814),
    "Rhinolophus beddomei": (183.6, 133.2, 0.997, 0.706),
    "Rhinolophus lepidus": (130.6, 90.4, 0.987, 0.758),
    "Rhinolophus rouxii": (191.6, 84.8, 0.970, 0.852),
    "Rhinopoma hardwickii": (28.4, 18.0, 1.000, 0.831),
    "Scotophilus heathi": (249.6, 239.2, 0.989, 0.714),
    "Tadarida aegyptiaca": (218.4, 207.6, 1.000, 0.681),
    "Taphozous melanopogon": (136.4, 119.2, 0.897, 0.869),
}

GLOBAL_STATS = {
    "peak_freq": {"max": 250.0},
    "bandwidth": {"max": 250.0},
    "harmonic": {"mean": 0.985, "std": 0.025},
    "qcf": {"mean": 0.778, "std": 0.062},
}


def _build_plasma_lut() -> np.ndarray:
    lut = (cm.get_cmap("plasma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return lut[:, ::-1].copy()


PLASMA_LUT = _build_plasma_lut()
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class MultiHeadAcousticAttention(nn.Module):
    def __init__(self, feat_dim, acoustic_dim, n_heads, dropout=0.08):
        super().__init__()
        assert feat_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(acoustic_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(acoustic_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x, acoustic):
        bsz, n_cls = x.size(0), acoustic.size(0)
        heads, dim = self.n_heads, self.head_dim
        q = self.q_proj(x).view(bsz, heads, dim).permute(1, 0, 2)
        k = self.k_proj(acoustic).view(n_cls, heads, dim).permute(1, 0, 2)
        v = self.v_proj(acoustic).view(n_cls, heads, dim).permute(1, 0, 2)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        attn = self.drop(attn)
        out = torch.bmm(attn, v).permute(1, 0, 2).contiguous().view(bsz, -1)
        return self.norm(x + self.out_proj(out)), attn.mean(0)


class FrequencyGate(nn.Module):
    def __init__(self, acoustic_dim, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(acoustic_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, acoustic_mean):
        return x * self.net(acoustic_mean)


class FrequencyDisambiguator(nn.Module):
    def __init__(self, feat_dim, freq_prior_dim=2):
        super().__init__()
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_prior_dim, 128),
            nn.GELU(),
            nn.Linear(128, feat_dim // 2),
        )

    def forward(self, x, freq_priors):
        f_ctx = self.freq_proj(freq_priors).mean(0)
        return f_ctx.unsqueeze(0).expand(x.size(0), -1)


class BatNet(nn.Module):
    def __init__(self, n_classes, acoustic_vecs, freq_priors, dropout=0.4):
        super().__init__()
        self.backbone = self._build_backbone()
        self.reduce = nn.Sequential(
            nn.Linear(FEAT_DIM, ATT_DIM),
            nn.GELU(),
            nn.LayerNorm(ATT_DIM),
            nn.Dropout(dropout * 0.4),
        )
        self.skip = nn.Linear(FEAT_DIM, ATT_DIM)
        self.freq_gate = FrequencyGate(ACOUSTIC_DIM, ATT_DIM)
        self.attn1 = MultiHeadAcousticAttention(ATT_DIM, ACOUSTIC_DIM, N_HEADS, dropout=0.08)
        self.attn2 = MultiHeadAcousticAttention(ATT_DIM, ACOUSTIC_DIM, N_HEADS, dropout=0.06)
        self.disambig = FrequencyDisambiguator(ATT_DIM, freq_prior_dim=2)
        cls_in = ATT_DIM + ATT_DIM // 2
        self.clf = nn.Sequential(
            nn.Linear(cls_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, n_classes),
        )
        self.register_buffer("acoustic_vecs", acoustic_vecs)
        self.register_buffer("freq_priors", freq_priors)

    @staticmethod
    def _build_backbone():
        try:
            backbone = EfficientNet.from_pretrained("efficientnet-b0")
        except Exception:
            backbone = EfficientNet.from_name("efficientnet-b0")
        backbone._fc = nn.Identity()
        backbone._dropout = nn.Dropout(p=0.0)
        return backbone

    def forward(self, x):
        raw = self.backbone(x)
        feat = self.reduce(raw) + self.skip(raw)
        feat = self.freq_gate(feat, self.acoustic_vecs.mean(0, keepdim=True))
        feat, _ = self.attn1(feat, self.acoustic_vecs)
        feat, _ = self.attn2(feat, self.acoustic_vecs)
        ctx = self.disambig(feat, self.freq_priors)
        return self.clf(torch.cat([feat, ctx], dim=1))


_device = None
_model = None
_classes = None


def build_acoustic_vectors(classes):
    vecs = []
    for cls in classes:
        sd = SPECIES_DATA.get(cls)
        if sd is None:
            vecs.append([0.5, 0.3, 0.0, 0.0])
            continue
        peak, bw, harm, qcf = sd
        p_peak = peak / GLOBAL_STATS["peak_freq"]["max"]
        p_bw = min(bw / GLOBAL_STATS["bandwidth"]["max"], 1.0)
        p_harm = float(np.tanh((harm - GLOBAL_STATS["harmonic"]["mean"]) / (GLOBAL_STATS["harmonic"]["std"] + 1e-8)))
        p_qcf = float(np.tanh((qcf - GLOBAL_STATS["qcf"]["mean"]) / (GLOBAL_STATS["qcf"]["std"] + 1e-8)))
        vecs.append([p_peak, p_bw, p_harm, p_qcf])
    return torch.tensor(vecs, dtype=torch.float32)


def build_freq_priors(classes):
    rows = []
    for cls in classes:
        sd = SPECIES_DATA.get(cls, (60.0, 50.0, 0.99, 0.78))
        rows.append([sd[0] / 250.0, sd[1] / 250.0])
    return torch.tensor(rows, dtype=torch.float32)


def _ensure_model_loaded():
    global _device, _model, _classes

    if _model is not None and _classes is not None and _device is not None:
        return _model, _classes, _device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(str(MODEL_PATH), map_location=_device, weights_only=False)

    if "classes" in raw:
        _classes = raw["classes"]
    else:
        with open(CLASSES_PATH, "r", encoding="utf-8") as jf:
            _classes = json.load(jf)

    if "acoustic_vecs" in raw and "freq_priors" in raw:
        acoustic_vecs = raw["acoustic_vecs"].float()
        freq_priors = raw["freq_priors"].float()
    else:
        acoustic_vecs = build_acoustic_vectors(_classes)
        freq_priors = build_freq_priors(_classes)

    _model = BatNet(len(_classes), acoustic_vecs, freq_priors).to(_device).eval()

    if "model_state_dict" in raw:
        _model.load_state_dict(raw["model_state_dict"])
    else:
        _model.load_state_dict(raw)

    return _model, _classes, _device


def _classify_shape(freq_profile):
    valid = [f for f in freq_profile if f is not None]
    if len(valid) < 4:
        return "Unknown"
    f = np.array(valid, dtype=float)
    bw = f.max() - f.min()
    x = np.linspace(0, 1, len(f))
    sl = np.polyfit(x, f, 1)[0]
    cu = np.polyfit(x, f, 2)[0]
    rs = float(np.std(f - np.polyval(np.polyfit(x, f, 1), x)))
    if bw < 2.0:
        return "CF-Exact"
    if bw < 5.0:
        return "CF-Nearly"
    if rs < 0.8 and abs(sl) < 3:
        return "FM-Linear"
    if sl < -1.0 and cu < -0.5:
        return "FM-Descending"
    if sl > 1.0 and cu > 0.5:
        return "FM-Ascending"
    if sl < -0.5:
        return "FM-Descending"
    if sl > 0.5:
        return "FM-Ascending"
    return "FM-Linear"


def cv_validate_spectrogram(png_path: Path):
    try:
        bgr = cv2.imread(str(png_path))
        if bgr is None:
            return False, "BLANK", "None"

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        if gray.max() < 5:
            return False, "BLANK", "None"

        thresh = np.percentile(gray, CV_ENERGY_PERCENTILE)
        binary = gray > thresh

        col_active = binary.any(axis=0)
        ar = float(col_active.mean())
        if ar < CV_MIN_ACTIVE_RATIO:
            return False, "SILENT", "None"

        active_cols = np.where(col_active)[0]
        peak_rows = np.argmax(gray[:, active_cols], axis=0)
        freqs_arr = FREQ_MAX_KHZ - (peak_rows / h) * (FREQ_MAX_KHZ - FREQ_MIN_KHZ)

        dr = float(freqs_arr.max() - freqs_arr.min()) if len(freqs_arr) > 0 else 0.0
        if dr < CV_MIN_FREQ_RANGE_KHZ and ar < 0.25:
            return False, "FLAT_NOISE", "None"

        labeled, nc = nd_label(binary.astype(np.uint8))
        if nc == 0:
            return False, "NO_COMPONENT", "None"
        comp_sizes = np.bincount(labeled.ravel())
        max_area = int(comp_sizes[1:].max()) if len(comp_sizes) > 1 else 0
        if max_area < CV_MIN_COMPONENT_AREA:
            return False, "NO_COMPONENT", "None"

        edges = cv2.Canny((binary * 255).astype(np.uint8), 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_arc = max((cv2.arcLength(c, False) for c in contours), default=0.0)
        if max_arc < CV_MIN_CONTOUR_LENGTH:
            return False, "NO_ARC", "None"

        row_e = gray.mean(axis=1)
        row_n = row_e / (row_e.max() + 1e-8)
        flt = float(row_n.std())
        if flt < (1.0 - CV_FLATNESS_MAX):
            return False, "FLAT_SPECTRUM", "None"

        if contours:
            lg = max(contours, key=lambda c: cv2.arcLength(c, False))
            _, _, wb, hb = cv2.boundingRect(lg)
            asp = wb / (hb + 1e-8)
            if asp > 15 and hb < 6:
                return False, "H_BAR", "None"
            if asp < 0.07 and wb < 6:
                return False, "V_BAR", "None"

        fp = [None] * w
        for idx, col in enumerate(active_cols):
            fp[col] = float(freqs_arr[idx])

        shape = _classify_shape(fp)
        if shape == "Unknown" and ar < 0.08:
            return False, "AMBIGUOUS", "Unknown"

        return True, "OK", shape
    except Exception as ex:
        return False, f"CV_ERROR: {ex}", "None"


def _get_tuned_db(s: np.ndarray, db_min: float = DB_MIN) -> np.ndarray:
    return np.clip(librosa.amplitude_to_db(s, ref=np.max), db_min, 0.0)


def _to_bgr(u8: np.ndarray, w: int = 900, h: int = 600) -> np.ndarray:
    bgr = PLASMA_LUT[np.flipud(u8)]
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def _denoise_none(tuned_db: np.ndarray, db_min: float) -> np.ndarray:
    u8 = ((tuned_db - db_min) / (-db_min) * 255).astype(np.uint8)
    return _to_bgr(u8)


def _denoise_subtract(tuned_db: np.ndarray) -> np.ndarray:
    energy = tuned_db.mean(axis=0)
    thresh = np.percentile(energy, SUB_PERCENTILE)
    nf = tuned_db[:, energy <= thresh]
    if nf.shape[1] == 0:
        u8 = (((tuned_db - tuned_db.min()) / (tuned_db.max() - tuned_db.min() + 1e-8)) * 255).astype(np.uint8)
        return _to_bgr(u8)
    noise = np.median(nf, axis=1, keepdims=True)
    clean = np.maximum(tuned_db - noise, 0.0)
    d_max = clean.max()
    if d_max < 1e-8:
        u8 = (((tuned_db - tuned_db.min()) / (tuned_db.max() - tuned_db.min() + 1e-8)) * 255).astype(np.uint8)
        return _to_bgr(u8)
    return _to_bgr(np.clip(clean / d_max * 255, 0, 255).astype(np.uint8))


def _denoise_pcen(tuned_db: np.ndarray, sr: int) -> np.ndarray:
    s_lin = librosa.db_to_amplitude(tuned_db)
    pcen = librosa.pcen(
        s_lin * (2**31),
        sr=sr,
        hop_length=512,
        gain=PCEN_GAIN,
        bias=PCEN_BIAS,
        power=PCEN_POWER,
        time_constant=PCEN_TC,
        eps=1e-6,
    ).astype(np.float32)
    lo, hi = pcen.min(), pcen.max()
    u8 = ((pcen - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return _to_bgr(u8)


def _denoise_full(tuned_db: np.ndarray, sr: int) -> np.ndarray:
    energy = tuned_db.mean(axis=0)
    thresh = np.percentile(energy, SUB_PERCENTILE)
    nf = tuned_db[:, energy <= thresh]
    if nf.shape[1] > 0:
        noise = np.median(nf, axis=1, keepdims=True)
        after_sub = np.maximum(tuned_db - noise, 0.0)
    else:
        after_sub = tuned_db

    s_lin = librosa.db_to_amplitude(after_sub)
    pcen = librosa.pcen(
        s_lin * (2**31),
        sr=sr,
        hop_length=512,
        gain=PCEN_GAIN,
        bias=PCEN_BIAS,
        power=PCEN_POWER,
        time_constant=PCEN_TC,
        eps=1e-6,
    ).astype(np.float32)
    lo, hi = pcen.min(), pcen.max()
    u8 = ((pcen - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return _to_bgr(u8)


def render_chunk_to_png(chunk: np.ndarray, sr: int, out_path: Path, denoise_mode: str = "full"):
    try:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < 1e-7:
            return False, "silent chunk"

        n_fft, hop = 2048, 512
        d = librosa.stft(chunk, n_fft=n_fft, hop_length=hop)
        s = np.abs(d)
        tuned_db = _get_tuned_db(s, DB_MIN)

        dyn_range = float(tuned_db.max() - tuned_db.min())
        if dyn_range < 10.0:
            return False, f"flat chunk (dyn={dyn_range:.1f} dB)"

        if denoise_mode == "none":
            bgr = _denoise_none(tuned_db, DB_MIN)
        elif denoise_mode == "subtract":
            bgr = _denoise_subtract(tuned_db)
        elif denoise_mode == "pcen":
            bgr = _denoise_pcen(tuned_db, sr)
        else:
            bgr = _denoise_full(tuned_db, sr)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)
        return True, f"ok dyn={dyn_range:.1f} mode={denoise_mode}"
    except Exception as ex:
        return False, str(ex)


def split_audio_to_chunks(audio_path: Path):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    if not np.any(y):
        return [], sr
    spc = max(1, int(CHUNK_DUR_SEC * sr))
    n_chunks = max(1, int(np.ceil(len(y) / spc)))
    chunks = []
    for i in range(n_chunks):
        s = i * spc
        chunk = y[s:s + spc]
        if len(chunk) < spc:
            chunk = np.pad(chunk, (0, spc - len(chunk)), mode="constant")
        chunks.append(chunk)
    return chunks, sr


def predict_single_image(img_path: Path) -> np.ndarray:
    model, classes, device = _ensure_model_loaded()
    img = Image.open(str(img_path)).convert("RGB")
    tensor = EVAL_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    if len(probs) > len(classes):
        probs = probs[:len(classes)]
    return probs


def compute_weighted_scores(chunk_summary: list, classes: list, threshold: float):
    if not chunk_summary:
        return []

    n_valid = len(chunk_summary)
    sp_stats = {sp: {"probs": [], "n_det": 0} for sp in classes}

    for ck in chunk_summary:
        probs_arr = ck["probs"]
        det_set = {sp for sp, _ in ck["detections"]}
        for i, sp in enumerate(classes):
            p = float(probs_arr[i])
            sp_stats[sp]["probs"].append(p)
            if sp in det_set:
                sp_stats[sp]["n_det"] += 1

    results = []
    for sp, st in sp_stats.items():
        probs_arr = np.array(st["probs"], dtype=np.float32)
        denom = probs_arr.sum()
        ws = 0.0 if denom < 1e-8 else float((probs_arr * probs_arr).sum() / denom)
        n_det = st["n_det"]
        det_rate = n_det / max(n_valid, 1)
        results.append({
            "species": sp,
            "weighted_score": round(ws, 4),
            "n_detections": n_det,
            "detection_rate": round(det_rate, 4),
            "max_conf": round(float(probs_arr.max()), 4),
            "mean_conf": round(float(probs_arr.mean()), 4),
            "above_threshold": n_det > 0,
        })

    results.sort(key=lambda x: -x["weighted_score"])
    return results


def predict_audio_file(audio_path: str, threshold: float = MODEL_THRESHOLD_DEFAULT, denoise_mode: str = "full", apply_cv_filter: bool = False, tmp_dir: str = None):
    try:
        _, classes, _ = _ensure_model_loaded()
        base_tmp = Path(tmp_dir) if tmp_dir else (SCRIPT_DIR / "tmp_spectrograms")
        base_tmp.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        chunks, sr = split_audio_to_chunks(Path(audio_path))
        if not chunks:
            return {"error": "Could not load audio"}

        n = len(chunks)
        spec_paths = []
        all_vectors = []
        valid_idx = []
        cv_reject_log = []

        for i, chunk in enumerate(chunks):
            out_png = base_tmp / f"chunk_{ts}_{i:04d}.png"
            ok, msg = render_chunk_to_png(chunk, sr, out_png, denoise_mode)
            if not ok:
                cv_reject_log.append((i + 1, msg, "pre-render"))
                continue

            if apply_cv_filter:
                is_valid, reason, shape = cv_validate_spectrogram(out_png)
                if not is_valid:
                    cv_reject_log.append((i + 1, reason, shape))
                    try:
                        out_png.unlink()
                    except Exception:
                        pass
                    continue

            try:
                probs = predict_single_image(out_png)
                all_vectors.append(probs)
                valid_idx.append(i)
                spec_paths.append(str(out_png))
            except Exception as ex:
                cv_reject_log.append((i + 1, f"infer_error: {ex}", "None"))
                try:
                    out_png.unlink()
                except Exception:
                    pass

        if not all_vectors:
            return {
                "error": "No valid chunks produced after spectrogram generation and CV filtering",
                "cv_reject_log": cv_reject_log,
                "n_chunks_total": n,
            }

        chunk_summary = []
        for ci, (vi, probs) in enumerate(zip(valid_idx, all_vectors)):
            top_idx = int(np.argmax(probs))
            top_sp = classes[top_idx]
            top_conf = float(probs.max())
            top5 = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
            chunk_summary.append({
                "chunk": vi + 1,
                "top_species": top_sp,
                "top_conf": top_conf,
                "top5": [(classes[idx], float(p)) for idx, p in top5],
                "probs": probs,
                "spec_path": spec_paths[ci],
                "detections": [(classes[idx], float(p)) for idx, p in enumerate(probs) if p >= threshold],
            })

        weighted = compute_weighted_scores(chunk_summary, classes, threshold)

        return {
            "n_chunks_total": n,
            "n_chunks_valid": len(all_vectors),
            "n_cv_rejected": n - len(all_vectors),
            "sr": sr,
            "duration_s": len(chunks) * CHUNK_DUR_SEC,
            "chunk_summary": chunk_summary,
            "spec_paths": spec_paths,
            "cv_reject_log": cv_reject_log,
            "weighted_scores": weighted,
            "denoise_mode": denoise_mode,
            "error": None,
        }
    except Exception:
        return {"error": traceback.format_exc()}


def generate_preview_spectrogram(audio_path: str, output_path: str, denoise_mode: str = "full"):
    chunks, sr = split_audio_to_chunks(Path(audio_path))
    if not chunks:
        return False
    ok, _ = render_chunk_to_png(chunks[0], sr, Path(output_path), denoise_mode)
    return ok


def weighted_scores_to_species_list(weighted_scores: list):
    species = []
    for item in weighted_scores:
        if not item.get("above_threshold"):
            continue
        species.append({
            "species": item["species"].replace(" ", "_"),
            "confidence": round(float(item["weighted_score"]) * 100.0, 1),
        })
    return species
