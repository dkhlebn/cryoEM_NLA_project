import os
import numpy as np
import mrcfile
import tntorch as tn
import torch


def load_mrc(path):
    with mrcfile.open(path, permissive=True) as m:
        arr = m.data.astype(np.float32)
    return arr


def save_mrc(path, arr):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(arr.astype(np.float32))


def patch_grid(h, w, patch, stride):
    ys = list(range(0, h-patch+1, stride))
    xs = list(range(0, w-patch+1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    if ys[-1] != h-patch:
        ys.append(h-patch)
    if xs[-1] != w-patch:
        xs.append(w-patch)
    return ys, xs


def hann2d(h, w):
    wy = np.hanning(h) if h>1 else np.ones(h)
    wx = np.hanning(w) if w>1 else np.ones(w)
    return np.outer(wy, wx)


def factors(n):
    k = int(np.ceil(np.sqrt(n)))
    a = k
    b = n//k
    if a*b != n:
        return n,1
    return a,b


def decompose_tensor(
    tensor,
    decomposition="tt",
    ranks=None,
    eps=None,
    device="cpu"
):
    device_torch = torch.device(device) if device and device != "cpu" else torch.device("cpu")
    t_input = torch.as_tensor(tensor, dtype=torch.float32, device=device_torch)

    if decomposition == "tt":
        if ranks is not None:
            tn_tensor = tn.Tensor(t_input, ranks_tt=ranks, device=device_torch)
        elif eps is not None:
            tn_tensor = tn.Tensor(t_input, eps=eps, device=device_torch)
        else:
            tn_tensor = tn.Tensor(t_input, ranks_tt=24, device=device_torch)

    if decomposition == "tucker":
        tn_tensor = tn.Tensor(t_input, ranks_tucker=ranks, device=device_torch)
    return tn_tensor.numpy().astype(np.float32)



def compute_tt_patch(patch, ranks=None, eps=None, device='cpu', decomposition="tt"):
    H, W = patch.shape
    h1,h2 = factors(H)
    w1,w2 = factors(W)
    pad_h = h1*h2 - H if h1*h2 > H else 0
    pad_w = w1*w2 - W if w1*w2 > W else 0
    p = patch
    if pad_h or pad_w:
        p = np.pad(p, ((0, pad_h), (0, pad_w)))
    modes = p.reshape((h1, h2, w1, w2))
    rec = decompose_tensor(modes, decomposition=decomposition,
                           ranks=ranks, eps=eps, device=device)
    rec_patch = rec.reshape((h1*h2, w1*w2))[:H, :W]
    return rec_patch.astype(np.float32)


def denoise_image(img, patch=256, stride=128, ranks=24, eps=None, device='cpu', decomposition="tt"):
    H,W = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    weight = np.zeros_like(img, dtype=np.float32)
    ys, xs = patch_grid(H,W,patch,stride)
    for y in ys:
        for x in xs:
            sub = img[y:y+patch, x:x+patch]
            rec = compute_tt_patch(sub, ranks=ranks, eps=eps,
                                   device=device, decomposition=decomposition)
            win = hann2d(rec.shape[0], rec.shape[1])
            out[y:y+rec.shape[0], x:x+rec.shape[1]] += rec * win
            weight[y:y+rec.shape[0], x:x+rec.shape[1]] += win
    mask = weight < 1e-8
    weight[mask] = 1.0
    out /= weight
    return out


def process_file(path, outdir, patch, stride, ranks, eps, device, decomposition="tt"):
    """Process a single .mrc for TT-SVD denoising"""
    arr = load_mrc(path)
    if arr.ndim == 2:
        den = denoise_image(arr, patch=patch, stride=stride,
                            ranks=ranks, eps=eps, device=device,
                            decomposition=decomposition)
        outname = os.path.join(
            outdir,
            os.path.splitext(os.path.basename(path))[0] + "_ttsvd.mrc"
        )
        save_mrc(outname, den)
        return


def process_stack(stack,
                  patch=64,
                  stride=32,
                  ranks=None,
                  eps=None,
                  device='cpu', decomposition="tt"):
    _, H, W = stack.shape
    ph = pw = patch

    out_stack = np.zeros_like(stack, dtype=np.float32)
    weight_stack = np.zeros_like(stack, dtype=np.float32)

    ys, xs = patch_grid(H, W, patch, stride)
    win2d = hann2d(ph, pw).astype(np.float32)

    for y in ys:
        for x in xs:
            ph_cur = min(ph, H - y)
            pw_cur = min(pw, W - x)

            patch3d = stack[:, y:y+ph_cur, x:x+pw_cur].astype(np.float32)

            pad_h = ph - ph_cur
            pad_w = pw - pw_cur
            if pad_h or pad_w:
                patch3d = np.pad(
                    patch3d,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="reflect"
                )

            rec3d = decompose_tensor(patch3d, decomposition=decomposition,
                                     ranks=ranks, eps=eps, device=device)
            rec3d = rec3d[:, :ph_cur, :pw_cur]
            win = win2d[:ph_cur, :pw_cur]

            out_stack[:, y:y+ph_cur, x:x+pw_cur] += rec3d * win[None, :, :]
            weight_stack[:, y:y+ph_cur, x:x+pw_cur] += win[None, :, :]

    weight_stack[weight_stack < 1e-8] = 1.0
    denoised_stack = out_stack / weight_stack

    return denoised_stack
