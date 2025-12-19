import subprocess
from collections import defaultdict
import numpy as np


def delete_job(job_uid, project_uid):
    cmd = f"delete_job(project_uid='{project_uid}', job_uid='{job_uid}')"
    subprocess.run(
        ["cryosparcm", "cli", cmd],
        check=True
    )

def _get_micro_params(entry):
    return (entry.get("EMPIAR_ID"), float(entry.get("raw_px_size_ang")),
            int(entry.get("acc_volt_kV")), float(entry.get("sph_abb_mm")),
            float(entry.get("tot_ex_e_angsq"))
           )


def _get_particle_diams(entry):
    return int(entry.get("min_ptcl_d_ang")), int(entry.get("max_ptcl_d_ang"))


def _get_picker_params(entry, margin=0.3, appx=16):
    min_d, max_d = _get_particle_diams(entry)
    box = (1 + margin) * ((min_d + max_d) / (2 * entry.get("raw_px_size_ang")))
    box_size = int(np.ceil(box / appx) * appx)
    # rule-of-thumb
    if 50 <= box_size <= 120:
        return int(box_size), int(box_size)

    bin_fct = int(np.ceil(box_size / 120))
    bin_fct = np.clip(bin_fct, 1, 8) #hard-coded rule-of-thumb
    fcrop = box_size // bin_fct
    while bin_fct > 1 and fcrop < 505:
        bin_fct -= 1
        fcrop = box_size // bin_fct
    return int(box_size), int(fcrop)


def _get_rows_per_class(dset):
    class_to_rows = defaultdict(list)
    for row in dset.rows():
        class_id = int(row["alignments2D/class"])
        class_to_rows[class_id].append(row)
    return class_to_rows


def _crop_with_padding(img, cx, cy, box):
    half = box // 2
    H, W = img.shape

    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    patch = np.zeros((box, box), dtype=img.dtype)

    ix0 = max(0, x0)
    ix1 = min(W, x1)
    iy0 = max(0, y0)
    iy1 = min(H, y1)

    px0 = ix0 - x0
    px1 = px0 + (ix1 - ix0)
    py0 = iy0 - y0
    py1 = py0 + (iy1 - iy0)

    patch[py0:py1, px0:px1] = img[iy0:iy1, ix0:ix1]
    return patch


def _get_particle_patch(row, micrographs):
    path = row["blob/path"]
    stack = micrographs[path]

    # handle stack vs micrograph
    if stack.ndim == 3:
        idx = int(row["blob/idx"])
        img = stack[idx]
    else:
        img = stack

    H, W = img.shape

    cx = row["location/center_x_frac"] * W
    cy = row["location/center_y_frac"] * H
    dx, dy = row["alignments2D/shift"]

    cx = int(round(cx - dx))
    cy = int(round(cy - dy))
    box = int(row["blob/shape"][0])

    patch = _crop_with_padding(img, cx, cy, box)
    patch = patch.astype(np.float32)
    patch *= float(row.get("blob/sign", 1.0))

    return patch, (path, row["blob/idx"], cx, cy, box)


def _paste_patch(img, patch, cx, cy, box, idx=None):
    if img.ndim == 3:
        target = img[int(idx)]
    else:
        target = img

    half = box // 2
    H, W = target.shape

    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    ix0 = max(0, x0)
    ix1 = min(W, x1)
    iy0 = max(0, y0)
    iy1 = min(H, y1)

    px0 = ix0 - x0
    px1 = px0 + (ix1 - ix0)
    py0 = iy0 - y0
    py1 = py0 + (iy1 - iy0)

    target[iy0:iy1, ix0:ix1] = patch[py0:py1, px0:px1]


def cryosparc_whiten(stack, eps=1e-6):
    N = stack.shape[0]
    out = np.empty_like(stack)

    for i in range(N):
        img = stack[i]
        img = img - img.mean()
        std = img.std()
        if std < eps:
            out[i] = 0.0
        else:
            out[i] = img / std
    return out


def clip_outliers(stack, sigma=5.0):
    mu = stack.mean()
    std = stack.std()
    lo = mu - sigma * std
    hi = mu + sigma * std
    return np.clip(stack, lo, hi)
