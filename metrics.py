import json
import bson
import mrcfile
import numpy as np
from utils import _get_particle_patch

def local_resolution_metrics(pj, lres_uid):
    bson_path = pj.dir() / lres_uid / "events.bson"
    with open(bson_path, 'rb') as f:
        data = bson.loads(f.read())["events"]
    for i in range(len(data)):
        if "Local resolution stats" in data[i].get("text"):
            idx = i + 1

    raw_vals = data[idx].get("text").strip("()\n")
    mnm, q25, med, q75, mxm = [float(x) for x in raw_vals.split(", ")]
    return {"locres_min": mnm,
            "locres_q25": q25,
            "locres_median": med,
            "locres_q75": q75,
            "locres_max": mxm}


def ptcl_cls_retention(pj, select_uid):
    bson_path = pj.dir() / select_uid / "events.bson"
    with open(bson_path, 'rb') as f:
        data = bson.loads(f.read())["events"]

    dt = {"pcl_sel": 0,
          "pcl_exc": 0,
          "cls_sel": 0,
          "cls_exc": 0}
    for i in range(len(data)):
        txt = data[i].get("text")
        pcl = "Particles" in txt
        cls = "Templates" in txt
        sel = "selected" in txt
        exc = "excluded" in txt
        if (pcl or cls) and (sel or exc):
            n = int(data[i].get("text").split(' ')[-1])
            if pcl and sel:
                dt["pcl_sel"] += n
            elif pcl and exc:
                dt["pcl_exc"] += n
            elif cls and sel:
                dt["cls_sel"] += n
            elif cls and exc:
                dt["cls_exc"] += n
    return {"ptcls": dt["pcl_sel"] + dt["pcl_exc"],
            "ptcl_retention": dt["pcl_sel"] / (dt["pcl_sel"] + dt["pcl_exc"]),
            "cls_retention": dt["cls_sel"] / (dt["cls_sel"] + dt["cls_exc"])}


def class2d_metrics(pj, select_uid):
    last_job = pj.find_job(select_uid)
    dset = last_job.load_output("particles_selected",
        ["blob", "alignments2D",
         "pick_stats", "location"]
    )
    files = list(set(dset["blob/path"]))
    mgraphs = {}
    for mrcf in files:
        fname = pj.dir() / mrcf
        with mrcfile.open(fname, permissive=True) as m:
            mgraphs[mrcf] = m.data.astype(np.float32)

    classes = sorted(set(dset["alignments2D/class"]))
    stacks = {cls: [] for cls in classes}
    for row in dset.rows():
        patch, _ = _get_particle_patch(row, mgraphs)
        cls = int(row["alignments2D/class"])
        stacks[cls].append(patch)

    class_sizes = []
    class_variances = {"means": [],
                       "medians": []}
    for cls in classes:
        stack = np.stack(stacks[cls], axis=0)
        class_sizes.append(stack.shape[0])
        voxel_var = stack.var(axis=0)
        voxel_var_flat = voxel_var.flatten()
        class_variances["means"].append(voxel_var_flat.mean())
        class_variances["medians"].append(np.median(voxel_var_flat))

    metrics = {
        "class2D_occ_min": int(np.min(class_sizes)),
        "class2D_occ_q25": int(np.percentile(class_sizes, 25)),
        "class2D_occ_median": int(np.median(class_sizes)),
        "class2D_occ_q75": int(np.percentile(class_sizes, 75)),
        "class2D_occ_max": int(np.max(class_sizes)),
        "class2D_var_mean": float(np.mean(class_variances["means"])),
        "class2D_var_median": float(np.median(class_variances["medians"]))
        }
    return metrics


def fsc_metrics(pj, href_uid):
    job = pj.find_job(href_uid).__dict__["_doc"]
    out_stats = job["output_result_groups"][1]["latest_summary_stats"]["fsc_info_best"]

    return {"fcs_0.143": float(out_stats["radwn_noisesub_A"])}


def bfactor_metric(pj, sharpen_uid):
    job = pj.find_job(sharpen_uid).__dict__["_doc"]
    out_stats = job["output_result_groups"][0]["latest_summary_stats"]

    return {"b_factor": float(out_stats["b_factor"])}


def angular_distribution_entropy(pj, href_uid, n_bins=32):
    job = pj.find_job(href_uid)
    particles = job.load_output("particles")["alignments3D/pose"]
    rotvecs = np.stack(list(particles))

    angles = np.linalg.norm(rotvecs, axis=1)
    axes = rotvecs / angles[:, None]
    axes[angles == 0] = 0

    x, y, z = axes[:,0], axes[:,1], axes[:,2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    hist, _, _ = np.histogram2d(theta, phi, bins=[n_bins, 2*n_bins],
                                range=[[0, np.pi], [-np.pi, np.pi]])
    p = hist.flatten()
    p = p[p > 0]
    p /= p.sum()
    entropy = -np.sum(p * np.log(p))
    entropy_norm = entropy / np.log(len(p))

    return {"angular_entropy": float(entropy),
            "angular_entropy_norm": float(entropy_norm)}


def compute_all_metrics(pj, job_uids, fname):
    metrics = {}
    metrics.update(fsc_metrics(pj, job_uids["homo_refine"]))
    metrics.update(local_resolution_metrics(pj, job_uids["local_resolution"]))
    metrics.update(ptcl_cls_retention(pj, job_uids["select2d"]))
    metrics.update(class2d_metrics(pj, job_uids["select2d"]))
    metrics.update(bfactor_metric(pj, job_uids["sharpen"]))
    metrics.update(angular_distribution_entropy(pj, job_uids["homo_refine"]))

    with open(f"{fname}.json", 'w') as f:
        json.dump(metrics, f)
