import os
import mrcfile
import numpy as np
from cryosparc.dataset import Dataset
from ttsvd_denoise import process_file, process_stack
from utils import (_get_micro_params, _get_particle_diams,
                   _get_picker_params,_get_particle_patch,
                   _paste_patch)

CWD = os.getcwd()


def run_import_files(entry, kind, ws=None):
    emp_id, psA, acckV, cs, dose = _get_micro_params(entry)
    import_job = ws.create_job(
        f"import_{kind}",
        params={
            "blob_paths": f"{CWD}/{emp_id}/*.mrc*",
            "psize_A": psA,
            "accel_kv": acckV,
            "cs_mm": cs,
            "total_dose_e_per_A2": dose
        },
    )
    import_job.queue()
    import_job.wait_for_done()
    return import_job.uid


def run_motion_correction(job_uid, ws):
    mc_job = ws.create_job(
        "patch_motion_correction_multi",
        params={},
        connections={"movies": (job_uid, "imported_movies")
        }
    )
    mc_job.queue(lane="default")
    mc_job.wait_for_done()

    return mc_job.uid


def run_mrc_TTSVD(project, job_uid, ws, decomp="tt", ranks=24):
    last_job = project.find_job(job_uid)
    out_kw = "micrographs"
    if last_job.type.startswith("import"):
        out_kw = "imported_" + out_kw
    ttsvd_job = project.create_external_job(ws.uid,
                                            title="TT-SVD MRC Denoising")
    ttsvd_job.add_input(type="exposure", name="input_micrographs",
                        min=1, max=1, slots=["micrograph_blob", "mscope_params"])
    ttsvd_job.connect("input_micrographs", job_uid, out_kw,
                      slots=["micrograph_blob", "mscope_params"])
    output_name = "denoised_micrographs"
    ttsvd_job.add_output(
        type="exposure", name=output_name,
        slots=["micrograph_blob", "mscope_params"],
        passthrough="input_micrographs")

    ttsvd_job.start()
    ttsvd_job.mkdir("patch_denoised")
    mic_rows = ttsvd_job.load_input("input_micrographs", ["micrograph_blob", "mscope_params"])

    rows = {"micrograph_blob/path": []}
    for mic in mic_rows.rows():
        rel_in = mic["micrograph_blob/path"]
        basename = os.path.basename(rel_in)
        source = os.path.join(CWD, project.dir(), rel_in)

        rel_out = os.path.join("patch_denoised",
                               basename.replace(".mrc", "_ttsvd.mrc"))
        proc_outdir = os.path.join(CWD, project.dir(),
                                   ttsvd_job.uid, "patch_denoised")
        os.makedirs(proc_outdir, exist_ok=True)

        process_file(path=source, outdir=proc_outdir, patch=256,
                     stride=128, ranks=ranks, eps=None, device="cuda",
                     decomposition=decomp)
        dataset_path = os.path.join(ttsvd_job.uid, rel_out)
        for k, v in mic.items():
            if k not in rows:
                rows[k] = []
            if k == "micrograph_blob/path":
                rows[k].append(dataset_path)
            else:
                rows[k].append(v)

    dset = Dataset(rows)
    ttsvd_job.save_output("denoised_micrographs", dataset=dset)
    ttsvd_job.stop()
    return ttsvd_job.uid


def run_ctf_est(job_uid, project, ws):
    prev_job_type = project.find_job(job_uid).type
    if prev_job_type == "snowflake":
        output_name = "denoised_micrographs"
    else:
        output_name = "imported_micrographs"
    ctf_job = ws.create_job(
        "patch_ctf_estimation_multi",
        params={},
        connections={
            "exposures": (job_uid, output_name)
        }
    )

    ctf_job.queue(lane="default")
    ctf_job.wait_for_done()

    return ctf_job.uid


def run_ppicking(entry, job_uid, ws):
    min_pd, max_pd = _get_particle_diams(entry)
    blob_job = ws.create_job(
        "blob_picker_gpu",
        params={
            "diameter": min_pd,
            "diameter_max": max_pd,
            "num_plot": 5,
            "use_circle": True,
        },
        connections={
            "micrographs": (job_uid, "exposures")
        },
    )

    blob_job.queue(lane="default")
    blob_job.wait_for_done()
    return blob_job.uid


def run_pextract(entry, job_uid, ws):
    box, fcrop = _get_picker_params(entry)
    mrc_extr = ws.create_job(
        "extract_micrographs_multi",
        params={
            "box_size_pix": box,
            "bin_size_pix": fcrop,
            "compute_num_gpus": 1,
            "output_f16": False
        },
        connections={
            "micrographs": (job_uid, "micrographs"),
            "particles": (job_uid, "particles")
        }
    )
    mrc_extr.queue(lane="default")
    mrc_extr.wait_for_done()

    return mrc_extr.uid


def run_2DClass(job_uid, ws, filam_flag=False):
    cls_job = ws.create_job(
        "class_2D_new",
        params={
            "class2D_K": 100,
            "class2D_max_res": 6,
            "class2D_sigma_init_factor": 4,
            "class2D_min_res_align": 50,
            "class2D_do_ctf": True,
            "class2D_window": True,
            "class2D_num_full_iter": 2,
            "class2D_num_full_iter_batch": 60,
            "class2D_num_full_iter_batchsize_per_class": 400,
            "class2D_estimate_in_plane_pose": filam_flag,
            "compute_use_ssd": False,
            "compute_num_gpus": 1,
            "random_seed": 42
        },
        connections={
            "particles": (job_uid, "particles")
        }
    )
    cls_job.queue(lane="default")
    cls_job.wait_for_done()

    return cls_job.uid


def run_2Dmanual(job_uid, ws):
    pick_2D = ws.create_job(
        "select_2D",
        params={},
        connections={
            "particles": (job_uid, "particles"),
            "templates": (job_uid, "class_averages")
        }
    )
    pick_2D.queue()
    pick_2D.wait_for_status("waiting")
    return pick_2D.uid


def run_stack_TTSVD(project, job_uid, ws, decomp="tt", ranks=24):
    """
    Under construction
    """
    last_job = project.find_job(job_uid)
    last_job.wait_for_done()

    ttsvd_job = project.create_external_job(ws.uid,
                                            title="TT-SVD ClassStacks Denoising")

    ttsvd_job.add_input(type="particle", name="input_particles",
        min=1, max=1, slots=["blob", "ctf", "alignments2D",
                             "pick_stats", "location"]
    )
    ttsvd_job.connect("input_particles", job_uid,
        "particles_selected",  slots=["blob", "ctf", "alignments2D",
                             "pick_stats", "location"]
    )
    ttsvd_job.add_output(type="particle", name="denoised_particles",
        slots=["blob", "ctf", "alignments2D",
               "pick_stats", "location"],
        passthrough="input_particles"
    )
    ttsvd_job.start()
    dset = ttsvd_job.load_input("input_particles",
        ["blob", "ctf", "alignments2D",
         "pick_stats", "location"]
    )

    # read in all mrcs at once, no gooing aroung this
    files = list(set(dset["blob/path"]))
    mgraphs = {}
    for mrcf in files:
        fname = project.dir() / mrcf
        with mrcfile.open(fname, permissive=True) as m:
            mgraphs[mrcf] = m.data.astype(np.float32)

    # collect 2D classes stacks
    classes = sorted(set(dset["alignments2D/class"]))
    stacks = {cls: [] for cls in classes}
    stacks_data = {cls: [] for cls in classes}
    for row in dset.rows():
        patch, meta = _get_particle_patch(row, mgraphs)
        cls = int(row["alignments2D/class"])
        stacks[cls].append(patch)
        stacks_data[cls].append(meta)

    # TT-SVD-based denoising of class stacks
    # also paste resulting patches back to MRCs
    for cls in classes:
        stack = np.stack(stacks[cls], axis=0)
        denoised = process_stack(stack, decomposition=decomp,
                                 ranks=ranks, device="cuda")
        for i, (path, blob_idx, cx, cy, box) in enumerate(stacks_data[cls]):
            _paste_patch(mgraphs[path], denoised[i],
                         cx, cy, box, idx=blob_idx)

    # create new MRC files for downstream
    ttsvd_job.mkdir("particle_denoised", exist_ok=True)
    new_paths = {}
    for old_path, img in mgraphs.items():
        new_name = os.path.basename(old_path).replace(".mrc", "_TTSVDSTACK.mrc")
        rel_path = os.path.join(ttsvd_job.uid, "particle_denoised", new_name)
        abs_path = project.dir() / rel_path
        with mrcfile.new(abs_path, overwrite=True) as m:
            m.set_data(img.astype(np.float32))
        new_paths[old_path] = rel_path

    # reassign dataset
    rows = {k: [] for k in dset.keys()}
    for row in dset.rows():
        for k in dset.keys():
            if k == "blob/path":
                rows[k].append(new_paths[row[k]])
            else:
                rows[k].append(row[k])
    out_dset = Dataset(rows)
    ttsvd_job.save_output("denoised_particles", dataset=out_dset)
    ttsvd_job.stop()
    return ttsvd_job.uid


def run_abinitio(job_uid, project, ws):
    prev_job_type = project.find_job(job_uid).type
    if prev_job_type == "snowflake":
        output_name = "denoised_particles"
    else:
        output_name = "particles_selected"

    abinit = ws.create_job(
        "homo_abinit",
        params={
            "abinit_K": 1,
            "abinit_num_init_iters": 500,
            "abinit_num_final_iters": 1000,
            "abinit_max_res": 1,
            "abinit_radwn_step": 0.02,
            "compute_use_ssd": False,
            "random_seed": 42
        },
        connections={"particles": (job_uid, output_name)}
    )
    abinit.queue(lane="default")
    abinit.wait_for_done()
    return abinit.uid



def run_homoref(job_uid, ws):
    href_job = ws.create_job(
        "homo_refine_new",
        params={"compute_use_ssd": False},
        connections={"particles": (job_uid, "particles_all_classes"),
                     "volume": (job_uid, "volume_class_0")}
    )
    href_job.queue(lane="default")
    href_job.wait_for_done()
    return href_job.uid


def run_lrdist_estim(job_uid, ws):
    lrdist_job = ws.create_job(
        "local_resolution",
        params={},
        connections={"volume": (job_uid, "volume")}
    )
    lrdist_job.queue(lane="default")
    lrdist_job.wait_for_done()
    return lrdist_job.uid
