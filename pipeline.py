import os
import pandas as pd
from cryosparc.tools import CryoSPARC
from cli_parse import parse_cli, get_decomposition
from cs_wrappers import (run_import_files, run_motion_correction,
                         run_mrc_TTSVD, run_ctf_est, run_ppicking,
                         run_pextract, run_2DClass, run_2Dmanual,
                         run_stack_TTSVD, run_abinitio,
                         run_homoref, run_lrdist_estim)
from utils import delete_job

MOVIES_JOB_TYPE = "import_movies"
MICRO_JOB_TYPE = "import_micrographs"
PATCH_MC_TYPE = "patch_motion_correction_multi"
CWD = os.getcwd()


def main():
    args = parse_cli()
    license_id = os.environ["LICENSE_ID"]
    pwd = os.environ["CRYOSPARC_PWD"]
    cs_email = os.environ["CRYOSPARC_EMAIL"]
    server_port = int(os.environ["CRYOSPARC_PORT"])

    cs = CryoSPARC(license=license_id,
               host=args.hostname, base_port=server_port,
               email=cs_email,
               password=pwd)

    project = cs.find_project(args.project)
    if args.workspace is None:
        workspace = project.create_workspace("W1", desc=f"{args.dataset_dir} Workspace")
    else:
        workspace = project.find_workspace(args.workspace)
    df = pd.read_table(args.tsv).query(f"EMPIAR_ID == '{args.dataset_dir}'").iloc[0, :]
    mrc_decomp, mrc_ranks = get_decomposition(args, "mrc")
    cls_decomp, cls_ranks = get_decomposition(args, "cls")

    # main loop
    jobs = []
    if df["EMPIAR_ID"] == "EMPIAR-10017":
        mtcorr_uid = run_import_files(df, "micrographs", ws=workspace)
        jobs.append(mtcorr_uid)
    else:
        import_uid = run_import_files(df, "movies", ws=workspace)
        mtcorr_uid = run_motion_correction(import_uid, ws=workspace)
        jobs.extend([import_uid, mtcorr_uid])
    if mrc_decomp is not None:
        mtcorr_uid = run_mrc_TTSVD(project, mtcorr_uid,
                               workspace, decomp=mrc_decomp,
                               ranks=mrc_ranks)
        jobs.append(mtcorr_uid)
    ctf_est_uid = run_ctf_est(mtcorr_uid, project, workspace)
    bpick_uid = run_ppicking(df, ctf_est_uid, workspace)
    pextr_uid = run_pextract(df, bpick_uid, workspace)
    class2d_uid = run_2DClass(df, pextr_uid, workspace)
    select_2D_uid = run_2Dmanual(class2d_uid, workspace)
    jobs.extend([ctf_est_uid, bpick_uid, pextr_uid,
                 class2d_uid, select_2D_uid])
    if cls_decomp is not None:
        select_2D_uid = run_stack_TTSVD(project, select_2D_uid, workspace,
                                        decomp=cls_decomp, ranks=cls_ranks)
        jobs.append(select_2D_uid)
    abin_uid = run_abinitio(select_2D_uid, project, workspace)
    href_uid = run_homoref(abin_uid, workspace)
    locres_uid = run_lrdist_estim(href_uid, workspace)
    jobs.extend([abin_uid, href_uid, locres_uid])

     #delete jobs
    for j in reversed(jobs):
        delete_job(j, project.uid)

if __name__ == "__main__":
    main()
