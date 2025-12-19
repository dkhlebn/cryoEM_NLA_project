import argparse

def parse_cli():
    p = argparse.ArgumentParser(description="Create cryoSPARC pipeline from EMPIAR TSV")
    p.add_argument("--project", required=True, help="CryoSPARC project ID (e.g., P4)")
    p.add_argument("--workspace", default=None, help="CryoSPARC workspace ID (e.g., W1)")
    p.add_argument("--tsv", required=True, help="Path to TSV file")
    p.add_argument("--dataset_dir", required=True, help="Directory name in CWD with MRC data in it")
    p.add_argument("--hostname", required=True,
                   help="Hostname where ports for CryoSPARC are opened")
    p.add_argument("--out_dir", default="./pipeline_out",
                   help="Output directory where json will be stored")
    mrc_group = p.add_mutually_exclusive_group()
    mrc_group.add_argument("--mrc_ttsvd_job", action="store_true",
                           help="Apply TT-SVD denoising to MRCs")
    mrc_group.add_argument("--mrc_tucker_job",action="store_true",
                           help="Apply Tucker denoising to MRCs")
    p.add_argument("--mrc_ranks", default=None, help="Ranks for MRC decomposition.")

    cls_group = p.add_mutually_exclusive_group()
    cls_group.add_argument("--cls_ttsvd_job", action="store_true",
                           help="Apply TT-SVD denoising to 2D class stacks")
    cls_group.add_argument("--cls_tucker_job",action="store_true",
                           help="Apply Tucker denoising to 2D class stacks")
    p.add_argument("--cls_ranks", default=None, help="Ranks for CLS decomposition.")

    args = p.parse_args()
    return args


def get_decomposition(args, stage):
    if stage == "mrc":
        rks = int(args.mrc_ranks) if args.mrc_ranks else None
        if args.mrc_ttsvd_job:
            return "tt", rks
        if args.mrc_tucker_job:
            return "tucker", rks

    if stage == "cls":
        rks = int(args.cls_ranks) if args.cls_ranks else None
        if args.cls_ttsvd_job:
            return "tt", rks
        if args.cls_tucker_job:
            return "tucker", rks
    return None, None


def get_out_fname(args):
    fname = f"{args.outdir}/{args.dataset_dir}"
    if args.mrc_ranks is not None:
        mode = "Tucker" if args.mrc_tucker_job else "TTSVD"
        fname += f"_mrc{mode}{args.mrc_ranks}"
    else:
        fname += "_mrcNone"

    if args.cls_ranks is not None:
        mode = "Tucker" if args.cls_tucker_job else "TTSVD"
        fname += f"_cls{mode}{args.cls_ranks}"
    else:
        fname += "_clsNone"

    return fname
