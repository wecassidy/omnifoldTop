"""
Define the command line interface for unfolding.

Call ``cli.parser.parse_args()`` to get the command line arguments passed
to the script.

Attributes
----------
parser : argparse.ArgumentParser
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--observables-train",
    dest="observables_train",
    nargs="+",
    default=["th_pt", "th_y", "th_phi", "th_e", "tl_pt", "tl_y", "tl_phi", "tl_e"],
    help="List of observables to use in training.",
)
parser.add_argument(
    "--observables",
    nargs="+",
    default=[
        "mtt",
        "ptt",
        "ytt",
        "ystar",
        "chitt",
        "yboost",
        "dphi",
        "Ht",
        "th_pt",
        "th_y",
        "th_eta",
        "th_phi",
        "th_m",
        "th_e",
        "th_pout",
        "tl_pt",
        "tl_y",
        "tl_eta",
        "tl_phi",
        "tl_m",
        "tl_e",
        "tl_pout",
    ],
    help="List of observables to unfold",
)
parser.add_argument(
    "--observable-config",
    dest="observable_config",
    default="configs/observables/vars_ttbardiffXs.json",
    help="JSON configurations for observables",
)
parser.add_argument(
    "-d",
    "--data",
    required=True,
    nargs="+",
    type=str,
    help="Observed data npz file names",
)
parser.add_argument(
    "-s",
    "--signal",
    required=True,
    nargs="+",
    type=str,
    help="Signal MC npz file names",
)
parser.add_argument(
    "-b", "--background", nargs="+", type=str, help="Background MC npz file names"
)
parser.add_argument(
    "--bdata",
    nargs="+",
    type=str,
    default=None,
    help="Background MC files to be mixed with data",
)
parser.add_argument(
    "-o", "--outputdir", default="./output", help="Directory for storing outputs"
)
parser.add_argument(
    "-t",
    "--truth-known",
    dest="truth_known",
    action="store_true",
    help="MC truth is known for 'data' sample",
)
parser.add_argument(
    "-c",
    "--plot-correlations",
    dest="plot_correlations",
    action="store_true",
    help="Plot pairwise correlations of training variables",
)
parser.add_argument(
    "-i",
    "--iterations",
    type=int,
    default=4,
    help="Numbers of iterations for unfolding",
)
parser.add_argument(
    "--weight", default="totalWeight_nominal", help="name of event weight"
)
parser.add_argument(
    "-m",
    "--background-mode",
    dest="background_mode",
    choices=["default", "negW", "multiClass"],
    default="default",
    help="Background mode",
)
parser.add_argument(
    "-r",
    "--reweight-data",
    dest="reweight_data",
    choices=["linear_th_pt", "gaussian_bump", "gaussian_tail"],
    default=None,
    help="Reweight strategy of the input spectrum for stress tests",
)
parser.add_argument(
    "-v", "--verbose", action="count", default=0, help="Verbosity level"
)
parser.add_argument(
    "-g",
    "--gpu",
    type=int,
    choices=[0, 1],
    default=None,
    help="Manually select one of the GPUs to run",
)
parser.add_argument(
    "--unfolded-weights",
    dest="unfolded_weights",
    nargs="*",
    type=str,
    help="Unfolded weights file names. If provided, load event weights directly from the files and skip training.",
)
parser.add_argument(
    "--binning-config",
    dest="binning_config",
    default="configs/binning/bins_10equal.json",
    type=str,
    help="Binning config file for variables",
)
parser.add_argument(
    "--plot-history",
    dest="plot_history",
    action="store_true",
    help="If true, plot intermediate steps of unfolding",
)
parser.add_argument(
    "--nresamples",
    type=int,
    default=25,
    help="number of times for resampling to estimate the unfolding uncertainty using the bootstrap method",
)
parser.add_argument(
    "-e",
    "--error-type",
    dest="error_type",
    choices=["sumw2", "bootstrap_full", "bootstrap_model"],
    default="sumw2",
    help="Method to evaluate uncertainties",
)
parser.add_argument(
    "--batch-size",
    dest="batch_size",
    type=int,
    default=512,
    help="Batch size for training",
)
parser.add_argument(
    "--model-name",
    dest="model_name",
    default="dense_3hl",
    help="Model architecture name, referring to one of the models in python/models.py or --model-file",
)
parser.add_argument(
    "--model-file",
    dest="model_file",
    help="Path to a Python module containing module architecture functions as in python/models.py",
)

# parser.add_argument('-n', '--normalize',
#                    action='store_true',
#                    help="Normalize the distributions when plotting the result")
# parser.add_argument('--alt-rw', dest='alt_rw',
#                    action='store_true',
#                    help="Use alternative reweighting if true")
