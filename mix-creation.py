from glob import glob
import argparse

from hw_ss.utils.mixer import *

parser = argparse.ArgumentParser()
parser.add_argument("--path_mixtures_train")
parser.add_argument("--path_mixtures_val")
parser.add_argument("--speakers_files_train")
parser.add_argument("--speakers_files_val")
parser.add_argument("--nfiles-train", default=5000)
parser.add_argument("--nfiles-val", default=5000)
args = parser.parse_args()

# TODO: initialize
mixer_train = MixtureGenerator(speakers_files=args.speakers_files_train,
                               out_folder=args.path_mixtures_train,
                               nfiles=args.nfiles_train)
mixer_val = MixtureGenerator(speakers_files=args.speakers_files_val,
                             out_folder=args.path_mixtures_val,
                             nfiles=args.nfiles_val,
                             test=True)

mixer_train.generate_mixes(snr_levels=[-5, 5],
                           num_workers=2,
                           update_steps=100,
                           trim_db=20,
                           vad_db=20,
                           audioLen=3)

mixer_val.generate_mixes(snr_levels=[-5, 5],
                           num_workers=2,
                           update_steps=100,
                           trim_db=None,
                           vad_db=20,
                           audioLen=3)
