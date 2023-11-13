import glob
from glob import glob
import argparse

from hw_ss.utils.mixer import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_mixtures_train", default="data/datasets/mixes/train")
    parser.add_argument("--path_mixtures_val", default="data/datasets/mixes/val")
    parser.add_argument("--speakers_files_train", default="data/datasets/librispeech/train-clean-100")
    parser.add_argument("--speakers_files_val", default="data/datasets/librispeech/test-clean")
    parser.add_argument("--nfiles-train", default=3000)
    parser.add_argument("--nfiles-val", default=500)
    args = parser.parse_args()

    speakers_train = [elem.name for elem in os.scandir(args.speakers_files_train)]
    speakers_val = [elem.name for elem in os.scandir(args.speakers_files_val)]
    speakers_files_train = [LibriSpeechSpeakerFiles(i, args.speakers_files_train, "*.flac") for i in speakers_train]
    speakers_files_val = [LibriSpeechSpeakerFiles(i, args.speakers_files_val, "*.flac") for i in speakers_val]

    mixer_train = MixtureGenerator(speakers_files=speakers_files_train,
                                out_folder=args.path_mixtures_train,
                                nfiles=args.nfiles_train)
    mixer_val = MixtureGenerator(speakers_files=speakers_files_val,
                                out_folder=args.path_mixtures_val,
                                nfiles=args.nfiles_val,
                                test=True)

    print("About to generate training mixes:")
    mixer_train.generate_mixes(snr_levels=[-5, 5],
                            num_workers=2,
                            update_steps=100,
                            trim_db=20,
                            vad_db=0, # inspired by @dogfew
                            audioLen=3)
    
    print("About to generate validation mixes")
    mixer_val.generate_mixes(snr_levels=[0, 0],
                            num_workers=2,
                            update_steps=100,
                            trim_db=None,
                            vad_db=0,
                            audioLen=3)
