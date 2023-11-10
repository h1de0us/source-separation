## Installation guide

To install all dependencies, run:
```shell
pip install -r ./requirements.txt
```

To create a mix, run `./create_mixture.sh` to download the dataset automatically and create mixes with the default parameters, or download dataset manually and run the following command:
```
python3 mix-creation.py \
    --speakers_files_train <your-path-to-train-data> \
    --speakers_files_val <your-path-to-val-data> \
    --path_mixtures_train <output-folder-for-training-mixes> \
    --path_mixtures_val <output-folder-for-validation-mixes> \
    --nfiles-train <number-of-files-to-use-for-training-mixes> \
    --nfiles-val <number-of-files-to-use-for-validation-mixes>
```

To download the model checkpoint and configs, run `./run.sh`


## Before submitting

0) Make sure your projects run on a new machine after complemeting the installation guide or by 
   running it in docker container.
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checkpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize

