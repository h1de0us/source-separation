from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import hw_ss.augmentations
import hw_ss.datasets
from hw_ss.collate_fn.collate import collate_fn
from hw_ss.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            wave_augs, spec_augs = hw_ss.augmentations.from_configs(configs)
            drop_last = True
        else:
            wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(
                obj_dict=ds, 
                default_module=hw_ss.datasets, 
                config_parser=configs,
                wave_augs=wave_augs, 
                spec_augs=spec_augs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collate_fn,
            shuffle=shuffle, num_workers=num_workers,
            drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders