# This file contains the main script for NeMF evaluation on AirMSPI data.
# It is based on VIP-CT source code ('https://github.com/ronenroi/VIPCT') by Roi Ronen
# Copyright (c) Roi Ronen et al.
# All rights reserved.

# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Inbal Kom Betzer, Roi Ronen, Vadim Holodovsky, Yoav. Y. Schechner and Ilan Koren,
# "NeMF: Neural Microphysics Fields",
# TBD TPAMI 2024.
#
# Copyright (c) Inbal Kom Betzer. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.

import os, time
import warnings
import hydra
import numpy as np
from NeMF.microphysics_airmspi_dataset import get_real_world_microphysics_airmspi_datasets, trivial_collate
from NeMF.NeMFnet import *
from omegaconf import OmegaConf
from omegaconf import DictConfig
from NeMF.cameras import AirMSPICameras
import scipy.io as sio

CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
MASK_EST_TH = 0.5

@hydra.main(config_path=CONFIG_DIR, config_name="microphysics_test_airmspi")
def main(cfg: DictConfig):

    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device on which to run.
    if torch.cuda.is_available() and cfg.debug == False:
        n_device = torch.cuda.device_count()
        cfg.gpu = 0 if n_device==1 else cfg.gpu
        device = f"cuda:{cfg.gpu}"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the training is unlikely to finish in reasonable time."
        )
        device = "cpu"

    log_dir = os.getcwd()
    log_dir = log_dir.replace('outputs','test_results')
    results_dir = log_dir
    checkpoint_resume_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_resume_path)
    if len(results_dir) > 0:
        # Make the root of the experiment directory
        os.makedirs(results_dir, exist_ok=True)

    resume_cfg_path = os.path.join(checkpoint_resume_path.split('/checkpoints')[0],'.hydra/config.yaml')
    net_cfg = OmegaConf.load(resume_cfg_path)
    cfg = OmegaConf.merge(net_cfg,cfg)

    # Initialize VIP-CT model
    model = NeMFnetAirMSPI(cfg=cfg, n_cam=cfg.data.n_cam)

    # Load model
    assert os.path.isfile(checkpoint_resume_path)
    print(f"Resuming from checkpoint {checkpoint_resume_path}.")
    loaded_data = torch.load(checkpoint_resume_path, map_location=device)
    model.load_state_dict(loaded_data["model"])
    model.to(device)
    model.eval().float()


    batch_time_net = []

    val_dataset = get_real_world_microphysics_airmspi_datasets(cfg=cfg)
    val_image, grid, images_mapping_list, pixel_centers_list, masks = val_dataset[0]
    val_image = torch.tensor(val_image,device=device).float()[None]

    masks = torch.tensor(masks,device=device)[None]
    val_volume = Volumes(torch.unsqueeze(torch.tensor(masks, device=device).float(), 1), grid)
    val_camera = AirMSPICameras(mapping=torch.tensor(images_mapping_list, device=device).float(),
                                  centers=torch.tensor(pixel_centers_list).float(), device=device)


# Activate eval mode of the model (lets us do a full rendering pass).
    with torch.no_grad():
        est_lwc_full = torch.zeros((masks.shape[1], masks.shape[2], masks.shape[3]), device=masks.device)
        est_reff_full = torch.zeros((masks.shape[1], masks.shape[2], masks.shape[3]), device=masks.device)
        est_veff_full = torch.zeros((masks.shape[1], masks.shape[2], masks.shape[3]), device=masks.device)
        # n_points_mask = torch.sum(torch.stack(masks)*1.0) if isinstance(masks, list) else masks.sum()
        # if n_points_mask > cfg.min_mask_points:
        net_start_time = time.time()

        val_out = model(
            val_camera,
            val_image,
            val_volume,
            masks
        )

        assert len(val_out["output"]) == 1  ##TODO support validation with batch larger than 1

        est_lwc = val_out["output"][0][:,0]
        est_mask = val_out["output"][0][:,1]
        est_reff = val_out["output"][0][:, 2]
        est_veff = val_out["output"][0][:,3]

        est_lwc[est_mask<MASK_EST_TH] = 0
        est_reff[est_mask<MASK_EST_TH] = 0
        est_veff[est_mask<MASK_EST_TH] = 0


        est_lwc_full[masks.squeeze()] = est_lwc
        est_lwc_full[est_lwc_full < 0.001] = 0
        est_lwc_full[est_lwc_full > 2.5] = 2.5
        est_reff_full[masks.squeeze()] = est_reff
        est_reff_full[est_reff_full < 1] = 0
        est_reff_full[est_reff_full > 35] = 35
        est_veff_full[masks.squeeze()] = est_veff
        est_veff_full[est_veff_full < 0.01] = 0.01  # for SHDOM
        est_veff_full[est_veff_full > 0.55] = 0.55

        time_net = time.time() - net_start_time

        airmspi_cloud = {'cloud_lwc':est_lwc_full.cpu().numpy(),'cloud_reff':est_reff_full.cpu().numpy(),'cloud_veff':est_veff_full.cpu().numpy()}

        sio.savemat('airmspi_recovery_bomex.mat', airmspi_cloud)
        batch_time_net.append(time_net)

    batch_time_net = np.array(batch_time_net)

    print(f'Mean time = {np.mean(batch_time_net)}')


if __name__ == "__main__":
    main()


