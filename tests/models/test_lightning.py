import torch

import scvi
from scvi.data import synthetic_iid, heart_cell_atlas_subsampled
from scvi.model import SCVI
from scvi.train._callbacks import SaveBestState


def test_save_best_state_callback(save_path):
    n_latent = 10
    # adata = synthetic_iid()
    adata = heart_cell_atlas_subsampled()
    # SCVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    SCVI.setup_anndata(adata)
    model = SCVI(adata, n_latent=n_latent, n_layers=2, n_hidden=256)
    callbacks = [SaveBestState(verbose=True)]
    import time
    start = time.time()
    scvi.settings.dl_pin_memory_gpu_training = True
    model.train(50, batch_size=512, check_val_every_n_epoch=1, train_size=0.5, devices=2, callbacks=callbacks)
    end = time.time()
    print(end - start)
    # 2.6 s/it, 60-90% usage, 258.207s total
    # 5.10 s/it, 40-80% usage each, 517.616s total


def test_set_seed(save_path):
    scvi.settings.seed = 1
    n_latent = 5
    adata = synthetic_iid()
    SCVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model1 = SCVI(adata, n_latent=n_latent)
    model1.train(1)
    scvi.settings.seed = 1
    model2 = SCVI(adata, n_latent=n_latent)
    model2.train(1)
    assert torch.equal(
        model1.module.z_encoder.encoder.fc_layers[0][0].weight,
        model2.module.z_encoder.encoder.fc_layers[0][0].weight,
    )
