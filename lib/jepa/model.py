import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# Add project root so `cgm_jepa` is importable when running this module directly
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
from cgm_jepa.models.encoder import Encoder
from cgm_jepa.models.decoder import ForecastingDecoder
from cgm_jepa.config.model_configs import get_pretrained_metadata
import wandb
import copy
import optuna

from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual
from torch.utils.tensorboard import SummaryWriter
from lib.gluformer.utils.training import ExpLikeliLoss, \
                                         EarlyStop, \
                                         modify_collate, \
                                         adjust_learning_rate


class CGM_JEPA(torch.nn.Module):
    def __init__(self, encoder, decoder, patch_size=12):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        emb, _ = self.encoder(x)
        emb = torch.mean(emb, dim=1)
        return self.decoder(emb)

    def fit(self, 
            train_dataset: SamplingDatasetDual,
            val_dataset: SamplingDatasetDual,
            learning_rate: float = 1e-3,
            batch_size: int = 32,
            epochs: int = 100,
            num_samples: int = 100,
            device: str = 'cuda',
            model_path: str = None,
            trial: optuna.trial.Trial = None,
            logger: SummaryWriter = None,):
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate, betas=(0.1, 0.9))
        scaler = torch.cuda.amp.GradScaler()
        early_stop = EarlyStop(patience=10, delta=0.001)
        self.to(device)
        
        for epoch in range(epochs):
            train_loss = []
            self.train()
            for i, (past_target_series, 
                    past_covariates, 
                    future_covariates, 
                    static_covariates, 
                    future_target_series) in enumerate(train_loader):

                optimizer.zero_grad()
                past_target_series = past_target_series.to(device)
                future_target_series = future_target_series.to(device)

                num_patches = past_target_series.size(1) // self.patch_size
                patched = [past_target_series[:, i * self.patch_size : (i + 1) * self.patch_size] for i in range(num_patches)]
                patched = torch.stack(patched, dim=1).squeeze() # Stack along dimension 1 to create [batch_size, num_patches, patch_size]
                patched = patched

                # TODO: make sure timestamp is included for the encoder
                pred = self(patched) # only use the cgm data, wo covs

                loss = criterion(pred, future_target_series.squeeze())
                loss.backward()
                optimizer.step()

                if logger is not None:
                    logger.add_scalar('train_loss', loss.item(), epoch * len(train_loader) * i)
                train_loss.append(loss.item())
            
            # log loss
            if logger is not None:
                logger.add_scalar('train_loss_epoch', np.mean(train_loss), epoch)
        
            # eval
            val_loss = []
            self.eval()
            with torch.no_grad():
                for i, (past_target_series, 
                        past_covariates, 
                        future_covariates, 
                        static_covariates, 
                        future_target_series) in enumerate(val_loader):
                    
                    past_target_series = past_target_series.to(device)
                    future_target_series = future_target_series.to(device)

                    num_patches = past_target_series.size(1) // self.patch_size
                    patched = [past_target_series[:, i * self.patch_size : (i + 1) * self.patch_size] for i in range(num_patches)]
                    patched = torch.stack(patched, dim=1).squeeze() # Stack along dimension 1 to create [batch_size, num_patches, patch_size]

                    # TODO: make sure timestamp is included for the encoder
                    pred = self(patched) # only use the cgm data, wo covs

                    loss = criterion(pred, future_target_series.squeeze())
                    val_loss.append(loss.item())
                            # log loss
                    if logger is not None:
                        logger.add_scalar('val_loss', loss.item(), epoch * len(val_loader) + i)
                # log loss
                logger.add_scalar('val_loss_epoch', np.mean(val_loss), epoch)
                # check early stopping
                early_stop(np.mean(val_loss), self, model_path)
                if early_stop.stop:
                    break


    def predict(self, test_dataset: SamplingDatasetInferenceDual, 
                batch_size: int = 32,
                num_samples: int = 100,
                device: str = 'cuda',
                use_tqdm: bool = False):
        """
        Predict the future target series given the supplied samples from the dataset.

        Parameters
        ----------
        test_dataset : SamplingDatasetInferenceDual
            The dataset to use for inference.
        batch_size : int, optional
            The batch size to use for inference, by default 32
        num_samples : int, optional
            The number of samples to replicate predictions (for compatibility with variance models), by default 100
        device : str, optional
            Device to use, by default 'cuda'
        use_tqdm : bool, optional
            Whether to use tqdm progress bar, by default False
        
        Returns
        -------
        np.ndarray
            The predicted future target series in shape n x len_pred x num_samples, where
            n is total number of predictions.
        """
        # define data loader
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)
        # predict
        self.eval()
        # move to device
        self.to(device)
        predictions = []
        for i, (past_target_series,
                historic_future_covariates,
                future_covariates,
                static_covariates) in enumerate(tqdm(test_loader)) if use_tqdm else enumerate(test_loader):
            # move to device
            past_target_series = past_target_series.to(device)
            
            # patch the input data (same as in fit/eval)
            num_patches = past_target_series.size(1) // self.patch_size
            patched = [past_target_series[:, i * self.patch_size : (i + 1) * self.patch_size] for i in range(num_patches)]
            patched = torch.stack(patched, dim=1).squeeze()  # Stack along dimension 1 to create [batch_size, num_patches, patch_size]
            
            # TODO: make sure timestamp is included for the encoder
            # forward pass
            with torch.no_grad():
                pred = self(patched)  # only use the cgm data, wo covs
            
            # transfer to numpy
            pred = pred.cpu().detach().numpy()
            
            # Replicate predictions num_samples times to match expected shape (n, t, k)
            # where k is num_samples (for compatibility with variance models)
            pred = np.expand_dims(pred, axis=-1)  # Add sample dimension: (batch, len_pred, 1)
            pred = np.repeat(pred, num_samples, axis=-1)  # Replicate: (batch, len_pred, num_samples)
            
            predictions.append(pred)
        
        predictions = np.concatenate(predictions, axis=0)
        return predictions
       

def load_model(
    device,
    decoder_num_layers=2
):
    # Encoder (JEPA)
    api = wandb.Api()
    cgm_jepa_version = "v6"
    cgm_jepa_artifact = api.artifact(f'hadamuhammad-unsw/cgm-jepa/cgm-jepa:{cgm_jepa_version}', type="model")
    cgm_jepa_metadata = cgm_jepa_artifact.metadata

    untrained_encoder = Encoder(
        num_patches=None, # No need to specify
        dim_in=cgm_jepa_metadata["patch_size"], # patch size
        kernel_size=cgm_jepa_metadata["encoder_kernel_size"],
        embed_dim=cgm_jepa_metadata["encoder_embed_dim"],
        embed_bias=cgm_jepa_metadata["encoder_embed_bias"],
        nhead=cgm_jepa_metadata["encoder_nhead"],
        num_layers=cgm_jepa_metadata["encoder_num_layers"],
        jepa=False # we don't apply jepa training in downstream
    )

    cgm_jepa_dir = cgm_jepa_artifact.download()
    pretrained_encoder = copy.deepcopy(untrained_encoder).to(device)
    pretrained_encoder.load_state_dict(torch.load(f"{cgm_jepa_dir}/cgm-jepa", map_location=device)["encoder"], strict=False)

    decoder = ForecastingDecoder(
        emb_dim=cgm_jepa_metadata["encoder_embed_dim"],
        patch_size=cgm_jepa_metadata["patch_size"],
        num_layers=decoder_num_layers
    )

    return CGM_JEPA(
        patch_size=cgm_jepa_metadata["patch_size"],
        encoder=pretrained_encoder,
        decoder=decoder
    ).to(device)