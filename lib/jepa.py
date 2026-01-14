import sys
import os
from torch.cpu import is_available
import yaml
import datetime
import argparse
from functools import partial

import torch
import optuna
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *

from lib.jepa.model import load_model
from utils.darts_training import print_callback
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual

# Simple test function without variance
def test_no_var(series: np.ndarray, forecasts: np.ndarray):
    """
    Test the forecasts on the series without variance.
    
    Parameters
    ----------
    series
        The target time series of shape (n, t), 
        where t is length of prediction.
    forecasts
        The forecasted means of shape (n, t, k),
        where k is the number of samples.
    
    Returns
    -------
    np.ndarray
        Error array. Array of shape (n, 2) with MSE and MAE.
    """
    # Use mean of forecasts across samples
    if forecasts.ndim == 3:
        forecasts_mean = forecasts.mean(axis=-1)  # (n, t)
    else:
        forecasts_mean = forecasts.squeeze()
    
    series = series.squeeze()
    
    mse = np.mean((series - forecasts_mean)**2, axis=-1)
    mae = np.mean(np.abs(series - forecasts_mean), axis=-1)
    errors = np.stack([mse, mae], axis=-1)
    
    return errors

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weinstock')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--optuna', type=str, default='True')
parser.add_argument('--reduction1', type=str, default='mean')
parser.add_argument('--reduction2', type=str, default='median')
parser.add_argument('--reduction3', type=str, default=None)
args = parser.parse_args()
reductions = [args.reduction1, args.reduction2, args.reduction3]
if __name__ == '__main__':
    # define device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print("using gpu")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("using mps")
    else:
        device = torch.device('cpu')
        print("using cpu")
    # load data
    study_file = f'./output/jepa_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    formatter, series, scalers = load_data(seed=0, 
                                           study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=False, 
                                           cov_type='dual',
                                           use_static_covs=False)
    
    # hyperparameter optimization
    best_params = None
    if args.optuna == 'True':
        pass
    else:
        assert formatter.params["jepa"] is not None, "No saved hyperparameters found for this model"
        best_params = formatter.params["jepa"]

    # set parameters
    out_len = formatter.params['length_pred']
    model_path = os.path.join(os.path.dirname(__file__),
                              f'../output/tensorboard_jepa_{args.dataset}/model.pt')
    # suggest hyperparameters: input size
    in_len = best_params["in_len"]
    label_len = in_len // 3
    max_samples_per_ts = best_params["max_samples_per_ts"]
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    embed_dim = best_params["d_model"]
    nhead = best_params["n_heads"]
    num_layers = best_params["num_enc_layers"]
    patch_size = best_params["patch_size"]
    kernel_size = best_params["num_enc_kernel"]
    decoder_layers = best_params["num_dec_layers"]

    # Set model seed
    model_seeds = list(range(10, 20))
    id_errors_model = {key: [] for key in reductions if key is not None}
    ood_errors_model = {key: [] for key in reductions if key is not None}
    for model_seed in model_seeds:
        # Backtest on the test set
        seeds = list(range(1, 3))
        id_errors_cv = {key: [] for key in reductions if key is not None}
        ood_errors_cv = {key: [] for key in reductions if key is not None}
        for seed in seeds:
            writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 
                           f'../output/tensorboard_jepa_{args.dataset}/test_run{model_seed}_{seed}'))
            formatter, series, scalers = reshuffle_data(formatter=formatter, 
                                                        seed=seed, 
                                                        use_covs=True,
                                                        cov_type='past',
                                                        use_static_covs=True)
            # create datasets
            dataset_train = SamplingDatasetDual(series['train']['target'],
                                                series['train']['future'],
                                                output_chunk_length=out_len,
                                                input_chunk_length=in_len,
                                                use_static_covariates=True,
                                                max_samples_per_ts=max_samples_per_ts,)
            dataset_val = SamplingDatasetDual(series['val']['target'],
                                              series['val']['future'],   
                                              output_chunk_length=out_len,
                                              input_chunk_length=in_len,
                                              use_static_covariates=True,)
            dataset_test = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                        covariates=series['test']['future'],
                                                        input_chunk_length=in_len,
                                                        output_chunk_length=out_len,
                                                        use_static_covariates=True,
                                                        array_output_only=True)
            dataset_test_ood = SamplingDatasetInferenceDual(target_series=series['test_ood']['target'],
                                                            covariates=series['test_ood']['future'],
                                                            input_chunk_length=in_len,
                                                            output_chunk_length=out_len,
                                                            use_static_covariates=True,
                                                            array_output_only=True)
            # build the JEPA model
            model = load_model(
                device=device,
                decoder_num_layers=decoder_layers
            )
            # Dataset nomr is min max, lets see if this should be changed?

            # train the model
            model.fit(dataset_train,
                      dataset_val,
                      learning_rate=1e-4,
                      batch_size=32,
                      epochs=100,
                      num_samples=1,
                      device=device,
                      model_path=model_path,
                      trial=None,
                      logger=writer)

            # backtest on the test set
            predictions = model.predict(dataset_test,
                                        batch_size=32,
                                        num_samples=3,
                                        device=device,)
            trues = np.array([dataset_test.evalsample(i).values() for i in range(len(dataset_test))])
            trues = (trues - scalers['target'].min_) / scalers['target'].scale_
            predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
            id_errors_sample = test_no_var(trues, predictions)

            # backtest on the ood test set
            predictions = model.predict(dataset_test_ood,
                                        batch_size=32,
                                        num_samples=3,
                                        device=device,)
            trues = np.array([dataset_test_ood.evalsample(i).values() for i in range(len(dataset_test_ood))])
            trues = (trues - scalers['target'].min_) / scalers['target'].scale_
            predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
            ood_errors_sample = test_no_var(trues, predictions)
            
            # compute, save, and print results
            with open(study_file, "a") as f:
                for reduction in reductions:  
                    if reduction is not None:
                        # compute
                        reduction_f = getattr(np, reduction)
                        id_errors_sample_red = reduction_f(id_errors_sample, axis=0)
                        ood_errors_sample_red = reduction_f(ood_errors_sample, axis=0)
                        # save
                        id_errors_cv[reduction].append(id_errors_sample_red)
                        ood_errors_cv[reduction].append(ood_errors_sample_red)
                        # print
                        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID {reduction} of (MSE, MAE): {id_errors_sample_red}\n")
                        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD {reduction} of (MSE, MAE) stats: {ood_errors_sample_red}\n")

        # compute, save, and print results
        with open(study_file, "a") as f:
            for reduction in reductions:
                if reduction is not None:
                    # compute
                    id_errors_cv[reduction] = np.vstack(id_errors_cv[reduction])
                    ood_errors_cv[reduction] = np.vstack(ood_errors_cv[reduction])
                    id_errors_cv[reduction] = np.mean(id_errors_cv[reduction], axis=0)
                    ood_errors_cv[reduction] = np.mean(ood_errors_cv[reduction], axis=0)
                    # save
                    id_errors_model[reduction].append(id_errors_cv[reduction])
                    ood_errors_model[reduction].append(ood_errors_cv[reduction])
                    # print
                    f.write(f"\tModel Seed: {model_seed} ID {reduction} of (MSE, MAE): {id_errors_cv[reduction]}\n")
                    f.write(f"\tModel Seed: {model_seed} OOD {reduction} of (MSE, MAE): {ood_errors_cv[reduction]}\n")
                
    # compute, save, and print results
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # compute mean and std
                id_errors_model[reduction] = np.vstack(id_errors_model[reduction])
                ood_errors_model[reduction] = np.vstack(ood_errors_model[reduction])
                id_mean = np.mean(id_errors_model[reduction], axis=0)
                ood_mean = np.mean(ood_errors_model[reduction], axis=0)
                id_std = np.std(id_errors_model[reduction], axis=0)
                ood_std = np.std(ood_errors_model[reduction], axis=0)
                # print
                f.write(f"ID {reduction} of (MSE, MAE): {id_mean.tolist()} +- {id_std.tolist()}\n")
                f.write(f"OOD {reduction} of (MSE, MAE): {ood_mean.tolist()} +- {ood_std.tolist()}\n")
