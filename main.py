import torch
import pytorch_lightning as pl

import configparser

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

from ptls.nn import TrxEncoder, RnnEncoder
from ptls.frames.gpt import GptPretrainModule, GptDataset
from ptls.preprocessing import PandasDataPreprocessor
from ptls.data_load.datasets import MemoryMapDataset
from ptls.frames import PtlsDataModule

from dataset import MlmNoSliceDataset
from models import GptPretrainContrastiveModule, NextItemPredictionModule

import os
import pandas as pd

CONFIG_FILE = 'configs.ini'

def get_config_with_dirs(parser, config_ind):
    sections = parser.sections()

    # Pull parameters for a model with specific config
    data_conf = {
        'dataset_name' : parser[sections[config_ind]]['dataset_name'],
        'data_path' : parser[sections[config_ind]]['data_path'],
        'amount_bins' : parser[sections[config_ind]]['amount_bins'],
        'validtest_size' : parser[sections[config_ind]]['validtest_size'],
        'test_to_valid_share' : parser[sections[config_ind]]['test_to_valid_share'],
        'train_batch_size' : parser[sections[config_ind]]['test_to_valid_share'],
        'n_runs' : parser[sections[config_ind]]['n_runs']
    }

    model_conf = {
        'mcc_emb_size' : parser[sections[config_ind]]['mcc_emb_size'],
        'amount_emb_size' : parser[sections[config_ind]]['amount_emb_size'],
        'linear_projection_size' : parser[sections[config_ind]]['linear_projection_size'],
        'seq_hidden_size' : parser[sections[config_ind]]['seq_hidden_size']
    }

    learning_conf = {
        'max_lr_repr' : parser[sections[config_ind]]['max_lr_repr'],
        'total_steps_repr' : parser[sections[config_ind]]['total_steps_repr'],
        'max_lr_contr' : parser[sections[config_ind]]['max_lr_contr'],
        'total_steps_contr' : parser[sections[config_ind]]['total_steps_contr'],
        'max_lr_downstream' : parser[sections[config_ind]]['max_lr_downstream'],
        'total_steps_downstream' : parser[sections[config_ind]]['total_steps_downstream'],
    }

    params_conf = {
        'max_epochs_pretrain' : parser[sections[config_ind]]['max_epochs_pretrain'],
        'max_epochs_downstream' : parser[sections[config_ind]]['max_epochs_downstream'],
        'patience' : parser[sections[config_ind]]['patience'],
        'neg_count' : parser[sections[config_ind]]['neg_count'],
        'loss_temperature' : parser[sections[config_ind]]['loss_temperature'],
    }

    return data_conf, model_conf, learning_conf, params_conf
    

if __name__ == '__main__':
    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_FILE)
    sections = config_parser.sections()
    num_configs = len(sections)

    for config_ind in range(num_configs):
        data_conf, model_conf, learning_conf, params_conf = get_config_with_dirs(config_parser, config_ind = config_ind)
        
        config_name = sections[config_ind]

        # data preprocessing
        dataset_name = data_conf['dataset_name']
        data_path = data_conf['data_path']
        amount_bins = data_conf['amount_bins']
        validtest_size = data_conf['validtest_size']
        test_to_valid_share = data_conf['test_to_valid_share']
        train_batch_size = data_conf['train_batch_size']
        n_runs = data_conf['n_runs']

        # model hyperparameters
        mcc_emb_size = data_conf['mcc_emb_size']
        amount_emb_size = data_conf['amount_emb_size']
        linear_projection_size = data_conf['linear_projection_size']
        seq_hidden_size = data_conf['seq_hidden_size']

        # training hyperparameters
        max_lr_repr = data_conf['max_lr_repr']
        total_steps_repr = data_conf['total_steps_repr']
        max_lr_contr = data_conf['max_lr_contr']
        total_steps_contr = data_conf['total_steps_contr']
        max_lr_downstream = data_conf['max_lr_downstream']
        total_steps_downstream = data_conf['total_steps_downstream']

        # other
        max_epochs_pretrain = data_conf['max_epochs_pretrain']
        max_epochs_downstream = data_conf['max_epochs_downstream']

        patience = data_conf['patience']
        neg_count = data_conf['neg_count']
        loss_temperature = data_conf['loss_temperature']

        source_data = pd.read_csv(data_path)
        if dataset_name == "rosbank":
            source_data['TRDATETIME'] = pd.to_datetime(source_data['TRDATETIME'], format='%d%b%y:%H:%M:%S')
        elif dataset_name == "sberbank":
            source_data = source_data.rename({'trans_date': 'TRDATETIME'}, axis=1)

        source_data = source_data.rename(columns={'cl_id':'client_id', 'MCC':'small_group', 'amount':'amount_rur'})

        mcc_to_id = {mcc: i+1 for i, mcc in enumerate(source_data['small_group'].unique())}

        source_data['amount_rur_bin'] = 1 + KBinsDiscretizer(amount_bins, encode='ordinal', subsample=None).fit_transform(source_data[['amount_rur']]).astype('int')
        source_data['small_group'] = source_data['small_group'].map(mcc_to_id)

        preprocessor = PandasDataPreprocessor(
            col_id='client_id',
            col_event_time='TRDATETIME',
            event_time_transformation='dt_to_timestamp',
            cols_category=['small_group'],
            cols_numerical=['amount_rur_bin'],
            return_records=True,
        )

        dataset = preprocessor.fit_transform(source_data[['client_id', 'TRDATETIME', 'small_group', 'amount_rur_bin']])

        train, valid_test = train_test_split(dataset, test_size=validtest_size, random_state=42)

        valid, test = train_test_split(valid_test, test_size=test_to_valid_share, random_state=42)
        len(train), len(valid), len(test)

        train_dl = PtlsDataModule(
            train_data=GptDataset(
                MemoryMapDataset(
                    data=train,
                ),
                min_len=25, 
                max_len=200
            ),
            valid_data=MlmNoSliceDataset(
                MemoryMapDataset(
                    data=valid,
                ),
            ),
            test_data=MlmNoSliceDataset(
                MemoryMapDataset(
                    data=test,
                ),
            ),
            train_batch_size=train_batch_size,
        )

        for _ in range(n_runs):

            trx_encoder_params = dict(
                embeddings_noise=0.0,
                embeddings={
                    'small_group': {'in': len(mcc_to_id) + 1, 'out': mcc_emb_size},
                    'amount_rur_bin':{'in': amount_bins+1, 'out': amount_emb_size}
                },
                linear_projection_size = linear_projection_size
            )

            seq_encoder = RnnEncoder(
                    input_size=linear_projection_size,
                    hidden_size=seq_hidden_size,
                    type='gru',
            )

            model = GptPretrainModule(
                trx_encoder=TrxEncoder(**trx_encoder_params),
                seq_encoder=seq_encoder,
                max_lr=max_lr_repr,
                total_steps=total_steps_repr
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs_pretrain,
                gpus=1 if torch.cuda.is_available() else 0,
                callbacks=[pl.callbacks.EarlyStopping('gpt/valid_gpt_loss', patience = patience)],
                enable_progress_bar=False,
            )

            trainer.fit(model, train_dl)

            model.trx_encoder.requires_grad_(False)

            model_downstream = NextItemPredictionModule(
                trx_encoder=model.trx_encoder, 
                seq_encoder=seq_encoder, 
                target_col='small_group',
                max_lr=max_lr_downstream,
                total_steps=total_steps_downstream
            )


            trainer = pl.Trainer(
                max_epochs=max_epochs_downstream,
                gpus=1 if torch.cuda.is_available() else 0,
                callbacks=[pl.callbacks.EarlyStopping('gpt/valid_gpt_loss', patience=patience, mode='min')],
                enable_progress_bar=False,
            )

            trainer.fit(model_downstream, train_dl)

            data = {
                "Scores": [trainer.test(model_downstream, train_dl)[0]['gpt/test_f1_weighted']]
            }

            df = pd.DataFrame(data)

            stats = pd.read_csv(f'results/stats_repr_{dataset_name}_config_{config_name}.csv')

            stats = pd.concat([stats, df], ignore_index=True)

            stats.to_csv(f'results/stats_repr_{dataset_name}_config_{config_name}.csv', index = False)

        for _ in range(n_runs):
            trx_encoder_params = dict(
                embeddings_noise=0.0,
                embeddings={
                    'small_group': {'in': len(mcc_to_id) + 1, 'out': mcc_emb_size},
                    'amount_rur_bin':{'in': amount_bins+1, 'out': amount_emb_size}
                },
                linear_projection_size = linear_projection_size
            )

            seq_encoder = RnnEncoder(
                    input_size=linear_projection_size,
                    hidden_size=seq_hidden_size,
                    type='gru',
            )

            model = GptPretrainContrastiveModule(
                trx_encoder=TrxEncoder(**trx_encoder_params),
                seq_encoder=seq_encoder,
                max_lr=max_lr_contr,
                total_steps=total_steps_repr,
                neg_count=neg_count,
                loss_temperature=loss_temperature
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs_pretrain,
                gpus=1 if torch.cuda.is_available() else 0,
                callbacks=[pl.callbacks.EarlyStopping('mlm/valid_mlm_loss', patience=patience)],
                enable_progress_bar=False,
            )

            trainer.fit(model, train_dl)

            model.trx_encoder.requires_grad_(False)

            model_downstream = NextItemPredictionModule(
                trx_encoder=model.trx_encoder, 
                seq_encoder=seq_encoder, 
                target_col='small_group',
                max_lr=max_lr_contr,
                total_steps=total_steps_contr
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs_downstream,
                gpus=1 if torch.cuda.is_available() else 0,
                callbacks=[pl.callbacks.EarlyStopping('gpt/valid_gpt_loss', mode='min', patience=patience)],
                enable_progress_bar=False,
            )

            trainer.fit(model_downstream, train_dl)

            data = {
                "Scores": [trainer.test(model_downstream, train_dl)[0]['gpt/test_f1_weighted']]
            }

            df = pd.DataFrame(data)

            stats = pd.read_csv(f'results/stats_contr_{dataset_name}_config_{config_name}.csv')

            stats = pd.concat([stats, df], ignore_index=True)

            stats.to_csv(f'results/stats_contr_{dataset_name}_config_{config_name}.csv', index = False)
