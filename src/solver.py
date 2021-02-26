import os
import sys
import abc
import math
import yaml
import torch
from shutil import copyfile, SameFileError
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized, get_rank, get_world_size

from src.option import default_hparas
from src.util import human_format, Timer

DOWNSAMPLE_RATE_FROM_WAV = 320


class BaseSolver():
    ''' 
    Prototype Solver for all kinds of tasks
    Arguments
        config - yaml-styled config
        paras  - argparse outcome
    '''

    def __init__(self, config, paras, mode):
        # General Settings
        self.config = config
        self.paras = paras
        self.mode = mode
        for k, v in default_hparas.items():
            setattr(self, k, v)
        self.device = torch.device(
            'cuda') if self.paras.gpu and torch.cuda.is_available() else torch.device('cpu')

        # Name experiment
        self.exp_name = paras.name
        if self.exp_name is None:
            # By default, exp is named after config file
            self.exp_name = paras.config.split('/')[-1].replace('.yaml', '')
            if mode == 'train':
                self.exp_name += '_sd{}'.format(paras.seed)

        # Plugin list
        self.emb_decoder = None

        if mode == 'train':
            if not is_initialized() or get_rank() == 0:
                # Filepath setup
                os.makedirs(paras.ckpdir, exist_ok=True)
                self.ckpdir = os.path.join(paras.ckpdir, self.exp_name)
                os.makedirs(self.ckpdir, exist_ok=True)
                self._backup_config(self.paras.config)

                # Logger settings
                self.logdir = os.path.join(paras.logdir, self.exp_name)
                self.log = SummaryWriter(
                    self.logdir, flush_secs=self.TB_FLUSH_FREQ)

            self.timer = Timer()

            # Hyperparameters
            self.step = 0
            self.n_epochs = 0
            self.valid_step = config['hparas']['valid_step']
            self.max_step = config['hparas']['max_step']

            self.verbose('Exp. name : {}'.format(self.exp_name))
            self.verbose('Loading data... large corpus may took a while.')

        elif mode == 'test':
            if not is_initialized() or get_rank() == 0:
                # Output path
                os.makedirs(paras.outdir, exist_ok=True)
                self.ckpdir = os.path.join(paras.outdir, self.exp_name)
                os.makedirs(self.ckpdir, exist_ok=True)
                self._backup_config(self.paras.config)
                self._backup_config(self.config['src']['config'])

            # Load training config to get acoustic feat, text encoder and build model
            self.src_config = yaml.load(
                open(config['src']['config'], 'r'), Loader=yaml.FullLoader)
            self.paras.load = config['src']['ckpt']

            self.verbose('Evaluating result of tr. config @ {}'.format(
                config['src']['config']))
    
    def _backup_config(self, filepath):
        try:
            copyfile(filepath, f'{self.ckpdir}/{os.path.basename(filepath)}')
        except SameFileError:
            pass

    def set_upstream(self):
        '''Setup pretrained Upstream model'''
        if is_initialized() and get_rank() > 0:
            # While rank 0 is downloading, all other processes wait and set
            # their upstream_refresh to False to prevent double downloading
            self.paras.upstream_refresh = False
            torch.distributed.barrier()

        self.upstream = torch.hub.load(
            's3prl/s3prl',
            self.paras.upstream,
            feature_selection = self.paras.upstream_feature_selection,
            refresh = self.paras.upstream_refresh,
            ckpt = self.paras.upstream_ckpt,
            force_reload = self.paras.upstream_refresh,
        ).to(device=self.device)

        if is_initialized() and get_rank() == 0:
            # After rank 0 downloaded the latest checkpoints, notify
            # others to use this newly downloaded checkpoints
            torch.distributed.barrier()

        if is_initialized():
            self.upstream = DDP(self.upstream, device_ids=[self.paras.local_rank], find_unused_parameters=True)
            setattr(self.upstream, 'get_output_dim', self.upstream.module.get_output_dim)
            setattr(self.upstream, 'get_downsample_rate', self.upstream.module.get_downsample_rate)

        if self.paras.upstream_trainable:
            self.upstream.train()
        else:
            self.upstream.eval()

        self.feat_dim = self.upstream.get_output_dim()

    def upstream_extractor(self, wav, wav_len):
        def extract(wav, wav_len):
            ds = DOWNSAMPLE_RATE_FROM_WAV // self.upstream.get_downsample_rate() if self.paras.upstream_same_stride else 1
            feat = self.upstream([w[:l].view(-1)[::ds].to(self.device) for w, l in zip(wav, wav_len)])
            feat_len = torch.LongTensor([len(f) for f in feat])
            feat = pad_sequence(feat, batch_first=True)
            return feat, feat_len

        if self.upstream.training:
            feat, feat_len = extract(wav, wav_len)
        else:
            with torch.no_grad():
                feat, feat_len = extract(wav, wav_len)

        return feat, feat_len

    def backward(self, loss):
        '''
        Standard backward step with self.timer and debugger
        Arguments
            loss - the loss to perform loss.backward()
        '''
        self.timer.set()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.GRAD_CLIP)
        if math.isnan(grad_norm):
            self.verbose('Error : grad norm is NaN @ step '+str(self.step))
        else:
            self.optimizer.step()
        self.timer.cnt('bw')
        return grad_norm

    def load_ckpt(self):
        ''' Load ckpt if --load option is specified '''

        def modify_state_fn(state_dict, vocab_size):
            for key in list(state_dict.keys()):
                value = state_dict[key]
                new_key = key
                if self.paras.load_ddp_to_nonddp:
                    new_key = '.'.join(key.split('.')[1:])
                if self.paras.load_nonddp_to_ddp:
                    new_key = f'module.{key}'
                state_dict.pop(key)
                state_dict[new_key] = value
            if state_dict['ctc_layer.bias'].shape[0] != vocab_size:
                print('model vocab mismatch:', state_dict['ctc_layer.bias'].shape[0], '!=', vocab_size, flush=True)
                print('reinit ctc layer!', flush=True)
                new_bias = torch.zeros(vocab_size).normal_(0.1)
                new_weight = torch.zeros(vocab_size, state_dict['ctc_layer.weight'].shape[1]).normal_(0.1)
                new_bias[:state_dict['ctc_layer.bias'].shape[0]].copy_(state_dict['ctc_layer.bias'])
                new_weight[:state_dict['ctc_layer.weight'].shape[0], :].copy_(state_dict['ctc_layer.weight'])
                state_dict['ctc_layer.bias'] = new_bias
                state_dict['ctc_layer.weight'] = new_weight
            return state_dict

        if self.paras.load:
            # Load weights
            ckpt = torch.load(
                self.paras.load, map_location=self.device if self.mode == 'train' else 'cpu')
            self.model.load_state_dict(modify_state_fn(ckpt['model'], self.model.state_dict()['ctc_layer.bias'].shape[0]))
            if self.emb_decoder is not None:
                self.emb_decoder.load_state_dict(modify_state_fn(ckpt['emb_decoder']))
            if hasattr(self, 'upstream') and self.paras.upstream_trainable:
                self.upstream.load_state_dict(modify_state_fn(ckpt['upstream']))

            # Load task-dependent items
            metric = "None"
            score = 0.0
            for k, v in ckpt.items():
                if type(v) is float:
                    metric, score = k, v
            if self.mode == 'train':
                if not self.paras.reinit_optimizer:
                    self.step = ckpt['global_step']
                    self.optimizer.load_opt_state_dict(ckpt['optimizer'])
                self.verbose('Load ckpt from {}, restarting at step {} (recorded {} = {:.2f} %)'.format(
                              self.paras.load, self.step, metric, score))
            else:
                self.model.eval()
                if self.emb_decoder is not None:
                    self.emb_decoder.eval()
                if hasattr(self, 'upstream'):
                    self.upstream.eval()
                self.verbose('Evaluation target = {} (recorded {} = {:.2f} %)'.format(self.paras.load, metric, score))

    def verbose(self, msg):
        ''' Verbose function for print information to stdout'''
        if is_initialized() and get_rank() > 0: return

        if self.paras.verbose:
            if type(msg) == list:
                for m in msg:
                    print('[INFO]', m.ljust(100), flush=True)
            else:
                print('[INFO]', msg.ljust(100), flush=True)

    def progress(self, msg):
        ''' Verbose function for updating progress on stdout (do not include newline) '''
        if is_initialized() and get_rank() > 0: return

        if self.paras.verbose:
            sys.stdout.write("\033[K")  # Clear line
            print('[{}] {}'.format(human_format(self.step), msg), end='\r', flush=True)

    def write_log(self, log_name, log_dict):
        '''
        Write log to TensorBoard
            log_name  - <str> Name of tensorboard variable 
            log_value - <dict>/<array> Value of variable (e.g. dict of losses), passed if value = None
        '''
        if is_initialized() and get_rank() > 0: return

        if type(log_dict) is dict:
            log_dict = {key: val for key, val in log_dict.items() if (
                val is not None and not math.isnan(val))}
        if log_dict is None:
            pass
        elif len(log_dict) > 0:
            if 'align' in log_name or 'spec' in log_name:
                img, form = log_dict
                self.log.add_image(
                    log_name, img, global_step=self.step, dataformats=form)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_dict, self.step)
            else:
                self.log.add_scalars(log_name, log_dict, self.step)

    def save_checkpoint(self, f_name, metric, score, show_msg=True):
        '''' 
        Ckpt saver
            f_name - <str> the name phnof ckpt file (w/o prefix) to store, overwrite if existed
            score  - <float> The value of metric used to evaluate model
        '''
        if is_initialized() and get_rank() > 0: return

        ckpt_path = os.path.join(self.ckpdir, f_name)
        full_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.get_opt_state_dict(),
            "global_step": self.step,
            metric: score
        }

        if self.emb_decoder is not None:
            full_dict['emb_decoder'] = self.emb_decoder.state_dict()
        if self.paras.upstream_trainable:
            full_dict['upstream'] = self.upstream.state_dict()
        if hasattr(self, 'scaler'):
            full_dict['scaler'] = self.scaler.state_dict()

        torch.save(full_dict, ckpt_path)
        if show_msg:
            self.verbose("Saved checkpoint (step = {}, {} = {:.2f}) and status @ {}".
                         format(human_format(self.step), metric, score, ckpt_path))

    # ----------------------------------- Abtract Methods ------------------------------------------ #
    @abc.abstractmethod
    def load_data(self):
        '''
        Called by main to load all data
        After this call, data related attributes should be setup (e.g. self.tr_set, self.dev_set)
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def set_model(self):
        '''
        Called by main to set models
        After this call, model related attributes should be setup (e.g. self.l2_loss)
        The followings MUST be setup
            - self.model (torch.nn.Module)
            - self.optimizer (src.Optimizer),
                init. w/ self.optimizer = src.Optimizer(self.model.parameters(),**self.config['hparas'])
        Loading pre-trained model should also be performed here 
        No return value
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def exec(self):
        '''
        Called by main to execute training/inference
        '''
        raise NotImplementedError
