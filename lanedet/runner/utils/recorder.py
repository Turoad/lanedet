from collections import deque, defaultdict
import torch
import os
import datetime
from .logger import get_logger


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class Recorder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = self.get_work_dir()
        cfg.work_dir = self.work_dir
        self.log_path = os.path.join(self.work_dir, 'log.txt')

        self.logger = get_logger('lanedet', self.log_path)
        self.logger.info('Config: \n' + cfg.text)

        # scalars
        self.epoch = 0
        self.step = 0
        self.loss_stats = defaultdict(SmoothedValue)
        self.batch_time = SmoothedValue()
        self.data_time = SmoothedValue()
        self.max_iter = self.cfg.total_iter 
        self.lr = 0.

    def get_work_dir(self):
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        hyper_param_str = '_lr_%1.0e_b_%d' % (self.cfg.optimizer.lr, self.cfg.batch_size)
        work_dir = os.path.join(self.cfg.work_dirs, now + hyper_param_str)
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        return work_dir

    def update_loss_stats(self, loss_dict):
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        self.logger.info(self)
        # self.write(str(self))

    def write(self, content):
        with open(self.log_path, 'a+') as f:
            f.write(content)
            f.write('\n')

    def state_dict(self):
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        self.step = scalar_dict['step']

    def __str__(self):
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', 'lr: {:.4f}', '{}', 'data: {:.4f}', 'batch: {:.4f}', 'eta: {}'])
        eta_seconds = self.batch_time.global_avg * (self.max_iter - self.step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        return recording_state.format(self.epoch, self.step, self.lr, loss_state, self.data_time.avg, self.batch_time.avg, eta_string)


def build_recorder(cfg):
    return Recorder(cfg)

