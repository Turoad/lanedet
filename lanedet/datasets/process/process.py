import collections

from lanedet.utils import build_from_cfg

from ..registry import PROCESS 

class Process(object):
    """Compose multiple process sequentially.
    Args:
        process (Sequence[dict | callable]): Sequence of process object or
            config dict to be composed.
    """

    def __init__(self, processes, cfg):
        assert isinstance(processes, collections.abc.Sequence)
        self.processes = []
        for process in processes:
            if isinstance(process, dict):
                process = build_from_cfg(process, PROCESS, default_args=dict(cfg=cfg))
                self.processes.append(process)
            elif callable(process):
                self.processes.append(process)
            else:
                raise TypeError('process must be callable or a dict')

    def __call__(self, data):
        """Call function to apply processes sequentially.
        Args:
            data (dict): A result dict contains the data to process.
        Returns:
           dict: Processed data.
        """

        for t in self.processes:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.processes:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
