import chainer
import copy
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda, reporter
from chainer.training import extensions
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module

class MyEvaluator(extensions.Evaluator):
    def __init__(self, iterator, Rec,
        converter=convert.concat_examples, device=0, eval_hook=None,
        eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self._targets = {'main':Rec}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def evaluate(self):
        iterator = self._iterators['main']
        self.Rec = self._targets['main']
        #入力データ
        xp = np if int(self.device) == -1 else cuda.cupy
        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                input = self.converter(batch, self.device)
                x_batch = xp.array(input[0]) #入力データ
                t_batch = xp.array(input[1]) #教師データ
                #loss
                self.loss = 0
                #計算開始
                with chainer.using_config('train', False),\
                    chainer.using_config('enable_backprop', False):
                        y = self.Rec(x_batch)
                        self.loss = F.mean_squared_error(y, t_batch)
                observation['val/loss'] = self.loss
            summary.add(observation)
        return summary.compute_mean()