import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda, reporter
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module

class MyUpdater(training.StandardUpdater):
    def __init__(self, iterator, Rec, opt,
        converter=convert.concat_examples, device=0):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.Rec = Rec
        self._optimizers = {'main':opt}
        self.converter = converter
        self.device = device
        self.iteration = 0

    def update_core(self):
        iterator = self._iterators['main'].next()
        #入力データ
        input = self.converter(iterator, self.device)
        xp = np if int(self.device) == -1 else cuda.cupy
        x_batch = xp.array(input[0]) #入力データ
        t_batch = xp.array(input[1]) #教師データ
        #loss
        self.loss = 0
        #計算開始
        y = self.Rec(x_batch)
        self.loss = F.mean_squared_error(y, t_batch)
        #誤差伝播
        self._optimizers['main'].target.cleargrads()
        self.loss.backward()
        self._optimizers['main'].update()
        reporter.report({'main/loss':self.loss})
