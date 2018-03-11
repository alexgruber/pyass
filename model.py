
import numpy as np

class model(object):

    def __init__(self, mod, *args, **kwargs):
        """ Class that allows making any model updateable"""

        self.mod = mod(*args, **kwargs)
        self.gen = self.run()
        self.gen.send(None)


    def run(self):
        """ Generator for running the model"""

        f = self.mod.f
        while True:
            res = self.mod.step(f)
            f = yield res

    def update(self, x_upd=None, P_upd=None):
        if x_upd is not None:
            self.mod.x[:] = x_upd

        if P_upd is not None:
            self.mod.P[:] = P_upd

    def step(self, *args, **kwargs):
        """ Model step with prior state / error update """

        res = self.mod.step(*args, **kwargs)
        return res


