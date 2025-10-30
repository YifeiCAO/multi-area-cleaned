"""

"""
from __future__ import absolute_import

import importlib.util
import inspect
import os
import sys
import pdb

import numpy as np

from .defaults import defaults, generate_trial

THIS = 'pycog.model'


class Struct():
    """
    Treat a dictionary like a module.

    """

    def __init__(self, **entries):
        self.__dict__.update(entries)


class Model(object):
    """
    Provide a simpler interface to users, and check for obvious mistakes.

    """

    def __init__(self, modelfile=None, **kwargs):
        """
        If `modelfile` is provided infer everything from the file, otherwise the
        user is responsible for providing the necessary parameters through `kwargs`.

        Parameters
        ----------

        modelfile: str
                   A Python script containing model parameters.

        kwargs : dict

          Should contain

          generate_trial : function
                           Return a dictionary containing trial information.
                           This function should take as its arguments

                           rng    : numpy.random.RandomState
                           dt     : float
                           params : dict

        """

        if modelfile is not None:
            try:
                # Load the model file as a module using importlib (Py3 compatible)
                spec = importlib.util.spec_from_file_location('model', modelfile)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.m = module
            except IOError:
                print("[ {}.Model ] Couldn't load model file {}".format(THIS, modelfile))
                sys.exit(1)
        else:
            self.m = Struct(**kwargs)

        #---------------------------------------------------------------------------------
        # Perform model check
        #---------------------------------------------------------------------------------

        # generate_trial : existence
        try:
            f = self.m.generate_trial
            try:
                args = list(inspect.signature(f).parameters.keys())
            except Exception:
                args = inspect.getfullargspec(f).args
        except AttributeError:
            try:
                f = self.m.task.generate_trial
                try:
                    args = list(inspect.signature(f).parameters.keys())[1:]
                except Exception:
                    args = inspect.getfullargspec(f).args[1:]
            except AttributeError:
                print("[ {}.Model ] You need to define a function that returns trials."
                      .format(THIS))
                sys.exit(1)

        # generate_trial : usage
        if args != ['rng', 'dt', 'params']:
            print(("[ {}.Model ] Warning: Function generate_trial doesn't have the"
                   " expected list of argument names. It is OK if only the names are"
                   " different.").format(THIS))

        # var_in : size
            # checks that var_in exists and is an np.ndarray, as well as that it's dim is s 1.
        if (hasattr(self.m, 'var_in') and isinstance(self.m.var_in, np.ndarray)
                and self.m.var_in.ndim == 1):
            if len(self.m.var_in) != self.m.Nin:
                print("[ {}.Model ] The length of var_in doesn't match Nin.".format(THIS))
                sys.exit(1)

        # if terminate is given, performance should also be given
        if hasattr(self.m, 'terminate') and not hasattr(self.m, 'performance'):
            print(("[ {}.Model ] Warning: Termination criterion is provided, "
                   " but the performance measure is not defined").format(THIS))

    #/////////////////////////////////////////////////////////////////////////////////////

    def train(self, savefile, seed=None, compiledir=None, recover=True, gpus=0, lambdar=None, lambdaw=None, lr=None):
        """
        Train the network.

        Parameters
        ----------

        savefile : str
        seed : int, optional
        compiledir : str, optional
        recover : bool, optional
        gpus : int, optional

        """
        # Theano setup
        os.environ.setdefault('THEANO_FLAGS', '')
        if compiledir is not None:
            os.environ['THEANO_FLAGS'] += ',base_compiledir=' + compiledir
        os.environ['THEANO_FLAGS'] += ',floatX=float32,allow_gc=False'
        if gpus > 0:
            os.environ['THEANO_FLAGS'] += ',device=cuda{}'.format(gpus - 1)

        # Only involve Theano for training
        from .trainer import Trainer

        # The task
        try:
            task = self.m.task
        except AttributeError:
            task = Struct(generate_trial=self.m.generate_trial)

        # Parameters
        params = {}

        # Seed, if given
        if seed is not None:
            params['seed'] = seed

        # Optional parameters
        for k in defaults:
            if hasattr(self.m, k):
                params[k] = getattr(self.m, k)

        # lambda_r
        if lambdar is not None:
            params['lambda2_r'] = lambdar

        # MK -- Jan 8, 2020
        # weight regularization
        if lambdaw is not None:
            params['lambda2_in'] = lambdaw
            params['lambda2_rec'] = lambdaw
            params['lambda2_out'] = lambdaw

        if lr is not None:
            params['learning_rate'] = lr
        # Train

        # initialize the trainers with the parameters from the model file.
        trainer = Trainer(params)
        trainer.train(savefile, task, recover=recover)
