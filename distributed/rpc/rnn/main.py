import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import rnn

from functools import wraps

import unittest

def run_worker(name, rank, func, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    rpc.init_rpc(name, rank=rank, world_size=world_size)

    func()

    # block until all rpcs finish
    rpc.shutdown()

def run_ps():
    r"""
    parameter server, do nothing, trainer will contact parameter server during
    training.
    """
    pass

def run_trainer():
    r"""
    The trainer creates a distributed RNNModel and a DistributedOptimizer. Then,
    it performs training on using random input data.
    """
    batch = 5
    ntoken = 10
    ninp = 2

    nhid = 3
    nindices = 3
    nlayers = 4
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )

    model = rnn.RNNModel('ps', ntoken, ninp, nhid, nlayers)

    # setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    # train for 10 iterations
    for epoch in range(10):
        # create distributed autograd context
        with dist_autograd.context():
            inp = torch.LongTensor(batch, nindices) % ntoken
            hidden[0].detach_()
            hidden[1].detach_()
            output, hidden = model(inp, hidden)
            # run distributed backward pass
            dist_autograd.backward([output.sum()])
            # run distributed optimizer
            opt.step()
            # not necessary to zero grads as each iteration creates a different
            # distributed autograd context which hosts different grads
            print("Training epoch {}".format(epoch))


if __name__=="__main__":
    mp.set_start_method('spawn')
    ps = mp.Process(target=run_worker, args=("ps", 0, run_ps, 2))
    ps.start()

    trainer = mp.Process(target=run_worker, args=("trainer", 1, run_trainer, 2))
    trainer.start()
    print("1111")
    ps.join()
    trainer.join()
