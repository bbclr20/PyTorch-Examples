import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size):
   group = dist.new_group([0, 1])
   tensor = torch.ones(1)
   dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
   print('Rank ', torch.distributed.get_rank(), ' has data ', tensor[0])


def init_processes(rank, size, fn, backend="tcp"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    process = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        process.append(p)

    for p in process:
        p.join()
