import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


# def run(rank, size):
#     """Blocking point-to-point communication."""
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 3
#         dist.send(tensor=tensor, dst=3)
#     elif rank == 3:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor)


def run(rank, size):
    """Non-blocking point-to-point communication."""
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 3
        req = dist.isend(tensor=tensor, dst=3)
        print('Rank 0 started sending')
    elif rank == 3:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    if req is not None:
        req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
