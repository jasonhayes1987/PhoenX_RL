import cupy
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()

# Allreduce
sendbuf = cupy.arange(10, dtype='i')
recvbuf = cupy.empty_like(sendbuf)
comm.Allreduce(sendbuf, recvbuf)
print(cupy.allclose(recvbuf, sendbuf*size))