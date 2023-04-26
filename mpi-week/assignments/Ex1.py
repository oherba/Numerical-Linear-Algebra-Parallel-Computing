#Exercise 1 : Hello World
from mpi4py import MPI


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

print('hello world',RANK,SIZE)


print('hello world from the processor {:d} out of {:d}'.format(RANK, SIZE))

if RANK == 0:
    print('Hello world from processor {:d} out of {}'.format(RANK,SIZE))