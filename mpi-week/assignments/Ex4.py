from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 6
m = 4

A = np.arange(n*m, dtype=np.float64).reshape(n, m)

lcl_n = n//2
lcl_m = m//2

lcl_A = np.zeros((lcl_n, lcl_m), dtype=np.float64)

counts = [n*m//4, n*(m-m//2)//2, n*(m-m//2)//2, n*m//4]
displs = [0, n*m//4, n*m//4 + n*(m-m//2)//2, n*m//4 + n*(m-m//2)//2 + n*(m-m//2)//2]

if rank == 0:
    print("Global Matrix:\n", A)

comm.Scatterv([A, counts, displs, MPI.DOUBLE], lcl_A, root=0)

if rank == 1:
    print("Local Matrix on Process 1:\n", lcl_A)

if rank == 2:
    print("Local Matrix on Process 2:\n", lcl_A)

if rank == 3:
    print("Local Matrix on Process 3:\n", lcl_A)