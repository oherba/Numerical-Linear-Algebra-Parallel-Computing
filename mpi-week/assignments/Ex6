from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 840

pi_part = 0.0
for i in range(rank, N, size):
    pi_part += 1.0/(1.0 + ((i + 0.5)/N)**2)
pi_part *= 4.0/N
#Print the computed value of pi for each process
print("Process", rank, "partial sum:", pi_part)
if rank == 0:
    pi = pi_part
    for i in range(1, size):
        pi += comm.recv(source=i)
else:
    comm.send(pi_part, dest=0)
if rank == 0:
    print("Computed pi:", pi)
comm.Barrier() 
t1 = MPI.Wtime()
num_runs = 10000 
for i in range(num_runs):
    pi_part = 0.0
    for j in range(rank, N, size):
        #Divide the iterations among the processes
        pi_part += 1.0/(1.0 + ((j + 0.5)/N)**2)
    pi_part *= 4.0/N
    #Collect and sum the partial results on the rank 0 process
    if rank == 0:
        pi = pi_part
        for j in range(1, size):
            pi += comm.recv(source=j)
    else:
        comm.send(pi_part, dest=0)
    if rank == 0 and i == 0:
        print("Computed pi:", pi)
comm.Barrier()
t2 = MPI.Wtime()
if rank == 0:
    print("Taken Time:", t2 - t1)