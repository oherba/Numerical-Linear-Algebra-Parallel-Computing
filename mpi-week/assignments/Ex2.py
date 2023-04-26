from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

while True:
    if rank == 0:
        value = int(input("Enter an integer: "))
        if value < 0:
            break
    else:
        value = None
                                                            
    value = comm.bcast(value, root=0)
    print("Process", rank, "got", value)
MPI.Finalize()