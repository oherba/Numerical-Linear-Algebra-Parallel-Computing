from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
while(1):
    if rank == 0:
        data = int(input("Enter an integer: "))
        comm.send(data, 1)
    else:
        data = comm.recv(source = rank - 1)
        if rank < size - 1:
            if data < 0: data -= rank
            comm.send(data + rank, rank + 1)
    if data < 0:
        break
    print("rank:", rank, ",data:",data)