import numpy as np
from scipy.sparse import csr_matrix
from numpy.random import rand, seed
#from numba import njit
from mpi4py import MPI


''' This program compute parallel csc matrix vector multiplication using mpi '''

COMM = MPI.COMM_WORLD
nbOfproc = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)

def matrixVectorMult(A, b, x):
    try:
        row, col = A.shape
        for i in range(row):
            a = A[i]
            for j in range(col):
                x[i] += a[j] * b[j]
    except:
        return 0

########################initialize matrix A and vector b ######################
#matrix sizes
SIZE = 1000
Local_size = SIZE // nbOfproc

counts = [Local_size] * nbOfproc
counts[-1] += SIZE % nbOfproc

#########Send b to all procs and scatter A (each proc has its own local matrix#####
LocalMatrix = np.empty((Local_size, SIZE))
b_local = np.empty(Local_size)

if RANK == 0:
    A = rand(SIZE, SIZE)
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A = csr_matrix(A)
    b = rand(SIZE)
    
    # Scatter the matrix A
    sendcounts = tuple(counts)
    displs = tuple(np.cumsum((0,) + sendcounts[:-1]))
    COMM.Scatterv([A.data, sendcounts, displs, MPI.DOUBLE], LocalMatrix)

    # Scatter the vector b
    COMM.Scatterv([b, sendcounts, displs, MPI.DOUBLE], b_local)

else :
    A = None
    b = None




#####################Compute A*b locally#######################################
LocalX = np.zeros(Local_size)


start = MPI.Wtime()
matrixVectorMult(LocalMatrix, b_local, LocalX)
stop = MPI.Wtime()
if RANK == 0:
    print("CPU time of parallel multiplication is ", (stop - start)*1000)

##################Gather the results ###########################################
# Gather the local results into X
recvcounts = tuple(counts)
displs = tuple(np.cumsum((0,) + recvcounts[:-1]))
X = np.empty(SIZE)

COMM.Gatherv([LocalX, Local_size, MPI.DOUBLE], [X, recvcounts, displs, MPI.DOUBLE])

##################Print the results ###########################################

if RANK == 0 :
    X_ = A.dot(b)
    print("The result of A*b using dot is :", np.max(X_ - X))
    #print("The result of A*b using parallel version is :", X)