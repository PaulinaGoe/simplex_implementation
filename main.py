# Import libraries
import numpy as np
from numpy import linalg as LA
import math

# Define Input
# A_N_in = np.array([[1, 1, 3], [2, 2, 5], [4, 1, 2]])
# b_in = np.array([30, 24, 36])
# cN_in = np.array([3, 1, 2])
A_N_in = np.array([[1, 2], [1, 1], [0, 3]])
b_in = np.array([170, 150, 180])
cN_in = np.array([300, 500])

# Precision parameters
epsilon = 0.0001
nu = 0.0001

# Dimensions
m: int = b_in.shape[0]  # number of constraints (number of basic variables)
n: int = cN_in.shape[0]  # number of (non-basic) variables

# Auxiliary variable matrix and vector
A_B_in = np.identity(m)  # basic variable matrix
cB_in = np.zeros(m)  # basic variable vector
A_B = np.zeros((m, m))
A = np.zeros((m,n))
b = np.zeros(m)
Gate_FC = []
Gate_FR = []
Gate_IsOpt = []
Gate_IsUnbd = []
Gates_Classic = []

# Define Matrices
A_T = np.zeros((m+1, m+n+1))  # tableau matrix

# Define vectors
x = np.zeros(n+m)  # solution vector

# Auxiliary variables
alpha = np.zeros(m)  # ratio test
Basis = np.zeros(m)  # index set of basic variables
RC = np.zeros(m+n)  # reduced cost vector

# Counter
leav = 0  # leaving variable index
ent = 0  # entering variable index
iteration = 0 #iteration counter


# Define Basis set
for i in range(0, m):
    Basis[i] = n + i  # only variables with index in range n to n+m are in the basis

# Define simplex tableau
for i in range(1, m+1):
    A_T[i, 0] = b_in[i - 1]  # rows 1 to m+1 of the first column equal vector b

    for j in range(1, n+1):
        A_T[i, j] = A_N_in[i-1, j-1]  # columns 1 to n+1 without first entry (first row) equal columns of input matrix A

    for k in range(n+1, m+n+1):
        A_T[i, k] = A_B_in[i-1, k-n-m-1]  # columns n+1 to m+n+1 without first entry (first row) equal initial basis matrix (here: unit matrix)

for j in range(1, n+1):
    A_T[0, j] = cN_in[j-1]  # columns 1 to n+1 of the first row equal input vector c

for j in range(n+1, m+1):
    A_T[0, j] = cB_in[j-1]  # columns n+1 to n+m+1 of the first row equal initial basis vector c (here: zeros)

# Initialize objective value z
A_T[0, 0] = 0  # entry of row 0 and column 0 equals objective value

# Initialize RC
for i in range(1, n+1):
    RC[i-1] = A_T[0, i]  # columns 1 to n+1 of first row equal the entries of the reduced cost vector

print("Initialization:", A_T, RC, Basis)


# Simplex

# continue loop as long as the reduced cost vector contains at least one positive component
while RC.any() > 0:

    print("Iteration:", iteration)
    iteration = iteration+1

    # Column sparsity
    cspars = []
    for j in range(n + 1, n + m + 1):
        l=0
        for i in range(1, m + 1):
            if A_T[i, j] == 0:
                l = l + 1
            cspars.append(l)

    d_c = np.amax(cspars)

    print("Column sparsity:", d_c)

    # Row sparsity
    rspars = []
    for i in range( 1, m + 1):
        v = 0
        for j in range(n+1, n+ m + 1):
            if A_T[i, j] == 0:
                v = v + 1
            rspars.append(l)

    d_r = np.amax(rspars)

    print("Row sparsity:", d_r)

    d = np.maximum(d_r, d_c)

    print("Sparsity:", d)


    # Determine eta

    for i in range(1, m + 1):
        b[i - 1] = A_T[i, 0]  #  vector b

        for j in range(1, n + 1):
            A[i-1, j-1] = A_T[i, j]  #  matrix A

        for k in range(n + 1, m + n + 1):
            A_B[i-1, k-n-m-1] = A_T[i, k]   #  basis matrix


    MAX =[]
    for i in range(0,n-1):
        Amax = LA.norm(A[:,i], np.inf)
        MAX.append(Amax)

    InfnormA = np.amax(MAX)

    eta = np.maximum(InfnormA, LA.norm(b, np.inf))

    print("Eta:", eta)

    # Determine condition number
    kappa = LA.cond(A_B)

    print("Condition number:", kappa)

    # Contribution to complexity in Nannicini

    Gate_IsOpt.append((1/epsilon)*kappa*d*math.sqrt(n)*(d_c*n+d*n))

    Gate_FC.append((1/epsilon)*kappa*d*math.sqrt(n)*(d_c*n+d*n))

    Gate_IsUnbd.append((1/nu)*kappa*(d**2)*(m**1.5))

    Gate_FR.append((1/nu)*eta*(kappa**2)*(d**2)*(m**1.5))

    # Contribution to complexity classically

    Gates_Classic.append((d_c**0.7)*(m**1.9)+(m**3)+d_c*n)


    #print(Gate_FC, Gate_FR)

    # Determine maximum reduced cost
    for i in range(1, n+m+1):
        RC[i - 1] = A_T[0, i]  # determine current reduced cost vector (RC)

    for i in range(1, n+m+1):
        if RC[i-1] == np.amax(RC):  # determine maximum of current RC
            ent = i  # index of maximum of RC + 1 is the entering variable

    print("Entering variable:", ent)

    # Ratio test
    for j in range(1, m+1):
        if A_T[j, ent] != 0:  # fix column with index ent (pivot column) and look for elements unequal to zero
            alpha[j-1] = np.divide(A_T[j, 0], A_T[j, ent])  # divide j-th element of b by (j,ent)-th element of current A
        else:
            print("IS UNBOUNDED")
            break

    delta = np.amin(alpha[alpha > 0])  # determine minimum among all positive elements of alpha
    print("delta:", delta)

    alpha_list = alpha.tolist()  # convert alpha to list because I do not know how to access index of an array
    leav = alpha_list.index(delta)+1  # the index of minmum of amon all positive elements of alpha +1 is the leaving variable

    print("Leaving variable:", leav)

    # Update basis
    for i in range(0, m):
        if Basis[i] == leav+n-1:  # if the leaving index is in the basis set exchange it with the entering index
            Basis[i] = ent

    print("New basis:", Basis)

    # Update simplex tableau

    # Divide pivot row by leading term
    pivot_element = A_T[leav, ent]

    for i in range(0, m+n+1):
        A_T[leav, i] = A_T[leav, i]/pivot_element

    pivot_element = A_T[leav, ent]  # update pivot element

    # Elimination of pivot column for each row except from the pivot row
    for i in range(0, m+1):

        if i != leav:
            A_T[i, :] = A_T[i, :] - np.divide(A_T[i, ent], pivot_element) * A_T[leav, :]

    print(A_T)

    # Current objective value
    z = (-1)*A_T[0, 0]

    # Update RC
    for i in range(1, n+m+1):
        RC[i-1] = A_T[0, i]
    print(RC)


    # Break condition
    if np.amax(RC) <= 0:  # if all components of RC are negative or zero, we have found the optimal solution and thus break the loop
        print("OPTIMALITY REACHED!")
        break

# Determine solution vector
for i in range(1, n+m+1):
    for j in range(1, m+1):
        if A_T[j, i] == 1:
            x[i-1] = A_T[j, 0]

# Determine complexity of Nannicini

Total_gates_FR = sum(Gate_FR)

Total_gates_FC = sum(Gate_FC)

Total_gates_IsOpt = sum(Gate_IsOpt)

Total_gates_IsUnbd = sum(Gate_IsUnbd)

Total_No_It_FC = iteration*math.sqrt(n)

Total_gates_Quantum = Total_gates_FR + Total_gates_FC + Total_gates_IsOpt + Total_gates_IsUnbd

# Gate count classical simplex

Total_gates_Classic = sum(Gates_Classic)

print("Objective value z=", z, "with solution vector x=", x)
print("O_FR:", Total_gates_FR,"\n", "O_FC:", Total_gates_FC, "\n",  "O_IsUnbd:", Total_gates_IsUnbd, "\n", "O_IsOpt:", Total_gates_IsOpt, "\n",  "Total no of gates \n Nannicni:", Total_gates_Quantum, "\n", "Classical Simplex:", Total_gates_Classic)