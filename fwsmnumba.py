# Factorization using sympy Matrix and Null space operations.
# from fwsm import factorise
# Here are some samples for usage i don't automatically adjust B and I
# so you may have to play with the numbers for effeciency

# In [91]: factorise(1009732533765211, 2500, 10000)
# Found 20 potential solutions
# 11344301 89007911
# Out[91]: [mpz(11344301), mpz(89007911)]

# In [93]: factorise(32990125356016687985769067)
# Found 46 potential solutions
# 4898499751721 6734740640627
# Out[93]: [mpz(4898499751721), mpz(6734740640627)]

import random
import gmpy2
from gmpy2 import is_prime
from gmpy2 import mpz
import sympy
from sympy import Matrix
import numpy as np
from itertools import chain
import numba, ctypes, ctypes.util, math

def primes_sieve2(limit):
    a = np.ones(limit, dtype=bool)
    a[0] = a[1] = False

    for (i, is_prime) in enumerate(a):
        if is_prime:
            yield i
            for n in range(i*i, limit, i):
                a[n] = False  

def strailing(N):
   return N>>ffs(N)

def ffs(x):
    """Returns the index, counting from 0, of the
    least significant set bit in `x`.
    """
    return (x&-x).bit_length()-1

def get_mod_congruence(root, N, withstats=False):
  r = root - N
  if withstats==True:
    print(f"{root} ≡ {r} mod {N}")
  return r

  
def is_quadratic_residue(a, p):

    if a % p == 0:
        return True
    elif a < 0:
        a = (p - a) % p

    legendre_symbol = pow(a, (p - 1) // 2, p)  # Calculate Legendre symbol using modular exponentiation
    return legendre_symbol == 1
    
    
def matrixMul(A, B):
    TB = zip(*B)
    return [[sum(ea*eb for ea,eb in zip(a,b)) for b in TB] for a in A]
 
def pivotize(m):
    """Creates the pivoting matrix for m."""
    n = len(m)
    ID = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(m[i][j]))
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return ID
 
def LU(A):
    """Decomposes a nxn matrix A by PA=LU and returns L, U and P."""
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]
    P = pivotize(A)
    A2 = matrixMul(P, A)
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A2[i][j] - s1
        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (A2[i][j] - s2) / U[j][j]
    return (L, U, P)

def gauss_elim(M):
#reduced form of gaussian elimination, finds rref and reads off the nullspace
#https://www.cs.umd.edu/~gasarch/TOPICS/factoring/fastgauss.pdf
    
    marks = [False]*len(M[0])
    
    for i in range(len(M)):
        row = M[i]
        #print(row)
        
        for num in row: 
            if num == 1:
                j = row.tolist().index(num) 
                marks[j] = True
                
                for k in chain(range(0,i),range(i+1,len(M))): 
                    if M[k][j] == 1:
                        for i in range(len(M[k])):
                            M[k][i] = (M[k][i] + row[i])%2
                break
    
    M = np.transpose(M)
        
    sol_rows = []
    for i in range(len(marks)):
        if marks[i]== False:
            free_row = [M[i],i]
            sol_rows.append(free_row)
    
    if not sol_rows:
        print("No solution found. Need more smooth numbers.")
    print("Found {} potential solutions".format(len(sol_rows)))
    
                    
def sympy_nullspace_solve(t_matrix, factor_base, smooth_base, root_base, N):
   A = Matrix(t_matrix, dtyp='int8').nullspace()
   cumulitave=1
   Nsqrt=1
   for B in A: 
     for xx in range(len(B)):
       if B[xx] != 0:
         cumulitave*=root_base[xx]
         Nsqrt*=smooth_base[xx]
     factor = gmpy2.gcd(gmpy2.isqrt(Nsqrt)+cumulitave, N)
     if factor == N or factor == 1:
       continue
     else: break    
   return factor

def build_matrix(factor_base, smooth_base):
  factor_base = factor_base.copy()
  factor_base.insert(0, 2)
  
  sparse_matrix = []
  col = 0
  
  for xx in smooth_base:
    sparse_matrix.append([])
    for fx in factor_base:
      count = 0
      factor_found = False
      while xx % fx == 0:
        factor_found = True
        xx=xx//fx
        count+=1
      if count % 2 == 0:
        sparse_matrix[col].append(0)
        continue
      else:
        if factor_found == True:
          sparse_matrix[col].append(1)
        else:
          sparse_matrix[col].append(0)
    col+=1
                
  return np.transpose(sparse_matrix)   

def quad_residue(a,n):
    #checks if a is quad residue of n
    l=1
    q=(n-1)//2
    x = q**l
    if x==0:
        return 1
        
    a =a%n
    z=1
    while x!= 0:
        if x%2==0:
            a=(a **2) % n
            x//= 2
        else:
            x-=1
            z=(z*a) % n

    return z
    

  
def STonelli(n, p): #tonelli-shanks to solve modular square root, x^2 = N (mod p)
    assert quad_residue(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        r = pow(n, (p + 1) // 4, p)
        return r,p-r
    for z in range(2, p):
        #print(quad_residue(z, p))
        if p - 1 == quad_residue(z, p):
            break
    c = pow(z, q, p)
    r = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        for i in range(1, m):
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i

    return (r,p-r)
        
def fb_sm(N, B, I):

   factor_base, sieve_base, sieve_list, smooth_base, root_base, tonelli_base = [], [], [], [], [], []

   primes = list(primes_sieve2(B))
   
   i,root=-1,gmpy2.isqrt(N)
   
   for x in primes[1:]:
       if quad_residue(N, x) == 1:
         factor_base.append(x)

   for x in range(I):
      xx = get_mod_congruence((root+x)**2, N)
      sieve_list.append(xx)
      if xx % 2 == 0:
        xx = strailing(xx+1) #xx // lars_last_modulus_powers_of_two(xx)
      sieve_base.append(xx)


   for p in factor_base:
       residues = STonelli(N, p)
     
       for r in residues:
          for i in range((r-root) % p, len(sieve_list), p):
            while sieve_base[i] % p == 0: 
              sieve_base[i] //= p

   for o in range(len(sieve_list)):
     if len(smooth_base) >= len(factor_base)+1: #probability of no solutions is 2^-T
         break
     if sieve_base[o] == 1:
        smooth_base.append(sieve_list[o])
        root_base.append(root+o)
  
   return factor_base, smooth_base, root_base

@bb.njit
def primes_sieve2(limit):
    a = np.ones(limit, dtype=bool)
    a[0] = a[1] = False

    for (i, is_prime) in enumerate(a):
        if is_prime:
            yield i
            for n in range(i*i, limit, i):
                a[n] = False  
   
def factorise(N, B=10000, I=10000000):

  def primes_sieve2(limit):
    a = np.ones(limit, dtype=bool)
    a[0] = a[1] = False

    for (i, is_prime) in enumerate(a):
        if is_prime:
            yield i
            for n in range(i*i, limit, i):
                a[n] = False  

   global t_matrix

   factors_list = []

   while True:
   
     if is_prime(N):
        factors_list.append(N)
        break

     factor_base, smooth_base, root_base = fb_sm(N,B,I)
     t_matrix = build_matrix(factor_base, smooth_base)
     #L, U, P = LU(t_matrix)
     #gauss_elim(U)  # Replace gauss_elim with LU decomposition
     gauss_elim(t_matrix)

     factor = sympy_nullspace_solve(t_matrix, factor_base, smooth_base, root_base, N)
     
     if is_prime(factor):
       N = N//factor
     else:
       N = factor
     
     factors_list.append(factor)
        
     print(N, factor)
   
   return sorted(factors_list)   
   
   
