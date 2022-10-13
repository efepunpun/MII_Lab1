import numpy as np
import matplotlib.pyplot as plt

K = int(input('Enter a number K = ' ))
N = int(input('Enter a number N greater than 3 = ' ))

if N < 4 or (N % 2 !=0):
    print('N must be greater than 3 and even!')
    exit()
   

n = int(N/2)

B = np.random.randint(-10,10,(n,n))
print(f'SubArray B = \n{B}\n')

C = np.random.randint(0,10,(n,n))
print(f'SubArray C = \n{C}\n')

D = np.random.randint(-10,10,(n,n))
print(f'SubArray D = \n{D}\n')

E = np.random.randint(-10,10,(n,n))
print(f'SubArray E = \n{E}\n')

A = np.vstack([np.hstack([B,C]),np.hstack([D,E])])
print(f'MainArray A = \n{A}\n')

F = A.copy()


SumPerimC = int(C[0,:].sum() + C[n-1,:].sum() + C[:,0].sum() + C[:,n-1].sum() - C[0,0] - C[0,n-1] - C[n-1,0] - C[n-1,n-1])
ProdDiagC = int(np.prod(np.diagonal(C)))  


if SumPerimC > ProdDiagC:
    print('SumPerimC is more than ProdDiagC then symmetrically swap B and C: ')
    B1 = np.flip(B, axis=1)
    C1 = np.flip(C, axis=1)
    F = np.vstack([np.hstack([C1, B1]), np.hstack([D, E])])
else:
    print('SumPerimC is less than ProdDiagC then asymmetrically swap B and E: ')
    B1 = E.copy()
    E1 = B.copy()
    F = np.vstack([np.hstack([B1, C]), np.hstack([D, E1])])

print(F)

if np.linalg.det(A) > (np.diagonal(F).sum() + np.diagonal(np.flip(F, axis=1)).sum()):
    A_trans = np.transpose(A)
    A_inv = np.linalg.inv(A)
    F_inv = np.linalg.inv(F)
    result = A * A_trans - K * F_inv
else:
    A_trans = np.transpose(A)
    F_trans = np.transpose(F)
    G = np.tril(A)
    result = (A_trans + G - F_trans) * K


print("\nResult of expression:")
print(result)


plt.subplot(2, 2, 1)
plt.imshow(F[:int(n), :int(n)], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 2)
plt.imshow(F[:int(n), int(n):], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 3)
plt.imshow(F[int(n):, :int(n)], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 4)
plt.imshow(F[int(n):, int(n):], cmap='rainbow', interpolation='bilinear')
plt.show()

plt.subplot(2, 2, 1)
plt.plot(F[:int(n), :int(n)])
plt.subplot(2, 2, 2)
plt.plot(F[:int(n), int(n):])
plt.subplot(2, 2, 3)
plt.plot(F[int(n):, :int(n)])
plt.subplot(2, 2, 4)
plt.plot(F[int(n):, int(n):])
plt.show()