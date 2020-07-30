import numpy as np
vec1 = np.array([-1,4,-9],dtype = 'float32')
mat1 = np.array([[1,3,5],[7,-9,2],[4,6,8]],dtype = 'float32')
pi = np.pi
vec2 = pi * vec1
vec2 = np.cos(vec2)
vec3 = vec1 + 2 * vec2
print(vec3)
vec4 = np.matmul(vec3,mat1)
print(vec4)
print('mat1_transposed =',mat1.transpose(),sep = '\n')
print('mat1 =',mat1,sep='\n')
print('mat1_trace =',mat1.trace(),sep='\n')
print('mat1_det= ',np.linalg.det(mat1),sep='\n')
print('mat1_min= ',mat1.min(),sep='\n')
print(np.where(mat1 == mat1.min()))
A = np.array([[17,24,1,8,15],[23,5,7,14,16],[4,6,13,20,22],[10,12,19,21,3],[11,18,25,2,9]])
B = np.sum(A, axis = -1)
C = np.sum(A, axis = -2)
rowMin = np.min(B)
rowMax = np.max(B)
colMin = np.min(C)
colMax = np.max(C)
D = np.diag(A)
trace = np.sum(D)
E = np.fliplr(A)
F = np.diag(E)
othertrace = np.sum(F)
if rowMin == rowMax == colMin == colMax == trace == othertrace:
    print('Magic')
else:
    print('not Magic')
M = np.random.rand(10,10) 
mUL= M[0:5,0:5]
mUR= M[0:5,5:]
mLL = M[5:,0:5]
mLR = M[5:,5:]

