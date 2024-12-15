import numpy as np
import matplotlib.pyplot as plt
from Functions import a_lin, a_lin_multiplesources


def generate_source(N):
    real = (np.random.randn(round(N)))
    imag = (np.random.randn(round(N)))*1j

    source = real + imag
    source = source/ np.linalg.norm(source)

    return source

source = generate_source(5)
print(source)
print(np.linalg.norm(source))


def datamodel(M,N,theta0,d,v,f0):
    A = a_lin_multiplesources(theta0,M,d,v,f0)

    S = np.zeros((len(theta0),N),dtype="complex")
    for i in range(0,len(theta0)):
        s = generate_source(N)
        S[i,:] = s

    X = np.matmul(A,S)
    return X

M=6
N=50
theta0 = [0,np.pi/12]
d = 0.1
v=340
f0=100

X= datamodel(M,N,theta0,d,v,f0)

U, S, Vh = np.linalg.svd(X)

V = Vh.conj().T

print(np.shape(U))
print(S)
print(len(V))