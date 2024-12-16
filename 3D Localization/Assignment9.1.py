import numpy as np
import matplotlib.pyplot as plt


def a_lin(theta, M, d, v, f0):
    #Create empty complex vector
    a = (np.ones((M,1),dtype="complex_"))
    for i in range(0,M):
        #Compute each entry for vector
        a[i] = a[i] * np.exp(-1j*(i)*d/v*np.sin(theta)*2*np.pi*f0)
        
    return a

def a_z(s_position,mic_positions, M,v,f0):
    a = (np.ones((M,1),dtype="complex_"))

    for i in range(M):
        a[i]=    1/np.linalg.norm(s_position-mic_positions[i,:])*np.exp(-1j*f0/(2*np.pi)*np.linalg.norm(s_position-mic_positions[i,:])/v)

    return a


mic_positions=np.array( [[2.5 ,5.0, 0],
                [2.5,10,0 ],
                [2.5,15,0 ],
                [7.5,5,0 ],
                [7.5,10,0],
                [7.5,15,0]])
v=40
f0=150
s_position= [0,0,0]
M=6


print(a_z(s_position,mic_positions,M,v,f0))