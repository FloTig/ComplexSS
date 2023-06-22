

#Import modules 
import numpy as np 
#Module for arrays
import matplotlib.pyplot as plt
#Module for graphics
#Model parameters
f=1.5
q=0.015
e=0.59
#Numerical method parameters
t0=0.0 #Initial time
tf=100.0 #Final time
dt = 0.0001 #Time step
#Initial Condition
X0=0.5
Z0=0.5
#Functions
def fun1(x, z):
    return (x*(1-x)+f*z*((q-x)/(q+x)))/e
def fun2(x, z):
    return x-z
#Lists for data
X=[] #List for x variable
Z=[] #List for z varibale 
T=[] #List for time values
#Assign initial condition
x = X0
z = Z0
#Numerical method
for t in np.arange(t0, tf, dt):
    t=t+dt
    #Euler rule for x variable 
    x=fun1(x, z)*dt+x
    # Euler rule for z variable
    z=fun2(x, z) *dt+z
    X.append(x) #List for x variable 
    Z.append(z) #List for z variable
    T.append(t) #List for t variable

#Plot construction
f, axes = plt.subplots(1,2,figsize= (14,6))
axes[0].plot(T, X, 'b.', label='x')
axes[0].plot(T, Z, 'k', label='z')
axes[0].set_title('Oscillations',size=16)
axes[0].set_xlabel('t [adimensional]', size=14)
axes[0].set_ylabel('x,z [adimensional]', size=14)
axes[0].legend(loc='best')
axes[1].plot(X,Z, 'g', label='Phase Space')
axes[1].set_title('phase space', size =16)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].legend(loc='best')
plt.show()