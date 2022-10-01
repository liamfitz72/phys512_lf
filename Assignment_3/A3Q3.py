# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 08:38:38 2022

@author: liamf
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('dish_zenith.txt')
x_data=data[:,0]
y_data=data[:,1]
z_data=data[:,2]

A=np.empty([len(x_data),4])   
A[:,0]=x_data**2+y_data**2    #Defining A matrix based on parametrization
A[:,1]=x_data
A[:,2]=y_data
A[:,3]=1

u,s,vt=np.linalg.svd(A,False)  #A=USV^T , Singular Value Decompositon
s_inv=np.diag(1/s)   #Since s is diagonal
m=vt.T@s_inv@u.T@z_data # m=VS^-1U^Tx  , model parameters using SVD

#Plotting data points
ax=plt.axes(projection='3d')
ax.scatter(x_data,y_data,z_data)

#Plotting surface fit
max_xy=np.max(np.abs(np.append(x_data,y_data)))  
linspace=np.linspace(-max_xy,max_xy,1001)     #Creating fine mesh of x,y
x_fine,y_fine=np.meshgrid(linspace,linspace)
A_fine=[x_fine**2+y_fine**2,x_fine,y_fine,1]  #A matrix of fine mesh
z_fine=m@A_fine  #Fit parameters on fine mesh

ax.plot_wireframe(x_fine,y_fine,z_fine,rstride=100,cstride=100,color='orange')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.tick_params(labelsize=6)
ax.view_init(elev=40., azim=25)
ax.legend(['Data points','Linear least squares fit'])
plt.savefig('A3Q3_plot1.png',bbox_inches='tight',dpi=300)

#Returning a, x_0, y_0 and z_0
a=m[0]
x_0=m[1]/a
y_0=m[2]/a
z_0=m[3]-a*(x_0**2+y_0**2)
print('a =',a,'\nx_0 =',x_0,'\ny_0 =',y_0,'\nz_0 =',z_0)

#Determination of error in focal point
n=z_data-A@m  #Estimation of noise
N=np.outer(n,n)  
N_inv=np.linalg.inv(N)
m_err=np.linalg.inv(A.T@N_inv@A)@A.T@N_inv@n  # m-m_t=(A^TN^-1A)^-1A^TN^-1n
m_err=np.sqrt(np.abs(m_err))

f=1/(4*m[0])   #f=1/(4a)
f_err=1/(4*m[0]**2)*m_err[0]   #f(a+err)=f(a)+f'(a)*err
print('Focal length =',f,'+/-',f_err)