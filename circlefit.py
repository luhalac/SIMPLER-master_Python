# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:38:49 2021

@author: Lucia
"""

import numpy as np
from scipy.spatial.distance import pdist

def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None



def CircleFit(xy):
 
# Circle fit by Pratt
#  V. Pratt, "Direct least-squares fitting of algebraic surfaces",
#  Computer Graphics, Vol. 21, pages 145-152 (1987)

# Input:  XY(n,2) is the array of coordinates of n points x(i)=XY(i,1), y(i)=XY(i,2)

# Output: Par = [a b R] is the fitting circle:
#                       center (a,b) and radius R


    n = np.size(xy,0) # number of data points
    x = xy[:,0]
    y = xy[:,1]
    
    centroidx = np.mean(x) # the centroid of the data set
    centroidy = np.mean(y)
    # computing moments (note: all moments will be normed, i.e. divided by n)
    xc = x - centroidx # centering data
    yc = y - centroidy # centering data
    z = xc*xc + yc*yc
    Mxy = np.sum(xc*yc)/n
    Mxx = np.sum(xc*xc)/n
    Myy = np.sum(yc*yc)/n
    Mxz = np.sum(xc*z)/n
    Myz = np.sum(yc*z)/n
    Mzz = np.sum(z*z)/n

    # computing the coefficients of the characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx*Myy - Mxy*Mxy
    Mxz2 = Mxz*Mxz
    Myz2 = Myz*Myz
    A2 = 4*Cov_xy - 3*Mz*Mz - Mzz
    A1 = Mzz*Mz + 4*Cov_xy*Mz - Mxz2 - Myz2 - Mz*Mz*Mz
    A0 = Mxz2*Myy + Myz2*Mxx - Mzz*Cov_xy - 2*Mxz*Myz*Mxy + Mz*Mz*Cov_xy
    A22 = A2 + A2
    
    # Newton's method starting at x=0
    epsilon=1e-12 
    ynew=1e+20
    IterMax=1000
    xnew = 0

    for iter in np.arange(IterMax):
        yold = ynew;
        ynew = A0 + xnew*(A1 + xnew*(A2 + 4.*xnew*xnew))
        if (abs(ynew)>abs(yold)):
            print('Newton-Pratt goes wrong direction: |ynew| > |yold|');
            xnew = 0
            break
        
        Dy = A1 + xnew*(A22 + 16*xnew*xnew)
        xold = xnew
        xnew = xold - ynew/Dy
        
        if (abs((xnew-xold)/xnew) < epsilon): 
            break
        
        if (iter >= IterMax):
            print('Newton-Pratt will not converge')
            xnew = 0
            
        if (xnew<0):
            print('Newton-Pratt negative root')
            xnew = 0

    # computing the circle parameters
    DET = xnew*xnew - xnew*Mz + Cov_xy
    Center = [Mxz*(Myy-xnew)-Myz*Mxy , Myz*(Mxx-xnew)-Mxz*Mxy]/DET/2
    Par_1 = np.array((Center[0]+centroidx, Center[1]+centroidy, np.sqrt(np.sum(Center*Center)+Mz+2*xnew)))
 

    # Compute diferences from fitted radius and actual radius for each point
 
    dist_centroid = np.zeros((n))

    for i in np.arange(n):
        a = np.array((xy[i,:],[Par_1[1], Par_1[2]]))
        dist_centroid[i] = 100*np.sqrt((Par_1[2]-pdist(a))**2)/Par_1[2]
    
    percent_excluded = 25
    
    
    Par = Par_1*np.mean((np.percentile(dist_centroid,percent_excluded),np.percentile(dist_centroid,int(100-percent_excluded))))
    
    return Par_1