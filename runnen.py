# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:15:29 2022

@author: hester
"""

import matplotlib as plt
import math
import numpy as np
import scipy.linalg as lg
import functions as f
import time
from numpy.linalg import norm


def main():
    
    p = 4
    h = 1/(2**p)
    n = 2**p
    A_h = f.matrix_2d(f.matrix_1d(n),f.semi_I(n))
    #start_time1 = time.time()
    #L, U = f.LU(A_h)
    #print(plt.pyplot.imshow(L))
    #print(plt.pyplot.imshow(A_h))
    #print(A_h)
    #print("--- %s seconds ---" % (time.time() - start_time1))
    #start_time2 = time.time()
    f_h = f.functionvalues2(n)
    #u_h = f.LU_solver(L,U,f_h)
    #print(u_h)
    #print("--- %s seconds ---" % (time.time() - start_time2))
    u_h2, residual = f.SOR(A_h,f_h,1000) 
    print(u_h2)
    print(residual)
    
    res = norm(residual)
    f_norm = norm(f_h)
    error = res / f_norm
    print(error)
    
    # u_ex = f.u_2d(n)
    
    # length = len(u_ex)
    # max_distance = 0 
    
    # for i in range(length):
    #     if abs(u_h[i] - u_ex[i]) > max_distance:
    #         max_distance = abs(u_h[i] - u_ex[i])
    
    #return max_distance, u_h
    return
    
if __name__ == "__main__":
    main()