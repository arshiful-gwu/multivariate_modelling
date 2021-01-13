import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal
import copy

def cal_GPAC(acf_values,j_max,k_max):
    gpac_ndarray=np.zeros((j_max,k_max-1))
    for k in range(1,k_max):
        for j in range(0,j_max):
            #form the denominator matrix (k*k)
            den_mat=np.zeros((k,k))
            for row in range(k):
                for col in range(k):
                    den_mat[row][col]=acf_values[abs(j+row-col)]
    #form the numerator matrix (same as denominator matrix except for last column)
            num_mat=copy.deepcopy(den_mat)
            for row in range(k):
                num_mat[row][k-1]=acf_values[j+row+1]

            det_num=np.linalg.det(num_mat)
            det_den=np.linalg.det(den_mat)
            gpac_ndarray[j][k-1]=det_num/det_den
    
    # return the GPAC ndarray
    return gpac_ndarray

def autocorrelation_cal(y,k):
    T=len(y)
    mean_y=np.mean(y)
    numerator=0
    denominator=0
    T_k=0
    
    for t in range(0,T):
        denominator=denominator+(np.square(y[t]-mean_y))
    for t in range(k,T):
        numerator=numerator+((y[t]-mean_y)*(y[t-k]-mean_y))
        T_k=numerator/denominator
    return T_k

def acf_values(y,ml):
    #lags=[]
    autoCorr=[]
    max_lag=ml
    for i in range(0,max_lag):
        #lags.append(i)
        autoCorr.append(autocorrelation_cal(y,i))

    return autoCorr

       
def calc_e(y,na,theta):
    num = [1]
    den = [1]
    den=np.concatenate((den,theta[0:na]))
    num=np.concatenate((num,theta[na:]))

    if len(num)<len(den):
        z=np.zeros(len(den)-len(num))
        num=np.concatenate((num,z),axis=None)
    elif len(num)>len(den):
        z=np.zeros(len(num)-len(den))
        den=np.concatenate((den,z),axis=None)

    system = (den,num,1)
    T=len(y)
    t_in=np.arange(0,T)
    t_out, e = signal.dlsim(system,y,t=t_in)
    return e

def levenburgMarquardtStepOne(y,na,nb,theta,delta,N,n):
    e=calc_e(y,na,theta)
    E=np.mat(e)
    SSE=E.T.dot(E)
    X=np.zeros((N,n))
    X=np.mat(X)
    for i in range (0,n):#1 ≤ i ≤ n
        theta_copy=copy.deepcopy(theta)
        theta_copy[i]=theta_copy[i]+delta
        e2=calc_e(y,na,theta_copy)
        x=e-e2
        x=x/delta
        X[:,i]=x
    A=X.T.dot(X)
    g=X.T.dot(e)

    return A,g,SSE

def levenburgMarquardtStepTwo(y,na,nb,theta,A,g,mu,n):
    I=np.identity(n)
    del_theta=np.linalg.inv(A+(mu*I)).dot(g)
    del_theta_arr=np.array(del_theta).flatten()
    theta_new=theta+del_theta_arr
    e_new=calc_e(y,na,theta_new)
    E_NEW=np.mat(e_new)
    SSE_NEW=E_NEW.T.dot(E_NEW)
    return SSE_NEW,del_theta_arr,theta_new
    
def levenburgMarquardt(y,na,nb,numOfIter):
    # returns the estimated parameter
    # input parameters are:
    # y (generated using arma process)
    # order of ar process in arma, na
    # order of ma process in ma, nb

    #step 1
    # defining maximum number of iteration
    
    # the very first theta    
    N=len(y)
    n=na+nb
    theta=np.zeros((n))
    delta=0.001
    A,g,SSE=levenburgMarquardtStepOne(y,na,nb,theta,delta,N,n)

    mu=0.01
    SSE_NEW,del_theta,theta_new=levenburgMarquardtStepTwo(y,na,nb,theta,A,g,mu,n)

    iterator=0
    maxIterations=numOfIter
    mu_max=10000000000 
     
    while iterator < maxIterations:
        if SSE_NEW < SSE:
            mag_del_theta = np.linalg.norm(del_theta) 
            if mag_del_theta < 1:
                theta=theta_new
                sigma_e_sq=SSE_NEW/(N-n)
                cov=np.multiply(sigma_e_sq,np.linalg.inv(A))
                conf=np.diagonal(np.sqrt(cov))

                print("i="+str(iterator)+", SSE new less than SSE old, ||del_theta||<0.001 :")
                print("theta=")
                print(theta)
                print("confidence interval = +/-"+str(conf))
                print("Estimated variance of error:")
                print(sigma_e_sq)
                print("covariance matrix:")
                print(cov)
                print("SSE=")
                print(SSE)
                
                
                break
                #return theta
                
            else:
                theta=theta_new
                mu=mu/10

        while SSE_NEW > SSE:
            mu=mu*10
            if mu>mu_max:
                print("i="+str(iterator)+", mu>mu_max")
                print(theta)
                
                #return theta
            SSE_NEW,del_theta,theta_new=levenburgMarquardtStepTwo(y,na,nb,theta,A,g,mu,n)
           
        iterator=iterator+1
        if iterator>maxIterations:
            print("i="+str(iterator)+", iter > maxIter")
            print(theta)
            #return theta
            
            
        theta=theta_new
        A,g,SSE=levenburgMarquardtStepOne(y,na,nb,theta,delta,N,n)
        SSE_NEW,del_theta,theta_new=levenburgMarquardtStepTwo(y,na,nb,theta,A,g,mu,n)

    return theta  