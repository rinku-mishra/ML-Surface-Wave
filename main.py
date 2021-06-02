#!/usr/bin/env python3
import operator

import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

from polydata import dataSet
from polyreg import polyReg

def plot_figure(fig, ax, x_train,y_train,y_poly_pred,dataf):
    # Plot training data
    ax.scatter(x_train, y_train, s=10, label='data for length: %0.2f'%dataf)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x_train,y_poly_pred), key=sort_axis)
    x_train, y_poly_pred = zip(*sorted_zip)
    ax.plot(x_train, y_poly_pred, label='regress for length: %0.2f'%dataf)
    # plt.yscale('log')




if __name__ == "__main__":
    # istart,iend = 1,11

    score_param = 'RMSE'
    savedir = 'data'
    length = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    num_data = len(length)
    degree_high = 20
    rmse_all = np.zeros(degree_high)
    r2_all = np.zeros(degree_high)
    mae_all = np.zeros(degree_high)
    y_poly_pred_all = np.zeros([num_data,10])
    print(y_poly_pred_all.shape)
    opt_deg = np.zeros(num_data,dtype=int)

    fig, ax = plt.subplots(figsize=(10,6))

    for i in range(1,num_data+1): #len_files
        for j in range(1,degree_high+1): #poly_deg
            x_train, y_train = dataSet(i,i+1)
            y_poly_pred, rmse, r2, mae = polyReg(x_train, y_train,j)
            rmse_all[j-1] = rmse
            r2_all[j-1] = r2
            mae_all[j-1] = mae
            print('len = %d'%i+' degree = %d'%j+' RMSE = %f'%rmse+' R2 = %f'%r2+' MAE = %f'%mae)
        if score_param == 'MAE':
            opt_deg[i-1] = np.where(mae_all == np.min(mae_all))[0][0] + 1
        elif score_param == 'R2':
            opt_deg[i-1] = np.where(r2_all == np.max(r2_all))[0][0] + 1
        elif score_param == 'RMSE':
            opt_deg[i-1] = np.where(rmse_all == np.min(rmse_all))[0][0] + 1
        print(f'Optimum degree: {opt_deg[i-1]:d} based on {score_param:s} score')
        y_poly_pred, rmse, r2, mae = polyReg(x_train, y_train,opt_deg[i-1])
        y_poly_pred_all[i-1,:] = y_poly_pred[:,0]
        plot_figure(fig, ax,x_train,y_train,y_poly_pred,length[i-1])
    np.savez_compressed(pjoin(savedir,'training_data.npz'),length=length,poly_deg=opt_deg,x_train = x_train, y_poly_pred=y_poly_pred_all)
    ax.legend()
    plt.show()
