# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:02:17 2023

@author: allis
"""
import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from differential_equations import radioimmuno_response_model
import data_processing as dp
data = pd.read_csv("../data/White mice - no treatment.csv")
nit_max = 300
nit_T = 200
param_0 = [500000, 0.35, 0.02, 0.003, 0.1, 1.0, 1.5, 1e-06, 0.0, 1.82061785504427e-21, 500.0, 0.03, 0.8492481540336232, 0.3, 0.1, 0.01, 2.0, 0.0674559557659555, 287936.7977312881, 0.3009826699557746, 8.59576707979513e-09, 5.209325451740283e-07, 0, 5.0, 0.1, 1e-08, 20.0, 0.2549737362820806, 0.9595475657773529, 0.04652645764362457, 2.0858842542417699e-106, 0.2, 0, 0.5]
param_id = [1,11,12,13,14,18,19,20,21, 22, 25, 26, 27, 28,30,32] #index of parameters to be changed
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = 0
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 40
delta_t = 0.05
D = np.zeros(3)
t_rad = np.zeros(3)
t_treat_c4 = np.zeros(3)
t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0
param_best, *_, MSEs, _ = dp.annealing_optimization(data, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov)
for i in range(1, 17):
    row = dp.getCellCounts(data, i)
    day_length = int(len(row)/3)
    times = row[0:day_length]
    T = row[day_length:2*day_length]
    fittedVolumes, *_ = radioimmuno_response_model(param_best, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
    ind = np.rint(row[0:day_length].astype(int) / delta_t).astype(int)
    #print(fittedVolumes)
    vols = fittedVolumes[0][ind]
  # print(T)
  # print(fittedVolumes)
    plt.figure(figsize=(8,8))

    figure_name = "tumor volume vs time " + str(i) + " .png"

# Creating the second plot with two sets of data on the same plot
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
    plt.plot(times, vols, '--', color ='red', label ="optimized Tumor Cell data")
    plt.title('Tumour Volume vs Time')
    plt.legend()
    plt.savefig(figure_name)



plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
plt.plot(np.arange(0,nit_max*nit_T + 1), MSEs, 'o', label='Best MSE')
plt.title('Mean Square Errors')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("MSEs.png")

end_time = time.time()
time_taken = end_time - start_time

f = open("best parameters control.txt", "w")
f.write("best parameters " + str(param_best))
f.write("\n")
f.write("mean square error " + str(MSEs[-1]))
f.write("\n")
f.write("execution time " + str(time_taken))
f.close()
