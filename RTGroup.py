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
import new_data_processing as dp
from data_processing import getCellCounts
data = pd.read_csv("../data/White mice - RT only.csv")
nit_max = 300
nit_T = 200
param_0 = [500000, 0.35, 0.02, 0.003, 0.1, 1.0, 1.5, 1e-06, 0.0, 1.82061785504427e-21, 500.0, 0.03, 0.8492481540336232, 0.3, 0.1, 0.01, 2.0, 0.0674559557659555, 287936.7977312881, 0.3009826699557746, 8.59576707979513e-09, 5.209325451740283e-07, 0, 5.0, 0.1, 1e-08, 20.0, 0.2549737362820806, 0.9595475657773529, 0.04652645764362457, 2.0858842542417699e-106, 0.2, 0, 0.5]
param_id = [2,3,4,10,31] #index of parameters to be changed
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = 0
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 40
delta_t = 0.05
D = np.ones(5)*2
t_rad = np.array([10,11,12,13,14])
t_treat_c4 = np.zeros(3)
t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0
for i in range(1, 15):
  row = getCellCounts(data, i)

  #print(row)
  day_length = int(len(row)/3)
  #t_f2 = row[day_length]
  param_best, *_, MSEs, fittedVolumes = dp.annealing_optimization(row, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov, day_length)
  print(param_best)
  param_best_list.append(param_best)
  times = row[0:day_length]
  T = row[day_length:2*day_length]
  # print(T)
  # print(fittedVolumes)
  plt.figure(figsize=(8,8))

  plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
  plt.plot(np.arange(0,nit_max*nit_T + 1), MSEs, 'o', label='Best MSE')
  plt.title('Plot 1 with 1 set of data')
  plt.legend()

# Creating the second plot with two sets of data on the same plot
  plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
  plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
  plt.plot(times, fittedVolumes, '--', color ='red', label ="optimized Tumor Cell data")
  plt.title('Plot 2 with 2 sets of data')
  plt.legend()

  plt.tight_layout()
  figure_name = "control tumor volume vs time " + str(i) + " .png"
  plt.savefig(figure_name)

end_time = time.time()
time_taken = end_time - start_time
dataFrame = pd.DataFrame(param_best_list[0:])
std_devs = dataFrame.std()
means = dataFrame.mean()
dataFrame.to_csv("best parameters for RT set.csv", index=False)
std_devs.to_csv("errors for RT set.csv", index=False)
means.to_csv("mean of each parameter for RT set.csv", index=False)
f = open("best parameters RT.txt", "w")
f.write("best parameters " + str(param_best))
f.write("\n")
f.write("mean square error " + str(MSEs[-1]))
f.write("\n")
f.write("execution time " + str(time_taken))
f.close()
