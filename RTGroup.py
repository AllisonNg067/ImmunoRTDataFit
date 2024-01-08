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
nit_max = 100
nit_T = 100
param = pd.read_csv("mean of each parameter for control set.csv")
#print(param)
param_0 = list(np.transpose(np.array(param))[0])
#print(param_0)
param[26] = 10
param_id = [2,3,4,10,31] #index of parameters to be changed
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = 0
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 50
delta_t = 0.05
D = np.ones(5)*2
t_rad = np.array([10,11,12,13,14])
t_treat_c4 = np.zeros(3)
t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0
param_best_list = []
for i in range(1, 15):
  row = getCellCounts(data, i)

  #print(row)
  day_length = int(len(row)/3)
  #t_f2 = row[day_length]
  param_best, *_, MSEs, _ = dp.annealing_optimization(row, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov, day_length)
  print(param_best)
  param_best_list.append(param_best)
  times = row[0:day_length]
  T = row[day_length:2*day_length]
  # print(T)
  # print(fittedVolumes)
  fittedVolumes, _, Time, *_ = radioimmuno_response_model(param_best, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #crop fitted volumes so that its same size as array of data volumes
  indexes = [index for index, item in enumerate(Time) if item in times]
  fitVolumesCropped = [fittedVolumes[0][index] for index in indexes]
  plt.figure(figsize=(8,8))

  #plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
  #plt.plot(np.arange(0,nit_max*nit_T + 1), MSEs, 'o', label='Best MSE')
  #plt.title('Plot 1 with 1 set of data')
  #plt.legend()

# Creating the second plot with two sets of data on the same plot
  #plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
  plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
  plt.plot(times, fitVolumesCropped, '--', color ='red', label ="optimized Tumor Cell data")
  plt.title('Tumour Volume vs Time after RT')
  plt.legend()

  plt.tight_layout()
  figure_name = "RT tumor volume vs time " + str(i) + " .png"
  plt.savefig(figure_name)

end_time = time.time()
time_taken = end_time - start_time
dataFrame = pd.DataFrame(param_best_list[0:])
std_devs = dataFrame.std()
means = dataFrame.mean()
dataFrame.to_csv("best parameters for RT set.csv", index=False)
std_devs.to_csv("errors for RT set.csv", index=False)
means.to_csv("mean of each parameter for RT set.csv", index=False)
f = open("time taken RT.txt", "w")
f.write("execution time " + str(time_taken))
f.close()
