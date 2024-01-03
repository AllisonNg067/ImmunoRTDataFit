import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from differential_equations import radioimmuno_response_model
import new_data_processing as dp
from data_processing import getCellCounts
data = pd.read_csv("../data/White mice data - PD-1 10.csv")
print(len(data.columns))
nit_max = 100
nit_T = 100
param = pd.read_csv("mean of each parameter for RT set.csv")
#print(param)
param_0 = list(np.transpose(np.array(param))[0])
param_id = [26, 27, 28, 33] #index of parameters to be changed
free = [1,1,0]
param_0[-2] = 0.2
param_0[26] = 17
param_0[28] = 0.2
LQL = 0
activate_vd = 0
use_Markov = 0
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 90
delta_t = 0.05
D = np.zeros(3)
t_rad = np.zeros(3)
t_treat_c4 = np.zeros(3)
t_treat_p1 = np.array([10,12,14])
c4 = 0
p1 = 0.2
param_best_list = []
for i in range(1,9):
  param_0[32] = 0.2
  row = getCellCounts(data, i)

  #print(row)
  day_length = int(len(row)/3)
  #t_f2 = row[day_length]
  param_best, *_, MSEs, _ = dp.annealing_optimization(row, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov, day_length)
  print(param_best)
  param_best_list.append(param_best)
  times = row[0:day_length]
  T = row[day_length:2*day_length]
  fittedVolumes, _, Time, *_ = radioimmuno_response_model(param_best, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #crop fitted volumes so that its same size as array of data volumes
  indexes = [index for index, item in enumerate(Time) if item in times]
  fitVolumesCropped = [fittedVolumes[0][index] for index in indexes]
  # print(T)
  # print(fittedVolumes)
  plt.figure(figsize=(8,8))

  #plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
  #plt.plot(np.arange(0,nit_max*nit_T + 1), MSEs, 'o', label='Best MSE')
  #plt.title('MSEs')
  #plt.legend()

# Creating the second plot with two sets of data on the same plot
  #plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
  plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
  plt.plot(times, fitVolumesCropped, '--', color ='red', label ="optimized Tumor Cell data")
  plt.title('Volume vs Time for PD 1 Treatment')
  plt.legend()

  plt.tight_layout()
  figure_name = "anti PD L1 10 tumour volume vs time " + str(i) + " .png"
  plt.savefig(figure_name)

print()
print("NEXT SET")
data = pd.read_csv("../data/White mice data - PD-1 15.csv")
t_treat_p1 = np.array([15,17,19])
for i in range(1,7):
  param_0[32] = 0.2
  row = getCellCounts(data, i)

  #print(row)
  day_length = int(len(row)/3)
  #t_f2 = row[day_length]
  param_best, *_, MSEs, _ = dp.annealing_optimization(row, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov, day_length)
  print(param_best)
  param_best_list.append(param_best)
  times = row[0:day_length]
  T = row[day_length:2*day_length]
  fittedVolumes, _, Time, *_ = radioimmuno_response_model(param_best, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #crop fitted volumes so that its same size as array of data volumes
  indexes = [index for index, item in enumerate(Time) if item in times]
  fitVolumesCropped = [fittedVolumes[0][index] for index in indexes]
  # print(T)
  # print(fittedVolumes)
  plt.figure(figsize=(8,8))

  plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
  plt.plot(np.arange(0,nit_max*nit_T + 1), MSEs, 'o', label='Best MSE')
  plt.title('MSEs')
  plt.legend()

# Creating the second plot with two sets of data on the same plot
  plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
  plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
  plt.plot(times, fitVolumesCropped, '--', color ='red', label ="optimized Tumor Cell data")
  plt.title('Volume vs Time for PD 1 Treatment')
  plt.legend()

  plt.tight_layout()
  figure_name = "anti PD L1 15 tumour volume vs time " + str(i) + " .png"
  plt.savefig(figure_name)
print(param_best_list)

end_time = time.time()
time_taken = end_time - start_time
dataFrame = pd.DataFrame(param_best_list[0:])
std_devs = dataFrame.std()
means = dataFrame.mean()
dataFrame.to_csv(file_name, index=False)
std_devs.to_csv("PD1 errors.csv", index=False)
means.to_csv("PD1 mean.csv", index=False)
f = open("PD1 time.txt", "w")
f.write("execution time " + str(time_taken))
f.close()
