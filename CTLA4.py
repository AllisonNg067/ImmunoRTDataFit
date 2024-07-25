import time
start_time = time.time()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from differential_equations import radioimmuno_response_model
import new_data_processing as dp
from data_processing import getCellCounts
data = pd.read_csv("White mice data - PD-1 10 CTLA.csv")
#print(len(data.columns))
nit_max = 20
nit_T = 20
#param = [500000.0,0.4043764660304215,0.3348825012978519,0.0444792478133069,0.051958556519556,1.0,1.5,1.0000000000000004e-06,0.0,1.82061785504427e-21,1.6370031265953986,0.0492366376540959,0.6416711700693031,0.1617903030935988,0.0416924514099231,0.00416924514099231,2.0,0.0674559557659555,299838.7440652358,0.198909083172271,9.211522519585746e-09,8.284618352937945e-07,4.722366482869665e-51,5.0,0.00001,2e-5,0.1379556739056123,0.4073542114448485,0.0481351408570356,0.0099999999999999,1.1404642118810832e-106,0.0053955684115056,0.2,0.1]
#print(param)
param = [500000.0, 0.4043764660304215, 0.3348825012978519, 0.0444792478133069, 0.051958556519556, 1.0, 1.5, 1.0000000000000004e-06, 0.0, 1.82061785504427e-21, 1.6370031265953986, 0.0492366376540959, 0.6416711700693031, 0.1617903030935988, 0.0416924514099231, 0.00416924514099231, 2.0,
         0.0674559557659555, 299838.7440652358, 0.198909083172271, 9.211522519585746e-09, 8.284618352937945e-07, 2, 5.0, 1e-5, 2e-5, 0.1379556739056123, 0.4073542114448485, 0.0481351408570356, 0.0099999999999999, 1.1404642118810832e-106, 0.0053955684115056, 0.2, 0.1, 100]
# param = [500000.0, 0.4043764660304215, 0.3348825012978519, 0.0444792478133069, 0.051958556519556, 1.0, 1.5, 1.0000000000000004e-06, 0.0, 1.82061785504427e-21, 1.6370031265953986, 0.0492366376540959, 0.6416711700693031, 0.1617903030935988, 0.0416924514099231, 0.00416924514099231, 2.0,
#           0.0674559557659555, 299838.7440652358, 0.198909083172271, 9.211522519585746e-09, 8.284618352937945e-07, 4.722366482869665e-51, 5.0, 0.00001, 2e-5, 0.1379556739056123, 0.4073542114448485, 0.0481351408570356, 0.0099999999999999, 1.1404642118810832e-106, 0.0053955684115056, 0.2, 0.1]
param_id = [24, 25, -1] #index of parameters to be changed
free = [1,1,0]
param[-3] = 0.2
param[22] = 0.2 #ctla4
param[-1] = 10**20 #multiplier
#param[25]= 0.1
param_0 = param
#param_0[25] = 2e-5
#param_0[26] = 17
#param_0[28] = 0.2
#param_0[33] = 0.1
#param_0[25] = 0.2
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
t_treat_c4 = np.array([10])
t_treat_p1 = np.array([10,12,14])
c4 = 0.2
p1 = 0.2
param_best_list = []
param_0 = [500000.0, 0.4043764660304215, 0.3348825012978519, 0.0444792478133069, 0.051958556519556, 1.0, 1.5, 1.0000000000000004e-06, 0.0, 1.82061785504427e-21, 1.6370031265953986, 0.0492366376540959, 0.6416711700693031, 0.1617903030935988, 0.0416924514099231, 0.00416924514099231, 2.0, 0.0674559557659555, 299838.7440652358, 0.198909083172271, 9.211522519585746e-09, 8.284618352937945e-07, 0.2, 5.0, 12.130294122628728, 49.48496229768121, 0.1379556739056123, 0.4073542114448485, 0.0481351408570356, 0.0099999999999999, 1.1404642118810832e-106, 0.0053955684115056, 0.2, 0.1, 1.327659830343167e+25]
for i in range(1,5):
  #print('mouse number', i)
  param_0[32] = p1
  param_0[22] = c4
  row = getCellCounts(data, i)

  #print('row', row)
  day_length = int(len(row)/3)
  t_f2 = row[day_length - 1] + 1
  #print('max t', t_f2)
  param_best, *_, MSEs, _ = dp.annealing_optimization(row, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov, day_length)
  #print(param_best == param_0)
  param_best_list.append(param_best)
  times = row[0:day_length]
  #print('times', times)
  T = row[day_length:2*day_length]
  #print('T', T)
  fittedVolumes, _, Time, *_ = radioimmuno_response_model(param_best, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #crop fitted volumes so that its same size as array of data volumes
  indexes = [index for index, item in enumerate(Time) if item in times]
  #print('times', Time)
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
  plt.title('Volume vs Time for PD 1 and CTLA4 Treatment')
  plt.legend()

  plt.tight_layout()
  figure_name = "anti PD L1 10 CTLA4 tumour volume vs time " + str(i) + " .png"
  plt.savefig(figure_name)

print()
print("NEXT SET")
data = pd.read_csv("White mice data - PD1-15 CTLA.csv")
t_treat_p1 = np.array([15,17,19])
t_treat_c4 = np.array([15])
for i in range(1,9):
  param_0[32] = p1
  param_0[22] = c4
  if i in [1,2,6]:
      param_0[-1] = 10**60
  else:
      param_0[-1] = 1.327659830343167e+25
  row = getCellCounts(data, i)

  #print(row)
  day_length = int(len(row)/3)
  t_f2 = row[day_length - 1] + 1
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

#   plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
#   plt.plot(np.arange(0,nit_max*nit_T + 1), MSEs, 'o', label='Best MSE')
#   plt.title('MSEs')
#   plt.legend()

# # Creating the second plot with two sets of data on the same plot
#   plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
#   plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
#   plt.plot(times, fitVolumesCropped, '--', color ='red', label ="optimized Tumor Cell data")
#   plt.title('Volume vs Time for PD 1 and CTLA4 Treatment')
#   plt.legend()

#   plt.tight_layout()
  plt.plot(times, T, 'o', color ='red', label ="Tumor Cell data")
  plt.plot(times, fitVolumesCropped, '--', color ='red', label ="Optimized Model Fit")
  plt.title('Tumour Volume vs Time with anti-PD-L1 and anti-CTLA-4')
  plt.legend()
  plt.xlabel('Time (Days)')  # Replace 'Time (units)' with the appropriate label for the x-axis
  plt.ylabel('Volume (mm$^3$)')  # Replace 'Volume (units)' with the appropriate label for the y-axis
  figure_name = "anti PD L1 15 CTLA4 tumour volume vs time " + str(i) + " .png"
  plt.savefig(figure_name)
print(param_best_list)

end_time = time.time()
time_taken = end_time - start_time
dataFrame = pd.DataFrame(param_best_list[0:])
std_devs = dataFrame.std()
means = dataFrame.mean()
dataFrame.to_csv("PD1 CTLA4 best parameters.csv", index=False)
std_devs.to_csv("PD1 CTLA4 errors.csv", index=False)
means.to_csv("PD1 CTLA4 mean.csv", index=False)
f = open("PD1 CTLA4 time.txt", "w")
f.write("execution time " + str(time_taken))
f.close()
