import time
import concurrent.futures
start_time = time.time()
fittedVolumeList = []
timesList = []
TList = []
MSEList = []

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from differential_equations import radioimmuno_response_model
import new_data_processing as dp
from data_processing import getCellCounts
data = pd.read_csv("../data/White mice data - PD-1 10.csv")
nit_max = 300
nit_T = 200
param = pd.read_csv("RT means.csv")
#print(param)
param_0 = list(np.transpose(np.array(param))[0])
#print(param_0)
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
param_best_list = []
def process_data(i):
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

iterations = 9  # Or any other number of iterations

# Use a ThreadPoolExecutor to run the iterations in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=iterations) as executor:
    futures = {executor.submit(process_data, i): i for i in range(1, iterations + 1)}
    concurrent.futures.wait(futures)

for i in range(iterations):
    plt.figure(figsize=(8,8))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.plot(np.arange(0,nit_max*nit_T + 1), MSEList[i], 'o', label='Best MSE')
    plt.title('Mean Square Errors (MSEs)')
    plt.legend()

# Creating the second plot with two sets of data on the same plot
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    print(times)
    print(TList[i])
    plt.plot(timesList[i], TList[i], 'o', color ='red', label ="Tumor Cell data")
    plt.plot(timesList[i], fittedVolumeList[i], '--', color ='red', label ="optimized Tumor Cell data")
    plt.xlabel("Time (days)")
    plt.ylabel("Volume (mm^3)")
    plt.title('Tumour Volume vs Time with Model fit')
    plt.legend()

    plt.tight_layout()
    figure_name = "PD-L1 10 tumour volume vs time " + str(i) + " .png"
    plt.savefig(figure_name)

iterations = 6
data = pd.read_csv("../data/White mice data - PD-1 15.csv")
with concurrent.futures.ThreadPoolExecutor(max_workers=iterations) as executor:
    futures = {executor.submit(process_data, i): i for i in range(1, iterations + 1)}
    concurrent.futures.wait(futures)

for i in range(iterations):
    plt.figure(figsize=(8,8))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.plot(np.arange(0,nit_max*nit_T + 1), MSEList[i], 'o', label='Best MSE')
    plt.title('Mean Square Errors (MSEs)')
    plt.legend()

# Creating the second plot with two sets of data on the same plot
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    print(times)
    print(TList[i])
    plt.plot(timesList[i], TList[i], 'o', color ='red', label ="Tumor Cell data")
    plt.plot(timesList[i], fittedVolumeList[i], '--', color ='red', label ="optimized Tumor Cell data")
    plt.xlabel("Time (days)")
    plt.ylabel("Volume (mm^3)")
    plt.title('Tumour Volume vs Time with Model fit')
    plt.legend()

    plt.tight_layout()
    figure_name = "PD-L1 15 tumour volume vs time " + str(i) + " .png"
    plt.savefig(figure_name)

end_time = time.time()
time_taken = end_time - start_time
dataFrame = pd.DataFrame(param_best_list[0:])
std_devs = dataFrame.std()
means = dataFrame.mean()
dataFrame.to_csv("PD-L1 best parameters.csv", index=False)
std_devs.to_csv("PD-L1 errors.csv", index=False)
means.to_csv("PD-L1 means.csv", index=False)
f = open("time taken PD-L1 parallel.txt", "w")
f.write("execution time " + str(time_taken))
f.close()
