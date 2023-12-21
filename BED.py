import numpy as np
def get_equivalent_bed_treatment(param, D_BED_0, n):
    # Calculate BED_0
    #param2 is alpha, param3 is beta
    BED_0 = (1 / (param[2] / param[3])) * D_BED_0[0] * (D_BED_0[1] ** 2) + D_BED_0[0] * D_BED_0[1]
    print(BED_0)
    m = n

    # Calculate the dose per fraction for each treatment given by the number of fractions and the equivalent BED
    d = np.zeros(m)
    for i in range(m):
        d[i] = (-1*n + np.sqrt(n ** 2 - 4 * n / (param[2] / param[3]) * (-1*BED_0))) / (2 * n / (param[2] / param[3]))
    return d
