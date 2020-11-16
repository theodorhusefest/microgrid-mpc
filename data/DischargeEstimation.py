import json
import numpy as np
from sklearn.linear_model import LinearRegression

with open('new_data_format_OCT20.json') as file:
    data_dict = json.load(file)

C_bat = 1000  #Measured in KWh

SOC = data_dict['field']['data'][14]
Current = data_dict['field']['data'][12]
IsItCharging = data_dict['field']['data'][16]
IsItDisCharging = data_dict['field']['data'][17]
Energy = data_dict['field']['data'][13]
Voltage = data_dict['field']['data'][15]
y = []
x = []

#Set sign_check equal 1 for charge, and -1 for discharge.
sign_check = 1

for i in range(len(Energy)):
    if Energy[i] == Energy[i] and np.sign(SOC[i+1] - SOC[i]) == sign_check and np.sign(Energy[i+1] - Energy[i]) == sign_check:
        SOC_diff = np.abs(SOC[i+1] - SOC[i])/100 * C_bat
        ActEnergy = np.abs(Energy[i+1] - Energy[i])
        if SOC_diff == SOC_diff:
            y.append(SOC_diff)
            x.append(ActEnergy)

y = np.array(y)
x = np.array(x).reshape((-1, 1))


model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('slope:', model.coef_)
print('intercept:', model.intercept_)