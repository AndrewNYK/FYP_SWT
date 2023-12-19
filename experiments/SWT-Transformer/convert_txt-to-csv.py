import pandas as pd
import numpy

dtype = {
    'Date':str,
    'Time':str,
    'Global_active_power':str,
    'Global_reactive_power':str,
    'Voltage;Global_intensity':str,
    'Sub_metering_1':str,
    'Sub_metering_2':str,
    'Sub_metering_3':numpy.float64
}

read_file = pd.read_csv('household_power_consumption.txt', sep=';', dtype=dtype)
print(read_file.iloc[0])
head_data = read_file.iloc[0]
for x in head_data:
    print(str(x) + ": " + str(type(x)))
read_file.to_csv ('household_power_consumption.csv', index=None)