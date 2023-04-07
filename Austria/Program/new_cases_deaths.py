import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
district = "Austria_wave"  # ***
df_cases = pd.read_csv("Austria_wave1_integrated.csv")  # ***

starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
NPIs_no_vac = df_cases.iloc[0:, 2]
NPIs_vac = df_cases.iloc[0:, 3]
NPIs_index = df_cases.iloc[0:, 4]
daily_cases = df_cases.iloc[0:, 9]

daily_deaths = df_cases.iloc[0:, 10]
daily_recovered = df_cases.iloc[0:, 11]
daily_cases_average = df_cases.iloc[0:, 12]
daily_deaths_average = df_cases.iloc[0:, 13]
daily_recovered_average = df_cases.iloc[0:, 14]


length_of_days = len(NPIs_index)
print(f"length of days: {length_of_days}")
print(starting_time)
print(ending_time)
#print(NPIs_index)
time_scale = [starting_time + timedelta(days=i) for i in range(length_of_days)]

fig, axs = plt.subplots(2)

# *********************************************************************
# *** visualize the relationship between daily deaths and daily cases ***
# *********************************************************************
ax0 = axs[0].twinx()

daily_deaths_tr = list(daily_deaths[8:])
daily_deaths_average_tr = list(daily_deaths_average[8:])
#print(type(daily_recovered_tr))
insert_zeros = np.zeros(8)
daily_deaths_average_tr.extend(insert_zeros)

#axs[0].plot(time_scale, daily_deaths_tr, "--", linewidth=2, color='g', label="Daily deaths")
axs[0].plot(time_scale, daily_deaths_average_tr, "o", markersize=2, color='g', label="Daily deaths transport (average)")
ax0.plot(time_scale, daily_cases_average, "o", markersize=2, color='b', label="Daily Cases (average)")

axs[0].set_xlim(datetime.date(starting_time), datetime.date(ending_time))

axs[0].set_ylabel("Daily deaths (average)", color="g")
ax0.set_ylabel("Daily Cases (average)", color="b")

axs[0].tick_params(axis="y", colors="g")
ax0.tick_params(axis="y", colors="b")

ax0.spines["right"].set_color("b")
ax0.spines["left"].set_color("g")


# *********************************************************************
# *** visualize the relationship between daily recovered and daily cases ***
# *********************************************************************
#ax1 = axs[1].twinx()
daily_recovered_tr = list(daily_recovered[14:])
daily_recovered_average_tr = list(daily_recovered_average[12:])
#print(type(daily_recovered_tr))
insert_zeros = np.zeros(12)
daily_recovered_average_tr.extend(insert_zeros)
#print(daily_recovered_tr)
axs[1].plot(time_scale, daily_recovered_average_tr, "o", markersize=2, color='r', label="Daily recovered transport (average)")
#axs[1].plot(time_scale, daily_cases, "o", markersize=2, color='b', label="Daily Cases")
axs[1].plot(time_scale, daily_cases_average, "o", markersize=2, color='b', label="Daily Cases (average)")
#ax1.plot(time_scale, daily_cases, "o", markersize=2, color='b', label="Daily Cases")

axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))
axs[1].set_ylabel("Daily cases")
#axs[1].set_ylabel("Daily recovered", color="g")
#ax1.set_ylabel("Daily Cases", color="b")

#axs[1].tick_params(axis="y", colors="g")
#ax1.tick_params(axis="y", colors="b")

#ax1.spines["right"].set_color("b")
#ax1.spines["left"].set_color("g")
axs[1].legend()
plt.show()
