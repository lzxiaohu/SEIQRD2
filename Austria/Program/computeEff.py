import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import functions_revised as functions


file_path = "Austria_info_vaccination.csv"
file_name = functions.computeEff(file_path)
df_cases = pd.read_csv(file_name)  # ***

starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
total_vac = df_cases.iloc[:, 1]
total_booster = df_cases.iloc[:, 2]
daily_vac = df_cases.iloc[:, 3]
daily_booster = df_cases.iloc[:, 4]
eff = df_cases.iloc[:, 5]
time_scale = [starting_time + timedelta(days=i) for i in range(len(eff))]
starting_plot = starting_time + timedelta(days=2)

fig, axs = plt.subplots(3)

axs[0].plot(time_scale[2:], eff[2:], "-", linewidth=2, color='g', label="Effectiveness")
axs[0].set_xlim(starting_plot, datetime.date(ending_time))  # ***
axs[0].set_xlabel("Days")
axs[0].set_ylabel("Effectiveness")
axs[0].legend(loc='upper right')  # ***

ax1 = axs[1].twinx()
lns1 = axs[1].plot(time_scale[2:], daily_vac[2:]/100, "-", linewidth=2, color='g', label="Daily vaccination rate")
lns2 = ax1.plot(time_scale[2:], total_vac[2:]/100, "-", linewidth=2, color='b', label="Total vaccination coverage")
lns = lns1 + lns2
labels = [ln.get_label() for ln in lns]
axs[1].legend(lns, labels, loc='upper left')
# set x axis
axs[1].set_xlim(starting_plot, datetime.date(ending_time))  # ***
axs[1].set_xlabel("Days")
# set y axis
axs[1].set_ylabel("Vaccination rate", color="g")
ax1.set_ylabel("total proportion", color="b")
axs[1].tick_params(axis="y", colors="g")
ax1.tick_params(axis="y", colors="b")
ax1.spines["right"].set_color("b")
ax1.spines["left"].set_color("g")

del lns1, lns2, lns, labels

#axs[1].set_ylabel("vaccination rate")
# ***
#ax1.legend(loc='upper right')

ax2 = axs[2].twinx()
lns1 = axs[2].plot(time_scale[2:], daily_booster[2:]/100, "-", linewidth=2, color='g', label="Daily booster rate")
lns2 = ax2.plot(time_scale[2:], total_booster[2: ]/100, "-", linewidth=2, color='b', label="Total booster coverage")
lns = lns1 + lns2
labels = [ln.get_label() for ln in lns]
axs[2].legend(lns, labels, loc="upper left")

axs[2].set_xlim(starting_plot, datetime.date(ending_time))  # ***
axs[2].set_xlabel("Days")
# set y axis

axs[2].set_ylabel("booster rate", color="g")
ax2.set_ylabel("total proportion", color="b")
axs[2].tick_params(axis="y", colors="g")
ax2.tick_params(axis="y", colors="b")
ax2.spines["right"].set_color("b")
ax2.spines["left"].set_color("g")

del lns1, lns2, lns, labels

plt.show()