import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df_cases = pd.read_csv("Austria_infected_prediction_total.csv")
starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
data_points = len(df_cases.iloc[0:, 0])
data_points_fitting = data_points - 14
truth = df_cases.iloc[0:, 1]
prediction = df_cases.iloc[0:, 2]
ablation1 = df_cases.iloc[0:, 3]
ablation2 = df_cases.iloc[0:, 4]
ablation3 = df_cases.iloc[0:, 5]


# ******************* plot ************************
time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***
fig, axs = plt.subplots(1, figsize=(9, 6), dpi=100)
axs.plot(time_scale, prediction, "-", linewidth=1, color='C0', label="quarantined (SEIQRD$^2$)")  # ***
axs.plot(time_scale, ablation1, "-", linewidth=1, color='C1', label="quarantined (SPEIQRD)")  # ***
axs.plot(time_scale, ablation2, "-", linewidth=1, color='C2', label="quarantined (SVEIQRD)")  # ***
axs.plot(time_scale, ablation3, "-", linewidth=1, color='C3', label="quarantined (SEIQRD$^2_c$)")  # ***
axs.axvline(x=time_scale[data_points_fitting-1], linewidth=2, color='r')
axs.scatter(time_scale, truth, marker="o", s=8, color='C4', label="actual quarantined")  # ***
axs.annotate(r"", xy=(time_scale[data_points_fitting - 1], 3e4), xytext=(time_scale[data_points_fitting - 10], 3e4),
             arrowprops=dict(arrowstyle="<-", color='r'), font={'size': 8}, color='r')
axs.text(time_scale[data_points_fitting - 7], 3.1e4, s=r"Fitting", color='r', font={'size': 8})
axs.annotate(r"", xy=(time_scale[data_points_fitting - 1], 6e4), xytext=(time_scale[data_points_fitting + 10], 6e4),
             arrowprops=dict(arrowstyle="<-", color='r'), font={'size': 8}, color='r')
axs.text(time_scale[data_points_fitting + 1], 6.1e4, s=r"Prediction", color='r', font={'size': 8})

axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
axs.set_xlabel("Days")
axs.set_ylim(0, 2.1e5)
axs.set_ylabel("Cases")
axs.legend(loc='upper left', prop={"size": 8})  # ***
plt.savefig("comparison_quarantined_Austria.png", dpi=400, bbox_inches='tight')
plt.show()
