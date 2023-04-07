import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df_cases = pd.read_csv("Israel_infected_prediction_total.csv")
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
axs.plot(time_scale, prediction, "-", linewidth=1, color='C0', label="quarantined")  # ***
axs.plot(time_scale, ablation1, "-", linewidth=1, color='C1', label="quarantined (ablation1)")  # ***
axs.plot(time_scale, ablation2, "-", linewidth=1, color='C2', label="quarantined (ablation2)")  # ***
axs.plot(time_scale, ablation3, "-", linewidth=1, color='C3', label="quarantined (ablation3)")  # ***
axs.axvline(x=time_scale[data_points_fitting-1], linewidth=2, color='r')
axs.scatter(time_scale, truth, marker="o", s=8, color='C4', label="actual quarantined")  # ***
axs.annotate(r"", xy=(time_scale[data_points_fitting - 1], 2.5e4), xytext=(time_scale[data_points_fitting - 10], 2.5e4),
             arrowprops=dict(arrowstyle="<-", color='r'), font={'size': 8}, color='r')
axs.text(time_scale[data_points_fitting - 7], 2.6e4, s=r"Fitting", color='r', font={'size': 8})
axs.annotate(r"", xy=(time_scale[data_points_fitting - 1], 5e4), xytext=(time_scale[data_points_fitting + 10], 5e4),
             arrowprops=dict(arrowstyle="<-", color='r'), font={'size': 8}, color='r')
axs.text(time_scale[data_points_fitting + 1], 5.1e4, s=r"Prediction", color='r', font={'size': 8})

axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
axs.set_xlabel("Days")
axs.set_ylim(0, 16e4)
axs.set_ylabel("Cases")
axs.legend(loc='upper left', prop={"size": 8})  # ***
plt.savefig("comparison_quarantined_Israel.png", dpi=400, bbox_inches='tight')
plt.show()
