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
ablation = df_cases.iloc[0:, 3]
ablation2 = df_cases.iloc[0:, 4]
ablation3 = df_cases.iloc[0:, 5]


# ******************* plot ************************
time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***
fig, axs = plt.subplots(1, figsize=(9, 6), dpi=100)
axs.plot(time_scale, prediction, "-", linewidth=1, color='C0', label="Quarantined (prediction)")  # ***
axs.plot(time_scale, ablation, "-", linewidth=1, color='C1', label="Quarantined (ablation)")  # ***
axs.plot(time_scale, ablation2, "-", linewidth=1, color='C2', label="Quarantined (ablation2)")  # ***
axs.plot(time_scale, ablation3, "-", linewidth=1, color='C3', label="Quarantined (ablation3)")  # ***
axs.scatter(time_scale[0:data_points_fitting], truth[0: data_points_fitting], marker="o", s=8,
            color='C4')  # ***
axs.scatter(time_scale[data_points_fitting:], truth[data_points_fitting:],
            marker="*", s=10, color='C4', label="quarantined (actual)")  # ***

axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
axs.set_xlabel("Days")
axs.set_ylabel("Cases")
axs.legend(loc='upper left')  # ***
plt.savefig("comparison_Austria.png", dpi=400, bbox_inches='tight')
plt.show()

plt.show()
