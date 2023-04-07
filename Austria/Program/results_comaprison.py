import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

df_cases = pd.read_csv("Austria_infected_prediction_total.csv")
starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
data_points = len(df_cases.iloc[0:, 0])
truth = df_cases.iloc[0:, 1]
prediction = df_cases.iloc[0:, 2]
ablation = df_cases.iloc[0:, 3]
ablation2 = df_cases.iloc[0:, 4]


# ******************* plot ************************
time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***
fig, axs = plt.subplots(1)
axs.plot(time_scale, prediction, "-", markersize=1, color='r', label="Quarantined (prediction)")  # ***
axs.plot(time_scale, ablation, "-", markersize=1, color='g', label="Quarantined (ablation)")  # ***
axs.plot(time_scale, ablation2, "-", linewidth=1, color='purple', label="Quarantined (ablation2)")  # ***
axs.plot(time_scale, truth, "o", markersize=2, color='b', label="Quarantined")  # ***
axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
axs.set_xlabel("Days")
axs.set_ylabel("Cases")
axs.legend(loc='upper left')  # ***

plt.show()
