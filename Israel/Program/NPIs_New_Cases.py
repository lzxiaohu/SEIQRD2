import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta
district = "Israel_wave"
df_cases = pd.read_csv("Israel_wave1_integrated.csv")

starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
NPIs_no_vac = df_cases.iloc[0:, 2]
NPIs_vac = df_cases.iloc[0:, 3]
NPIs_index = df_cases.iloc[0:, 4]
daily_cases = df_cases.iloc[0:, 9]
length_of_days = len(NPIs_index)
print(f"length of days: {length_of_days}")
print(starting_time)
print(ending_time)
#print(NPIs_index)
time_scale = [starting_time + timedelta(days=i) for i in range(length_of_days)]
#print(time_scale)
fig, axs = plt.subplots(3)


# *********************************************************************
# *** visualize the relationship between NPIs index and daily cases ***
# *********************************************************************
ax0 = axs[0].twinx()
axs[0].plot(time_scale, NPIs_index, "--", linewidth=2, color='g', label="NPIs Index")
ax0.plot(time_scale, daily_cases, "o", markersize=2, color='b', label="Daily Cases")

axs[0].set_xlim(datetime.date(starting_time), datetime.date(ending_time))

axs[0].set_ylabel("NPIs Index", color="g")
ax0.set_ylabel("Daily Cases", color="b")

axs[0].tick_params(axis="y", colors="g")
ax0.tick_params(axis="y", colors="b")

ax0.spines["right"].set_color("b")
ax0.spines["left"].set_color("g")

axs[0].legend()
ax0.legend()
# *********************************************************************
# *** visualize the relationship between NPIs index and NPIs events ***
# *********************************************************************
axs[1].plot(time_scale, NPIs_index, "--", linewidth=2, color='g', label="NPIs Index")
NPIs_no_vac_relaxation = []
NPIs_no_vac_control = []
NPIs_vac_control = []
NPIs_vac_relaxation = []
for ii in range(length_of_days):
    if NPIs_no_vac[ii] == 1:
        NPIs_no_vac_control.append(starting_time + timedelta(days=ii))
    if NPIs_no_vac[ii] == 2:
        NPIs_no_vac_relaxation.append(starting_time + timedelta(days=ii))
    if NPIs_vac[ii] == 1:  # +1 is used to display the changes of NPIs, not real
        NPIs_vac_control.append(starting_time + timedelta(days=ii+1))
    if NPIs_vac[ii] == 2:  # +1 is used to display the changes of NPIs, not real
        NPIs_vac_relaxation.append(starting_time + timedelta(days=ii+1))

#print(NPIs_no_vac_relaxation)
#print(NPIs_no_vac_control)
if len(NPIs_no_vac_control) > 0:
    axs[1].axvline(x=NPIs_no_vac_control[0], color="b", linestyle="-", label="Control for unvaccinated")
if len(NPIs_no_vac_control) > 1:
    for ii in range(1, len(NPIs_no_vac_control)):
        axs[1].axvline(x=NPIs_no_vac_control[ii], color="b", linestyle="-")

if len(NPIs_no_vac_relaxation) > 0:
    axs[1].axvline(x=NPIs_no_vac_relaxation[0], color="b", linestyle="--", label="Relaxation for unvaccinated")
if len(NPIs_no_vac_relaxation) > 1:
    for ii in range(1, len(NPIs_no_vac_relaxation)):
        axs[1].axvline(x=NPIs_no_vac_relaxation[ii], color="b", linestyle="--")

if len(NPIs_vac_control) > 0:
    axs[1].axvline(x=NPIs_vac_control[0], color="r", linestyle="-", label="Control for vaccinated")
if len(NPIs_vac_control) > 1:
    for ii in range(1, len(NPIs_vac_control)):
        axs[1].axvline(x=NPIs_vac_control[ii], color="r", linestyle="-")

if len(NPIs_vac_relaxation) > 0:
    axs[1].axvline(x=NPIs_vac_relaxation[0], color="r", linestyle="--", label="Relaxation for vaccinated")
if len(NPIs_vac_relaxation) > 1:
    for ii in range(1, len(NPIs_vac_relaxation)):
        axs[1].axvline(x=NPIs_vac_relaxation[ii], color="r", linestyle="--")
axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))
axs[1].set_ylabel("NPIs Index")
axs[1].legend()

# **********************************************************************
# *** visualize the relationship between daily cases and NPIs events ***
# **********************************************************************
axs[2].plot(time_scale, daily_cases, "--", linewidth=2, color='g', label="Daily cases")

if len(NPIs_no_vac_control) > 0:
    axs[2].axvline(x=NPIs_no_vac_control[0], color="b", linestyle="-", label="Control for unvaccinated")
if len(NPIs_no_vac_control) > 1:
    for ii in range(1, len(NPIs_no_vac_control)):
        axs[2].axvline(x=NPIs_no_vac_control[ii], color="b", linestyle="-")

if len(NPIs_no_vac_relaxation) > 0:
    axs[2].axvline(x=NPIs_no_vac_relaxation[0], color="b", linestyle="--", label="Relaxation for unvaccinated")
if len(NPIs_no_vac_relaxation) > 1:
    for ii in range(1, len(NPIs_no_vac_relaxation)):
        axs[2].axvline(x=NPIs_no_vac_relaxation[ii], color="b", linestyle="--")

if len(NPIs_vac_control) > 0:
    axs[2].axvline(x=NPIs_vac_control[0], color="r", linestyle="-", label="Control for vaccinated")
if len(NPIs_vac_control) > 1:
    for ii in range(1, len(NPIs_vac_control)):
        axs[2].axvline(x=NPIs_vac_control[ii], color="r", linestyle="-")

if len(NPIs_vac_relaxation) > 0:
    axs[2].axvline(x=NPIs_vac_relaxation[0], color="r", linestyle="--", label="Relaxation for vaccinated")
if len(NPIs_vac_relaxation) > 1:
    for ii in range(1, len(NPIs_vac_relaxation)):
        axs[2].axvline(x=NPIs_vac_relaxation[ii], color="r", linestyle="--")
axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))
axs[2].set_ylabel("Daily cases")
axs[2].legend()

#plt.plot(time_full, exposed_est, "--", linewidth=2, label="expo_est")
#plt.plot(time_full, infected_est, "--", linewidth=2, label="infe_est")
#axs[0].plot(time_full[0:-10], quarantine, "--", linewidth=2, label="train")
#axs[0].plot(time_full[-10:], quarantine_predict, "r--", linewidth=2, label="predict")
#axs[1].plot(time_full[0:-10], infected_accum, "--", linewidth=2, label="train")
#axs[1].plot(time_full[-10:], infected_accum_predict, "r--", linewidth=2, label="predict")
#axs[2].plot(time_full[0:-10], death_accum, "--", linewidth=2, label="train")
#axs[2].plot(time_full[-10:], death_accum_predict, "r--", linewidth=2, label="predict")
#axs[3].plot(time_full[0:-10], recovered_accum, "--", linewidth=2, label="train")
#axs[3].plot(time_full[-10:], recovered_accum_predict, "r--", linewidth=2, label="predict")
#axs[0].plot(time_full, quarantine_prediction, "--", linewidth=2, label="quarantine_obs")
#axs[2].plot(time_full, death_est, "--", linewidth=2, label="deat_est")
#axs[2].plot(time_full, death_accum_prediction, "--", linewidth=2, label="death_obs")
#plt.plot(time_full, protected_est, "--", linewidth=2, label="prot_est")
#plt.xlabel("Timesteps")

#axs[0].set_title('Quaratine')
#axs[1].set_title('Infected')
#axs[2].set_title('Death')
#axs[3].set_title('Recovered')

#axs[0].legend(loc='best')
#axs[1].legend(loc='best')
#axs[2].legend(loc='best')
#axs[3].legend(loc='best')
#fig.savefig('output_'+district+'.png', dpi=1200)


plt.show()
