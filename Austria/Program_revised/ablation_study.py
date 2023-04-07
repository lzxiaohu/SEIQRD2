import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import functions_revised as functions
import time
import random
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
import matplotlib.dates as dt


if __name__ == "__main__":
    # time counting
    start = time.time()
    # fix the seed
    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    # inputs
    # the information of COVID-19: time, data points of fitting, vaccination, NPIs, and virus variants, epidemiology
    district = "Austria_wave"  # ***
    df_cases = pd.read_csv("Austria_wave1_integrated_revised.csv")  # ***
    # the starting and ending time of a wave
    starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
    ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
    # the predictions for 2 weeks
    data_points = len(df_cases.iloc[0:, 0])
    prediction_length = 14  # a constant
    # the length of data points for fitting
    data_points_fitting = data_points - prediction_length
    # the ending time of fitting data
    ending_time_fitting = datetime.strptime(df_cases.iloc[-1 * (prediction_length + 1), 0], "%d/%m/%Y")
    # PIs: vaccination
    protection_rates_daily_fitting = df_cases.iloc[0:data_points_fitting, 20]
    protection_rates_daily_prediction = np.ones(prediction_length) * 0.2 / 100  # a constant
    protection_rates_daily = np.concatenate((protection_rates_daily_fitting, protection_rates_daily_prediction), axis=0)
    # NPIs: interventions
    NPIs_fitting = df_cases.iloc[0:data_points_fitting, 2]
    NPIs_prediction = np.zeros(prediction_length)  # assume that there is no NPI change in the prediction
    NPIs = np.concatenate((NPIs_fitting, NPIs_prediction), axis=0)
    # ------ variants of virus: Alpha
    variant = "Delta"
    # ------ Targets: cases
    daily_cases_average_fitting = df_cases.iloc[0:data_points_fitting, 12]
    daily_deaths_fitting = df_cases.iloc[0:data_points_fitting, 13]
    daily_recovered_fitting = df_cases.iloc[0:data_points_fitting, 14]
    quarantined_fitting_truth = df_cases.iloc[0:data_points_fitting, 16]
    quarantined_prediction_truth = df_cases.iloc[data_points_fitting:, 16]
    quarantined_truth = df_cases.iloc[0:, 16]
    # Parameters:
    # initial values of hyper parameters
    total_population = df_cases.iloc[0, 15]
    init_protected = df_cases.iloc[0, 19] / 100.0 * total_population
    protection_rates = df_cases.iloc[:, 19] / 100.0
    init_exposed = 0.005 * total_population  # a constant
    init_infected = 0.003 * total_population  # a constant
    init_quarantined = df_cases.iloc[0, 16]
    init_recovered = df_cases.iloc[0, 7]
    init_death = df_cases.iloc[0, 6]
    init_susceptible = total_population - init_protected - init_exposed - init_infected - init_quarantined - \
                       init_recovered - init_death
    # data for fitting:
    # epidemiological data as a part of x_data
    susceptible_fitting = np.ones(data_points_fitting) * init_susceptible
    protected_fitting = np.ones(data_points_fitting) * init_protected
    exposed_fitting = np.ones(data_points_fitting) * init_exposed
    infected_fitting = np.ones(data_points_fitting) * init_infected
    quarantined_fitting = np.ones(data_points_fitting) * init_quarantined
    recovered_fitting = np.ones(data_points_fitting) * init_recovered
    death_fitting = np.ones(data_points_fitting) * init_death
    lams_fitting = functions.recoveryRates(daily_recovered_fitting, quarantined_fitting_truth)
    phis_fitting = functions.DeathRates(daily_deaths_fitting, quarantined_fitting_truth)
    pop_fitting = np.ones(data_points_fitting) * total_population
    # vaccination for fitting as a part of x_data
    alphas_fitting = protection_rates_daily_fitting
    # virus variants for fitting as a part of x_data
    pi_fitting = variant
    # x_data_fitting:
    # susceptible, protected, exposed, infected, quarantined, recovered, death, alphas, lams, phis, NPIs, pop
    # x_data_fitting is considered as a hyperparameter
    x_data_fitting = np.stack((susceptible_fitting, protected_fitting, exposed_fitting, infected_fitting,
                               quarantined_fitting, recovered_fitting, death_fitting, alphas_fitting, lams_fitting,
                               phis_fitting, NPIs_fitting, pop_fitting), axis=0)
    print("initial susceptible_fitting:", x_data_fitting[0, 0])
    print("initial protected_fitting:", x_data_fitting[1, 0])
    print("initial exposed_fitting:", x_data_fitting[2, 0])
    print("initial infected_fitting:", x_data_fitting[3, 0])
    print("initial quarantined_fitting:", x_data_fitting[4, 0])
    print("initial recovered_fitting: ", x_data_fitting[5, 0])
    print("initial death_fitting: ", x_data_fitting[6, 0])
    print("daily protection rates daily: ", x_data_fitting[7, 0])
    print("daily recovery rates: ", x_data_fitting[8, 0])
    print("daily death rates: ", x_data_fitting[9, 0])
    print("NPIs list: ", x_data_fitting[10, 0])
    print("the total population: ", x_data_fitting[11, 0])

    # NPIs for fitting
    # initial values of parameters
    init_beta = 0.2  # a constant
    init_NPIs_eff = []
    init_NPIs_eff_upper_bounds = []
    init_NPIs_eff_lower_bounds = []
    init_gamma = 0.2  # a constant
    init_delta = 0.5  # a constant
    init_NPIs_delay_days = 2  # a constant
    # record the numbers of change points in transmission rate
    tt = [0]
    # setting the initial values for change points
    for ii in range(len(NPIs_fitting)):
        # '1' stands for control; '2' stands for relaxation; '0' stands for unchanged
        if NPIs_fitting[ii] == 2:
            init_NPIs_eff.append(1e-6)
            init_NPIs_eff_upper_bounds.append(3)
            init_NPIs_eff_lower_bounds.append(0)
            tt[0] = tt[0] + 1
        if NPIs_fitting[ii] == 1:
            init_NPIs_eff.append(-1e-6)
            init_NPIs_eff_upper_bounds.append(0)
            init_NPIs_eff_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
    # init includes: init_beta, init_gamma, init_delta, init_NPIs_delay_days, init_NPIs_eff.
    # The variable init considered as a parameter
    init_tempt = [init_beta, init_gamma, init_delta, init_NPIs_delay_days]
    init = np.concatenate((init_tempt, init_NPIs_eff), axis=0)
    # set boundaries of init
    upper_bounds_2 = np.array([3, 1, 1, 7])
    lower_bounds_2 = np.array([0, 0, 0, 0])
    upper_bounds = np.concatenate((upper_bounds_2, init_NPIs_eff_upper_bounds), axis=0)
    lower_bounds = np.concatenate((lower_bounds_2, init_NPIs_eff_lower_bounds), axis=0)
    bounds = (lower_bounds, upper_bounds)
    # print("init_tempt ", init_tempt)
    # print("init: ", init)
    # print("upper bounds: ", upper_bounds)
    # target of optimization
    y_data_fitting = quarantined_fitting_truth
    # print("quarantined average: ", y_data_fitting)
    para_est = functions.ablationFit(init, x_data_fitting, y_data_fitting, bounds)
    print("para_est: ", para_est)
    print("it took", time.time() - start, "seconds.")

    # *********** predictions ******************
    # similar process for optimization
    # epidemiological data as a part of x_data
    susceptible_prediction = np.ones(prediction_length) * init_susceptible
    protected_prediction = np.ones(prediction_length) * init_protected
    exposed_prediction = np.ones(prediction_length) * init_exposed
    infected_prediction = np.ones(prediction_length) * init_infected
    quarantined_prediction = np.ones(prediction_length) * init_quarantined
    recovered_prediction = np.ones(prediction_length) * init_recovered
    death_prediction = np.ones(prediction_length) * init_death
    lams_prediction = np.ones(prediction_length) * lams_fitting[data_points_fitting - 1]
    phis_prediction = np.ones(prediction_length) * phis_fitting[data_points_fitting - 1]
    pop_prediction = np.ones(prediction_length) * total_population
    # vaccination for prediction as a part of x_data
    alphas_prediction = protection_rates_daily_prediction
    # virus variants for prediction as a part of x_data
    pi_prediction = variant
    # x_data_prediction:
    # susceptible, protected, exposed, infected, quarantined, recovered, death, alphas, lams, phis, NPIs, pop
    # x_data_prediction is considered as a hyperparameter
    x_data_prediction = np.stack((susceptible_prediction, protected_prediction, exposed_prediction,
                                  infected_prediction, quarantined_prediction, recovered_prediction, death_prediction,
                                  alphas_prediction, lams_prediction, phis_prediction, NPIs_prediction, pop_prediction),
                                 axis=0)
    print("prediction of susceptible:", x_data_fitting[0, 0])
    print("prediction of protection:", x_data_fitting[1, 0])
    print("prediction of exposed:", x_data_fitting[2, 0])
    print("prediction of infected:", x_data_fitting[3, 0])
    print("prediction of quarantined:", x_data_fitting[4, 0])
    print("prediction of recovered:", x_data_fitting[5, 0])
    print("prediction of death: ", x_data_fitting[6, 0])
    print("daily protection rate for prediction: ", x_data_fitting[7, 0])
    print("daily recovery rates for prediction: ", x_data_fitting[8, 0])
    print("daily death rates for prediction: ", x_data_fitting[9, 0])
    print("NPIs list: ", x_data_fitting[10, 0])
    print("the total population: ", x_data_fitting[11, 0])
    # the merge of x_data_fitting and x_data_prediction
    x_data = np.concatenate((x_data_fitting, x_data_prediction), axis=1)
    print("x_data: ", np.shape(x_data))

    susceptible_fit, protected_fit, exposed_fit, infected_fit, quarantined_fit, recovered_fit, death_fit = \
        functions.SPEIQRDSimulation(x_data, *para_est)
    df_infected = pd.read_csv("Austria_infected.csv")
    df_infected['ablation'] = quarantined_fit
    df_infected.to_csv("Austria_infected_prediction_ablation.csv", index=False)

    # *********** Evaluation ******************
    mape = functions.MAPECompute(quarantined_fit[data_points_fitting:], quarantined_prediction_truth)
    print("mape: ", mape)

    # *********** Explanation & Analysis ******************
    # compute the transmission rates of the model
    # a constant ***
    optimal_NPIs_delay_days = para_est[3]
    optimal_init_beta = para_est[0]
    optimal_NPIs_eff = para_est[4: 8]
    betas = functions.transmissionRateSPEIQRD(optimal_NPIs_delay_days, NPIs, optimal_init_beta, optimal_NPIs_eff)
    trans = betas

    # ******************* plot & Display ************************
    time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***
    # ******************* prediction *******************
    fig, axs = plt.subplots(1, figsize=(9, 6), dpi=100)
    axs.plot(time_scale, quarantined_fit, "-", linewidth=1, color='g', label="quarantined (prediction)")  # ***
    axs.scatter(time_scale[0:data_points_fitting], quarantined_truth[0: data_points_fitting], marker="o", s=8,
                   color='r')  # ***
    axs.scatter(time_scale[data_points_fitting:], quarantined_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="quarantined (actual)")  # ***

    axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs.set_ylim(0, 3e5)
    axs.set_ylabel("Cases")
    axs.legend(loc='upper right', prop={"size": 12})  # ***
    plt.savefig("prediction_Austria_ablation1.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ************************ explanation ****************************
    # daily cases
    fig, axs = plt.subplots(4, figsize=(9, 6), dpi=100)
    axs[0].plot(time_scale, df_cases.iloc[0:, 9], "-", linewidth=1, color='g', label="daily reported cases")  # ***
    # booster
    axs[0].scatter(time_scale[16], 1e3, marker="P", color="black")
    axs[0].text(time_scale[17], 2e2, s="PI", color="black", font={'size': 8})
    # the first NPIs
    axs[0].scatter(time_scale[21], 1e3, marker="^", color='r', s=16)  # for the unvaccinated
    axs[0].text(time_scale[22], 1e3, s=r"$R_{1}^{0}$", color='r', font={'size': 8})  # for the unvaccinated
    axs[0].scatter(time_scale[21], 3e3, marker="^", color='black', s=16)  # for the vaccinated
    axs[0].text(time_scale[22], 3e3, s=r"$R_{1}^{v}$", color='black', font={'size': 8})  # for the vaccinated
    # the second NPIs
    axs[0].scatter(time_scale[46], 10e3, marker="v", color='r', s=16)  # for the unvaccinated
    axs[0].text(time_scale[47], 10e3, s=r"$C_{1}^{0}$", color='r', font={'size': 8})  # for the unvaccinated
    # axs[0].scatter(time_scale[46], 10e3, marker="_", color='black', s=16)  # for the vaccinated
    # axs[0].text(time_scale[47], 10e3, s="no changes", color='black', font={'size': 8})
    # the third NPIs
    axs[0].scatter(time_scale[65], 10e3, marker="v", color='r', s=16)  # for the unvaccinated
    axs[0].text(time_scale[66], 10e3, s=r"$C_{2}^{0}$", color='r', font={'size': 8})  # for the unvaccinated
    # axs[0].scatter(time_scale[65], 10e4, marker="_", color='black', s=16)  # for the vaccinated
    # axs[0].text(time_scale[66], 10e4, s="no changes", color='black', font={'size': 8})  # for the vaccinated
    axs[0].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs[0].set_ylim(0, 25e3)
    axs[0].set_ylabel("Cases")
    axs[0].legend(loc='upper right', prop={"size": 8})  # ***

    # vaccination info
    file_name = "Austria_effectiveness_vaccine.csv"  # ***
    df_vaccine = pd.read_csv(file_name)  # ***
    total_vac = df_vaccine.iloc[243:335, 1] / 100
    # print(total_vac)
    total_booster = df_vaccine.iloc[243:335, 2] / 100
    axs[1].plot(time_scale, protection_rates, "-",linewidth=1, color='b', label="protection rates")
    axs[1].plot(time_scale, total_vac, "-", linewidth=1, color='g', label="vacc rate")
    axs[1].plot(time_scale, total_booster, "--", linewidth=1, color='g', label="bster rate")
    axs[1].scatter(time_scale[16], 0.05, marker="P", color="black")
    axs[1].annotate('start mass boosters', xy=(time_scale[17], 0.05), xytext=(time_scale[30], 0.1),
                    arrowprops=dict(arrowstyle="->"))
    axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs[1].set_ylim(0, 0.8)
    axs[1].set_ylabel("Effectiveness \n or coverage")
    axs[1].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # the display of impact of NPIs
    print("betas_zero_one", betas)
    axs[2].plot(time_scale, betas, "-", linewidth=1, color='r', label=r"$\beta_{base,1}^{0}$")  # ***

    axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[2].set_xlabel("Days")
    axs[2].set_ylim(0, 3.5)
    axs[2].set_ylabel("Basic \n transmission \n rates")
    axs[2].legend(loc='upper left')  # ***
    axs[2].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # print("trans_vac_one", trans_vac_one)
    # print("trans_vac_two", trans_vac_two)
    axs[3].plot(time_scale, trans, "-", linewidth=1, color='r', label=r"$\beta_{1}^{0}$")  # ***
    axs[3].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***

    mondays = WeekdayLocator(MONDAY)
    alldays = DayLocator()
    autodates = dt.AutoDateLocator()
    fig.autofmt_xdate()
    axs[3].xaxis.set_major_locator(mondays)
    axs[3].xaxis.set_minor_locator(alldays)
    formatter = DateFormatter('%y/%m/%d')
    axs[3].xaxis.set_major_formatter(formatter)
    axs[3].tick_params(axis='x', which='major', labelsize=6)
    axs[3].set_xlabel("Days", fontsize=8)
    axs[3].set_ylim(0, 2.5)

    axs[3].set_ylabel("Transmission \n rates")
    # axs[3].legend(loc='upper left')  # ***
    axs[3].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})
    plt.savefig("transmissionrates_Austria_ablation1.png", dpi=400, bbox_inches='tight')
    plt.show()