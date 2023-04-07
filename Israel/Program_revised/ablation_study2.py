import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import sys
import functions_revised as functions
import time
import random
import matplotlib.pyplot as plt
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
    district = "Israel_wave"  # ***
    df_cases = pd.read_csv("Israel_wave1_integrated_revised.csv")  # ***
    # the starting and ending time of a wave
    starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
    ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
    # the predictions for 2 weeks
    prediction_length = 14
    # the length of data points for fitting
    data_points = len(df_cases.iloc[0:, 0])
    data_points_fitting = data_points - prediction_length
    # the ending time of fitting data
    ending_time_fitting = datetime.strptime(df_cases.iloc[-1 * (prediction_length + 1), 0], "%d/%m/%Y")
    # PIs: vaccination
    vaccination_rate_fitting = df_cases.iloc[0:data_points_fitting, 1] / 100.0  # remember 100
    vaccination_rate_prediction = np.ones(prediction_length) * 0.002 / 100.0  # a constant of 0.15
    vaccine_eff_fitting = df_cases.iloc[0:data_points_fitting, 18]  # the effectiveness of vaccine
    vaccine_eff_prediction = np.ones(prediction_length) * df_cases.iloc[data_points_fitting - 1, 18]
    vaccine_eff = np.concatenate((vaccine_eff_fitting, vaccine_eff_prediction), axis=0)
    # NPIs: interventions
    NPIs_no_vac_fitting = df_cases.iloc[0:data_points_fitting, 2]
    NPIs_no_vac_prediction = np.zeros(prediction_length)  # assume that there is no NPI change in the prediction
    NPIs_no_vac = np.concatenate((NPIs_no_vac_fitting, NPIs_no_vac_prediction), axis=0)
    NPIs_vac_fitting = df_cases.iloc[0:data_points_fitting, 3]
    NPIs_vac_prediction = np.zeros(prediction_length)  # assume that there is no NPI change in the vaccinated
    NPIs_vac = np.concatenate((NPIs_vac_fitting, NPIs_vac_prediction), axis=0)
    # variants of virus: Alpha
    variant = "Delta"
    # Targets: cases
    daily_cases_fitting = df_cases.iloc[0:data_points_fitting, 12]
    daily_deaths_fitting = df_cases.iloc[0:data_points_fitting, 13]
    daily_recovered_fitting = df_cases.iloc[0:data_points_fitting, 14]
    quarantined_fitting_truth = df_cases.iloc[0:data_points_fitting, 16]
    quarantined_prediction_truth = df_cases.iloc[data_points_fitting:, 16]
    quarantined_truth = df_cases.iloc[0:, 16]
    # Parameters:
    # initial values of hyper parameters
    total_population = df_cases.iloc[0, 15]
    init_vaccinated = df_cases.iloc[0, 17] / 100.0 * total_population  # remember 100
    init_exposed = 0.0001 * total_population  # ***
    init_infected = 0.0001 * total_population  # ***
    init_quarantined = df_cases.iloc[0, 16]
    init_recovered = df_cases.iloc[0, 7]
    init_death = df_cases.iloc[0, 6]
    init_susceptible = total_population - init_vaccinated - init_exposed - init_infected - init_quarantined - \
                       init_recovered - init_death
    # data for fitting:
    # epidemiological data
    susceptible_fitting = np.ones(data_points_fitting) * init_susceptible
    vaccinated_fitting = np.ones(data_points_fitting) * init_vaccinated
    exposed_fitting = np.ones(data_points_fitting) * init_exposed
    infected_fitting = np.ones(data_points_fitting) * init_infected
    quarantined_fitting = np.ones(data_points_fitting) * init_quarantined
    recovered_fitting = np.ones(data_points_fitting) * init_recovered
    death_fitting = np.ones(data_points_fitting) * init_death
    lams_fitting = functions.recoveryRates(daily_recovered_fitting, quarantined_fitting_truth)
    phis_fitting = functions.DeathRates(daily_deaths_fitting, quarantined_fitting_truth)
    pop_fitting = np.ones(data_points_fitting) * total_population
    # vaccination for fitting
    alphas_fitting = vaccination_rate_fitting
    # virus variants for fitting
    pi_fitting = variant
    # x_data_fitting: susceptible, vaccinated, exposed, infected, quarantined, recovered, death,
    # alphas, lams, phis, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    # x_data_fitting is considered as a hyper parameter
    x_data_fitting = np.stack((susceptible_fitting, vaccinated_fitting, exposed_fitting, infected_fitting,
                               quarantined_fitting, recovered_fitting, death_fitting, alphas_fitting, lams_fitting,
                               phis_fitting, NPIs_no_vac_fitting, NPIs_vac_fitting, vaccine_eff_fitting, pop_fitting),
                              axis=0)
    print("initial susceptible_fitting:", x_data_fitting[0, 0])
    print("initial vaccinated_fitting:", x_data_fitting[1, 0])
    print("initial exposed_fitting:", x_data_fitting[2, 0])
    print("initial infected_fitting:", x_data_fitting[3, 0])
    print("initial quarantined_fitting:", x_data_fitting[4, 0])
    print("initial recovered_fitting: ", x_data_fitting[5, 0])
    print("initial death_fitting: ", x_data_fitting[6, 0])
    print("daily vaccination rates: ", x_data_fitting[7, 0])
    print("daily recovery rates: ", x_data_fitting[8, 0])
    print("daily death rates: ", x_data_fitting[9, 0])
    print("NPIs_no_vac list: ", x_data_fitting[10, 0])
    print("NPIs_vac list: ", x_data_fitting[11, 0])
    print("vaccine effectiveness: ", x_data_fitting[12, 0])
    print("the total population: ", x_data_fitting[13, 0])
    # NPIs for fitting
    # initial values of parameters
    init_beta_zero = 0.1
    init_NPIs_eff_zero = []
    init_NPIs_eff_zero_upper_bounds = []
    init_NPIs_eff_zero_lower_bounds = []
    init_beta_vac = 0.2
    init_NPIs_eff_vac = []
    init_NPIs_eff_vac_upper_bounds = []
    init_NPIs_eff_vac_lower_bounds = []
    init_gamma = 0.2
    init_delta = 0.5
    init_NPIs_delay_days = 2
    # record the numbers of change points in four transmission rates
    tt = [0, 0]
    # setting the initial values for change points
    for ii in range(len(NPIs_no_vac_fitting)):
        # '1' stands for control; '2' stands for relaxation; '0' stands for unchanged
        if NPIs_no_vac_fitting[ii] == 2:
            init_NPIs_eff_zero.append(1e-6)
            init_NPIs_eff_zero_upper_bounds.append(3)
            init_NPIs_eff_zero_lower_bounds.append(0)
            tt[0] = tt[0] + 1
        if NPIs_no_vac_fitting[ii] == 1:
            init_NPIs_eff_zero.append(-1e-6)
            init_NPIs_eff_zero_upper_bounds.append(0)
            init_NPIs_eff_zero_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
        if NPIs_vac_fitting[ii] == 2:
            init_NPIs_eff_vac.append(1e-6)
            init_NPIs_eff_vac_upper_bounds.append(3)
            init_NPIs_eff_vac_lower_bounds.append(0)
            tt[1] = tt[1] + 1
        if NPIs_vac_fitting[ii] == 1:
            init_NPIs_eff_vac.append(-1e-6)
            init_NPIs_eff_vac_upper_bounds.append(0)
            init_NPIs_eff_vac_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
    # init includes: init_beta_zero, init_beta_vac, init_gamma,
    # init_delta, init_NPIs_delay_days, init_NPIs_eff_zero, init_NPIs_eff_vac.
    # The variable init considered as a parameter
    init_tempt = [init_beta_zero, init_beta_vac, init_gamma, init_delta, init_NPIs_delay_days]
    init = np.concatenate((init_tempt, init_NPIs_eff_zero, init_NPIs_eff_vac), axis=0)
    # set boundaries of init
    upper_bounds_2 = np.array([3, 3, 1, 1, 7])
    lower_bounds_2 = np.array([0, 0, 0, 0, 0])
    upper_bounds_3 = np.concatenate((init_NPIs_eff_zero_upper_bounds, init_NPIs_eff_vac_upper_bounds), axis=0)
    lower_bounds_3 = np.concatenate((init_NPIs_eff_zero_lower_bounds, init_NPIs_eff_vac_lower_bounds), axis=0)
    upper_bounds = np.concatenate((upper_bounds_2, upper_bounds_3), axis=0)
    lower_bounds = np.concatenate((lower_bounds_2, lower_bounds_3), axis=0)
    bounds = (lower_bounds, upper_bounds)
    # print("init_tempt ", init_tempt)
    # print("init: ", init)
    # print("upper bounds: ", upper_bounds)
    # target of optimization
    y_data_fitting = quarantined_fitting_truth
    # print("quarantined average: ", y_data_fitting)
    para_est = functions.ablationFit2(init, x_data_fitting, y_data_fitting, bounds)
    print("para_est: ", para_est)
    print("it took", time.time() - start, "seconds.")

    # para_est = [3.73888244e-01,  3.00000000e+00,  3.00000000e+00,  3.00000000e+00,
    #             1.00000000e+00,  6.11591122e-01,  2.10775041e+00, -4.23851778e-24,
    #             -3.72863707e-22, -4.96985379e-16, -5.71679596e-10, -1.41679445e-16,
    #             -1.07238727e-01,  1.03372335e-34,  4.02350774e-31, -1.08049422e-20,
    #             -5.56452842e-19, -1.32950204e-10, -7.12923747e-01, -2.02557656e-13,
    #             -6.29723521e-01,  1.33947741e-35,  3.09999544e-26, -3.26334333e-19,
    #             -2.36011084e-17, -5.27234327e-01, -2.10330812e-14, -9.72127939e-01,
    #             -1.64482916e-13,  2.04248310e-34,  2.60841071e-24, -4.46803475e-06,
    #             -2.51828178e-05, -2.99999588e+00, -2.09635830e-05,  2.34584091e-36,
    #             3.73727141e-23]

    # *********** predictions ******************
    # similar process for optimization
    # epidemiological data
    susceptible_prediction = np.ones(prediction_length) * init_susceptible
    vaccinated_prediction = np.ones(prediction_length) * init_vaccinated
    exposed_prediction = np.ones(prediction_length) * init_exposed
    infected_prediction = np.ones(prediction_length) * init_infected
    quarantined_prediction = np.ones(prediction_length) * init_quarantined
    recovered_prediction = np.ones(prediction_length) * init_recovered
    death_prediction = np.ones(prediction_length) * init_death
    lams_prediction = np.ones(prediction_length) * lams_fitting[data_points_fitting - 1]
    phis_prediction = np.ones(prediction_length) * phis_fitting[data_points_fitting - 1]
    pop_prediction = np.ones(prediction_length) * total_population
    # vaccination for prediction
    alphas_prediction = vaccination_rate_prediction
    # virus variants for prediction
    pi_prediction = variant
    # x_data_prediction: susceptible, vaccination, exposed, infected, quarantined, recovered, death, alphas, lams, phis,
    # NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    # x_data_prediction is considered as a hyper parameter
    x_data_prediction = np.stack((susceptible_prediction, vaccinated_prediction, exposed_prediction,
                                  infected_prediction, quarantined_prediction, recovered_prediction,
                                  death_prediction, alphas_prediction, lams_prediction, phis_prediction,
                                  NPIs_no_vac_prediction, NPIs_vac_prediction, vaccine_eff_prediction, pop_prediction),
                                 axis=0)
    print("prediction of susceptible:", x_data_prediction[0, 0])
    print("prediction of vaccinated:", x_data_prediction[1, 0])
    print("prediction of exposed:", x_data_prediction[2, 0])
    print("prediction of infected:", x_data_prediction[3, 0])
    print("prediction of quarantined:", x_data_prediction[4, 0])
    print("prediction of recovered: ", x_data_prediction[5, 0])
    print("prediction of death: ", x_data_prediction[6, 0])
    print("vaccination rates prediction: ", x_data_prediction[7, 0])
    print("daily recovery rates prediction: ", x_data_prediction[8, 0])
    print("daily death rates prediction: ", x_data_prediction[9, 0])
    print("NPIs_no_vac list for prediction: ", x_data_prediction[10, 0])
    print("NPIs_vac list for prediction: ", x_data_prediction[11, 0])
    print("vaccine effectiveness prediction: ", x_data_prediction[12, 0])
    print("the total population prediction: ", x_data_prediction[13, 0])
    # the merge of x_data_fitting and x_data_prediction
    x_data = np.concatenate((x_data_fitting, x_data_prediction), axis=1)
    print("x_data: ", np.shape(x_data))

    susceptible_fit, vaccinated_fit, exposed_fit, infected_fit, quarantined_fit, recovered_fit, death_fit = \
        functions.SVEIQRDSimulation(x_data, *para_est)
    # print(quarantined_vac_fit + quarantined_zero_fit)
    df_infected = pd.read_csv("Israel_infected.csv")  # ***
    df_infected['ablation2'] = quarantined_fit
    df_infected.to_csv("Israel_infected_prediction_ablation2.csv", index=False)  # ***

    mape = functions.MAPECompute(quarantined_fit[data_points_fitting:], quarantined_prediction_truth)
    print("mape: ", mape)

    # *********** Explanation & Analysis ******************
    # compute the transmission rates of the model ***
    betas_zero, betas_vac = functions.transmissionRateSVEIQRD(para_est[4], NPIs_no_vac, NPIs_vac, para_est[0], para_est[1],
                                                      para_est[5: 8], para_est[8:11])
    trans_zero = betas_zero
    trans_vac = []
    for ii in range(data_points):
        trans_vac.append(betas_vac[ii]*(1-vaccine_eff[ii]))

    # ******************* plot & Display ************************
    time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***

    # the display of predictions
    fig, axs = plt.subplots(1, figsize=(9, 6), dpi=100)
    axs.plot(time_scale, quarantined_fit, "-", linewidth=1, color='g', label="quarantined (prediction)")  # ***
    axs.scatter(time_scale[0:data_points_fitting], quarantined_truth[0: data_points_fitting], marker="o", s=2,
                color='r')  # ***
    axs.scatter(time_scale[data_points_fitting:], quarantined_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="quarantined (actual)")  # ***

    axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs.set_ylim(0, 15e4)
    axs.set_ylabel("Cases")
    axs.legend(loc='upper right', prop={"size": 12})  # ***
    plt.savefig("prediction_Israel_ablation2.png", dpi=400, bbox_inches='tight')
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
    axs[0].set_ylim(0, 3e4)
    axs[0].set_ylabel("Cases")
    axs[0].legend(loc='upper right', prop={"size": 8})  # ***


    # vaccination info
    file_name = "Israel_effectiveness_vaccine.csv"  # ***
    df_vaccine = pd.read_csv(file_name)  # ***
    total_vac = df_vaccine.iloc[202:294, 1] / 100
    # print(total_vac)
    total_booster = df_vaccine.iloc[202:294, 2] / 100
    axs[1].plot(time_scale, vaccine_eff, "-", linewidth=1, color='b', label="vacc eff")
    axs[1].plot(time_scale, total_vac, "-", linewidth=1, color='g', label="vacc rate")
    axs[1].plot(time_scale, total_booster, "--", linewidth=1, color='g', label="bster rate")
    axs[1].scatter(time_scale[16], 0.05, marker="P", color="black")
    axs[1].annotate('start mass boosters', xy=(time_scale[17], 0.05), xytext=(time_scale[30], 0.1),
                    arrowprops=dict(arrowstyle="->"))
    axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs[1].set_ylim(0, 0.8)
    axs[1].set_ylabel("Effectiveness \n or coverage")
    axs[1].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # basic transmission rates
    print("betas_zero_one", betas_zero)
    print("betas_vac_one", betas_vac)
    axs[2].plot(time_scale, betas_zero, "-", linewidth=1, color='r', label=r"$\beta_{base,1}^{0}$")  # ***
    axs[2].plot(time_scale, betas_vac, "-", linewidth=1, color='purple', label=r"$\beta_{base,2}^{v}$")  # ***
    axs[2].scatter(time_scale[22], betas_vac[21]-0.05, marker="o", c='w', edgecolor="purple", s=12)

    axs[2].scatter(time_scale[20], 0.2, marker="^", color='black', s=16)  # for the vaccinated
    axs[2].annotate(r"$R_{1}^{v}$", xy=(time_scale[22], betas_vac[22]-0.1), xytext=(time_scale[21], 0.1),
                    arrowprops=dict(arrowstyle="->"), font={'size': 8})

    axs[2].scatter(time_scale[45], 2.7, marker="v", color='r', s=16)  # for the unvaccinated
    axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[2].set_xlabel("Days")
    axs[2].set_ylim(0, 2.5)
    axs[2].set_ylabel("Basic \n transmission \n rates")
    axs[2].legend(loc='upper left')  # ***
    axs[2].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # transmission rates
    axs[3].plot(time_scale, trans_zero, "-", linewidth=1, color='r', label=r"$\beta_{1}^{0}$")  # ***
    axs[3].plot(time_scale, trans_vac, "-", linewidth=1, color='purple', label=r"$\beta_{2}^{v}$")  # ***
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
    axs[3].set_ylim(0, 2.0)

    axs[3].set_ylabel("Transmission \n rates")
    # axs[3].legend(loc='upper left')  # ***
    axs[3].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})
    plt.savefig("transmissionrates_Israel_ablation2.png", dpi=400, bbox_inches='tight')
    plt.show()