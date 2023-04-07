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
    vaccination_rates_fitting = df_cases.iloc[0:data_points_fitting, 1] / 100.0  # remember 100
    vaccination_rates_prediction = np.ones(prediction_length) * 0.02 / 100.0  # a constant of 0.15 ***
    # vaccine_eff_fitting = df_cases.iloc[0:data_points_fitting, 18]  # the effectiveness of vaccine ***
    # keep the effectiveness of vaccination a constant 0.775 ***
    vaccine_eff_fitting = np.ones(data_points_fitting) * 0.775
    # vaccine_eff_prediction = np.ones(prediction_length) * df_cases.iloc[data_points_fitting - 1, 18]
    vaccine_eff_prediction = np.ones(prediction_length) * 0.775
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
    total_deaths_truth = df_cases.iloc[0:, 6]
    daily_recovered_fitting = df_cases.iloc[0:data_points_fitting, 14]
    quarantined_fitting_truth = df_cases.iloc[0:data_points_fitting, 16]
    quarantined_prediction_truth = df_cases.iloc[-1 * prediction_length:, 16]
    quarantined_truth = df_cases.iloc[0:, 16]
    # Parameters:
    # initial values of hyper parameters
    total_population = df_cases.iloc[0, 15]
    init_vaccinated = df_cases.iloc[0, 17] / 100.0 * total_population  # remember 100
    init_exposed_zero = 400  # *** 1579 1.7e-4 * total_population
    init_exposed_vac = 1200   # *** 929 1e-4 * total_population
    init_infected_zero = 55  # *** 929 1e-4 * total_population 550
    init_infected_vac = 180   # *** 929 1e-4 * total_population 1800
    init_quarantined_zero = 7000  # *** total quarantined about 2.2: 10220 1.1e-3 * total_population
    init_quarantined_vac = 14000  # *** 11018 df_cases.iloc[0, 16] - init_quarantined_zero
    init_recovered_zero = 0.8e-3 * total_population  # *** total recovered about 0.8e-3
    init_recovered_vac = df_cases.iloc[0, 7] - init_recovered_zero
    init_death_zero = 4e-6 * total_population  # *** total death about 4e-6
    init_death_vac = df_cases.iloc[0, 6] - init_death_zero
    init_susceptible = total_population - init_vaccinated - init_exposed_zero - init_exposed_vac - \
                       init_infected_zero - init_infected_vac - init_quarantined_zero - init_quarantined_vac - \
                       init_recovered_zero - init_recovered_vac - init_death_zero - init_death_vac
    # data for fitting:
    # epidemiological data
    susceptible_fitting = np.ones(data_points_fitting) * init_susceptible
    vaccinated_fitting = np.ones(data_points_fitting) * init_vaccinated
    exposed_zero_fitting = np.ones(data_points_fitting) * init_exposed_zero
    exposed_vac_fitting = np.ones(data_points_fitting) * init_exposed_vac
    infected_zero_fitting = np.ones(data_points_fitting) * init_infected_zero
    infected_vac_fitting = np.ones(data_points_fitting) * init_infected_vac
    quarantined_zero_fitting = np.ones(data_points_fitting) * init_quarantined_zero
    quarantined_vac_fitting = np.ones(data_points_fitting) * init_quarantined_vac
    recovered_zero_fitting = np.ones(data_points_fitting) * init_recovered_zero
    recovered_vac_fitting = np.ones(data_points_fitting) * init_recovered_vac
    death_zero_fitting = np.ones(data_points_fitting) * init_death_zero
    death_vac_fitting = np.ones(data_points_fitting) * init_death_vac
    lams_zero_fitting = functions.recoveryRates(daily_recovered_fitting, quarantined_fitting_truth)
    lams_vac_fitting = functions.recoveryRates(daily_recovered_fitting, quarantined_fitting_truth)
    phis_zero_fitting = functions.DeathRates(daily_deaths_fitting, quarantined_fitting_truth)
    phis_vac_fitting = functions.DeathRates(daily_deaths_fitting, quarantined_fitting_truth)
    pop_fitting = np.ones(data_points_fitting) * total_population
    # vaccination for fitting
    alphas_fitting = vaccination_rates_fitting
    # virus variants for fitting
    pi_fitting = variant
    # x_data_fitting: susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac, quarantined_zero
    # , quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac, alphas, lams_zero, lams_vac, phis_zero,
    # phis_vac, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    # x_data_fitting is considered as a hyperparameter
    x_data_fitting = np.stack((susceptible_fitting, vaccinated_fitting, exposed_zero_fitting, exposed_vac_fitting,
                               infected_zero_fitting, infected_vac_fitting, quarantined_zero_fitting,
                               quarantined_vac_fitting, recovered_zero_fitting, recovered_vac_fitting,
                               death_zero_fitting, death_vac_fitting, alphas_fitting, lams_zero_fitting,
                               lams_vac_fitting, phis_zero_fitting, phis_vac_fitting, NPIs_no_vac_fitting,
                               NPIs_vac_fitting, vaccine_eff_fitting, pop_fitting), axis=0)
    print("initial susceptible_fitting:", x_data_fitting[0, 0])
    print("initial vaccinated_fitting:", x_data_fitting[1, 0])
    print("initial exposed_zero_fitting:", x_data_fitting[2, 0])
    print("initial exposed_vac_fitting:", x_data_fitting[3, 0])
    print("initial infected_zero_fitting:", x_data_fitting[4, 0])
    print("initial infected_vac_fitting:", x_data_fitting[5, 0])
    print("initial quarantined_zero_fitting:", x_data_fitting[6, 0])
    print("initial quarantined_vac_fitting: ", x_data_fitting[7, 0])
    print("initial recovered_zero_fitting: ", x_data_fitting[8, 0])
    print("initial recovered_vac_fitting: ", x_data_fitting[9, 0])
    print("initial death_zero_fitting: ", x_data_fitting[10, 0])
    print("initial death_vac_fitting: ", x_data_fitting[11, 0])
    print("daily vaccination rates: ", x_data_fitting[12, 0])
    print("daily recovery rates (zero): ", x_data_fitting[13, 0])
    print("daily recovery rates (vac): ", x_data_fitting[14, 0])
    print("daily death rates (zero): ", x_data_fitting[15, 0])
    print("daily death rates (vac): ", x_data_fitting[16, 0])
    print("NPIs_no_vac list: ", x_data_fitting[17, 0])
    print("NPIs_vac list: ", x_data_fitting[18, 0])
    print("vaccine effectiveness: ", x_data_fitting[19, 0])
    print("the total population: ", x_data_fitting[20, 0])
    # NPIs for fitting
    # initial values of parameters
    init_beta_zero_one = 0.2
    init_NPIs_eff_zero_one = []
    init_NPIs_eff_zero_one_upper_bounds = []
    init_NPIs_eff_zero_one_lower_bounds = []
    init_beta_zero_two = 0.1
    init_NPIs_eff_zero_two = []
    init_NPIs_eff_zero_two_upper_bounds = []
    init_NPIs_eff_zero_two_lower_bounds = []
    init_beta_vac_one = 0.1
    init_NPIs_eff_vac_one = []
    init_NPIs_eff_vac_one_upper_bounds = []
    init_NPIs_eff_vac_one_lower_bounds = []
    init_beta_vac_two = 0.2
    init_NPIs_eff_vac_two = []
    init_NPIs_eff_vac_two_upper_bounds = []
    init_NPIs_eff_vac_two_lower_bounds = []
    init_gamma = 0.2
    init_delta = 0.5
    init_NPIs_delay_days = 2
    # record the numbers of change points in four transmission rates
    tt = [0, 0, 0, 0]
    # setting the initial values for change points
    for ii in range(len(NPIs_no_vac_fitting)):
        # '1' stands for control; '2' stands for relaxation; '0' stands for unchanged
        if NPIs_no_vac_fitting[ii] == 2 and NPIs_vac_fitting[ii] == 2:
            init_NPIs_eff_zero_one.append(1e-6)
            init_NPIs_eff_zero_one_upper_bounds.append(3)
            init_NPIs_eff_zero_one_lower_bounds.append(0)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_two.append(1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(3)
            init_NPIs_eff_zero_two_lower_bounds.append(0)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(3)
            init_NPIs_eff_vac_one_lower_bounds.append(0)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_two.append(1e-6)
            init_NPIs_eff_vac_two_upper_bounds.append(3)
            init_NPIs_eff_vac_two_lower_bounds.append(0)
            tt[3] = tt[3] + 1
        if NPIs_no_vac_fitting[ii] == 2 and NPIs_vac_fitting[ii] == 1:
            init_NPIs_eff_zero_one.append(1e-6)
            init_NPIs_eff_zero_one_upper_bounds.append(3)
            init_NPIs_eff_zero_one_lower_bounds.append(0)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_two.append(1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(3)
            init_NPIs_eff_zero_two_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(3)
            init_NPIs_eff_vac_one_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_two.append(-1e-6)
            init_NPIs_eff_vac_two_upper_bounds.append(0)
            init_NPIs_eff_vac_two_lower_bounds.append(-3)
            tt[3] = tt[3] + 1
        if NPIs_no_vac_fitting[ii] == 2 and NPIs_vac_fitting[ii] == 0:
            init_NPIs_eff_zero_one.append(1e-6)
            init_NPIs_eff_zero_one_upper_bounds.append(3)
            init_NPIs_eff_zero_one_lower_bounds.append(0)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_two.append(1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(3)
            init_NPIs_eff_zero_two_lower_bounds.append(0)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(3)
            init_NPIs_eff_vac_one_lower_bounds.append(0)
            tt[2] = tt[2] + 1
        if NPIs_no_vac_fitting[ii] == 1 and NPIs_vac_fitting[ii] == 2:
            init_NPIs_eff_zero_one.append(-1e-6)
            init_NPIs_eff_zero_one_upper_bounds.append(0)
            init_NPIs_eff_zero_one_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_two.append(1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(3)
            init_NPIs_eff_zero_two_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(3)
            init_NPIs_eff_vac_one_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_two.append(1e-6)
            init_NPIs_eff_vac_two_upper_bounds.append(3)
            init_NPIs_eff_vac_two_lower_bounds.append(0)
            tt[3] = tt[3] + 1
        if NPIs_no_vac_fitting[ii] == 1 and NPIs_vac_fitting[ii] == 1:
            init_NPIs_eff_zero_one.append(-1e-6)
            init_NPIs_eff_zero_one_upper_bounds.append(0)
            init_NPIs_eff_zero_one_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_two.append(-1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(0)
            init_NPIs_eff_zero_two_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(-1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(0)
            init_NPIs_eff_vac_one_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_two.append(-1e-6)
            init_NPIs_eff_vac_two_upper_bounds.append(0)
            init_NPIs_eff_vac_two_lower_bounds.append(-3)
            tt[3] = tt[3] + 1
        if NPIs_no_vac_fitting[ii] == 1 and NPIs_vac_fitting[ii] == 0:
            init_NPIs_eff_zero_one.append(-1e-6)
            init_NPIs_eff_zero_one_upper_bounds.append(0)
            init_NPIs_eff_zero_one_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_two.append(-1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(0)
            init_NPIs_eff_zero_two_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(-1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(0)
            init_NPIs_eff_vac_one_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
        if NPIs_no_vac_fitting[ii] == 0 and NPIs_vac_fitting[ii] == 2:
            init_NPIs_eff_zero_two.append(1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(3)
            init_NPIs_eff_zero_two_lower_bounds.append(0)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(3)
            init_NPIs_eff_vac_one_lower_bounds.append(0)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_two.append(1e-6)
            init_NPIs_eff_vac_two_upper_bounds.append(3)
            init_NPIs_eff_vac_two_lower_bounds.append(0)
            tt[3] = tt[3] + 1
        if NPIs_no_vac_fitting[ii] == 0 and NPIs_vac_fitting[ii] == 1:
            init_NPIs_eff_zero_two.append(-1e-6)
            init_NPIs_eff_zero_two_upper_bounds.append(0)
            init_NPIs_eff_zero_two_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_one.append(-1e-6)
            init_NPIs_eff_vac_one_upper_bounds.append(0)
            init_NPIs_eff_vac_one_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_two.append(-1e-6)
            init_NPIs_eff_vac_two_upper_bounds.append(0)
            init_NPIs_eff_vac_two_lower_bounds.append(-3)
            tt[3] = tt[3] + 1
    # init includes: init_beta_zero_one, init_beta_zero_two, init_beta_vac_one, init_beta_vac_two, init_gamma,
    # init_delta, init_NPIs_delay_days, init_NPIs_eff_zero_one, init_NPIs_eff_zero_two, init_NPIs_eff_vac_one,
    # init_NPIs_eff_vac_two.
    # The variable init considered as a parameter
    init_tempt = [init_beta_zero_one, init_beta_zero_two, init_beta_vac_one, init_beta_vac_two,
                  init_gamma, init_delta, init_NPIs_delay_days]
    init = np.concatenate((init_tempt, init_NPIs_eff_zero_one, init_NPIs_eff_zero_two,
                           init_NPIs_eff_vac_one, init_NPIs_eff_vac_two), axis=0)
    # set boundaries of init
    upper_bounds_2 = np.array([3, 1, 1, 3, 1, 1, 7])  # ***
    lower_bounds_2 = np.array([0, 0, 0, 0, 0, 0, 0])  # ***
    upper_bounds_3 = np.concatenate((init_NPIs_eff_zero_one_upper_bounds, init_NPIs_eff_zero_two_upper_bounds,
                                     init_NPIs_eff_vac_one_upper_bounds, init_NPIs_eff_vac_two_upper_bounds), axis=0)
    lower_bounds_3 = np.concatenate((init_NPIs_eff_zero_one_lower_bounds, init_NPIs_eff_zero_two_lower_bounds,
                                     init_NPIs_eff_vac_one_lower_bounds, init_NPIs_eff_vac_two_lower_bounds), axis=0)
    upper_bounds = np.concatenate((upper_bounds_2, upper_bounds_3), axis=0)
    lower_bounds = np.concatenate((lower_bounds_2, lower_bounds_3), axis=0)
    bounds = (lower_bounds, upper_bounds)
    # print("init_tempt ", init_tempt)
    # print("init: ", init)
    # print("upper bounds: ", upper_bounds)
    # target of optimization
    y_data_fitting = quarantined_fitting_truth
    # print("quarantined average: ", y_data_fitting)
    para_est = functions.fit(init, x_data_fitting, y_data_fitting, bounds)
    # para_est = [9.98411798e-01, 7.58754755e-01, 3.44910101e-01, 1.71597458e+00, 9.52678508e-01, 1.98333071e-01,
    #             4.52398545e+00, -9.82242479e-01, -4.50371448e-01, -1.53956241e-03, -7.58754976e-01, -4.51760352e-01,
    #             -5.75597773e-04, -1.78195425e-04, -3.50169315e-01, -3.06957677e-01, -9.82090615e-02, -1.14701299e-03,
    #             -1.08373806e+00, -3.20492421e-05, -7.88504139e-03]

    print("para_est: ", para_est)
    print("it took", time.time() - start, "seconds.")


    # *********** predictions ******************
    # similar process for optimization
    # epidemiological data
    susceptible_prediction = np.ones(prediction_length) * init_susceptible
    vaccinated_prediction = np.ones(prediction_length) * init_vaccinated
    exposed_zero_prediction = np.ones(prediction_length) * init_exposed_zero
    exposed_vac_prediction = np.ones(prediction_length) * init_exposed_vac
    infected_zero_prediction = np.ones(prediction_length) * init_infected_zero
    infected_vac_prediction = np.ones(prediction_length) * init_infected_vac
    quarantined_zero_prediction = np.ones(prediction_length) * init_quarantined_zero
    quarantined_vac_prediction = np.ones(prediction_length) * init_quarantined_vac
    recovered_zero_prediction = np.ones(prediction_length) * init_recovered_zero
    recovered_vac_prediction = np.ones(prediction_length) * init_recovered_vac
    death_zero_prediction = np.ones(prediction_length) * init_death_zero
    death_vac_prediction = np.ones(prediction_length) * init_death_vac
    lams_zero_prediction = np.ones(prediction_length) * lams_zero_fitting[data_points_fitting - 1]
    lams_vac_prediction = np.ones(prediction_length) * lams_vac_fitting[data_points_fitting - 1]
    phis_zero_prediction = np.ones(prediction_length) * phis_zero_fitting[data_points_fitting - 1]
    phis_vac_prediction = np.ones(prediction_length) * phis_vac_fitting[data_points_fitting - 1]
    pop_prediction = np.ones(prediction_length) * total_population
    # vaccination for prediction
    alphas_prediction = vaccination_rates_prediction
    # virus variants for prediction
    pi_prediction = variant
    # x_data_prediction: susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
    # quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac, alphas,
    # lams_zero, lams_vac, phis_zero, phis_vac, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    # x_data_prediction is considered as a hyperparameter
    x_data_prediction = np.stack((susceptible_prediction, vaccinated_prediction, exposed_zero_prediction,
                                  exposed_vac_prediction, infected_zero_prediction, infected_vac_prediction,
                                  quarantined_zero_prediction, quarantined_vac_prediction, recovered_zero_prediction,
                                  recovered_vac_prediction, death_zero_prediction, death_vac_prediction,
                                  alphas_prediction, lams_zero_prediction, lams_vac_prediction, phis_zero_prediction,
                                  phis_vac_prediction, NPIs_no_vac_prediction, NPIs_vac_prediction,
                                  vaccine_eff_prediction, pop_prediction), axis=0)
    print("prediction of susceptible:", x_data_prediction[0, 0])
    print("prediction of vaccinated:", x_data_prediction[1, 0])
    print("prediction of exposed_zero:", x_data_prediction[2, 0])
    print("prediction of exposed_vac:", x_data_prediction[3, 0])
    print("prediction of infected_zero:", x_data_prediction[4, 0])
    print("prediction of infected_vac:", x_data_prediction[5, 0])
    print("prediction of quarantined_zero:", x_data_prediction[6, 0])
    print("prediction of quarantined_vac: ", x_data_prediction[7, 0])
    print("prediction of recovered_zero: ", x_data_prediction[8, 0])
    print("prediction of recovered_vac: ", x_data_prediction[9, 0])
    print("prediction of death_zero: ", x_data_prediction[10, 0])
    print("prediction of death_vac: ", x_data_prediction[11, 0])
    print("vaccination rates prediction: ", x_data_prediction[12, 0])
    print("daily recovery rates (zero) prediction: ", x_data_prediction[13, 0])
    print("daily recovery rates (vac) prediction: ", x_data_prediction[14, 0])
    print("daily death rates (zero) prediction: ", x_data_prediction[15, 0])
    print("daily death rates (vac) prediction: ", x_data_prediction[16, 0])
    print("NPIs_no_vac list for prediction: ", x_data_prediction[17, 0])
    print("NPIs_vac list for prediction: ", x_data_prediction[18, 0])
    print("vaccine effectiveness prediction: ", x_data_prediction[19, 0])
    print("the total population prediction: ", x_data_prediction[20, 0])
    # the merge of x_data_fitting and x_data_prediction
    x_data = np.concatenate((x_data_fitting, x_data_prediction), axis=1)
    print("x_data: ", np.shape(x_data))

    susceptible_fit, vaccinated_fit, exposed_zero_fit, exposed_vac_fit, infected_zero_fit, infected_vac_fit, \
    quarantined_zero_fit, quarantined_vac_fit, recovered_zero_fit, recovered_vac_fit, death_zero_fit, death_vac_fit = \
        functions.SEIQRD2Simulation(x_data, *para_est)
    # print(quarantined_vac_fit + quarantined_zero_fit)

    quarantined_fit = quarantined_zero_fit + quarantined_vac_fit
    death_fit = death_zero_fit + death_vac_fit
    # save the result of prediction
    df_infected = pd.read_csv("Israel_infected.csv")  # ***
    df_infected['predictions'] = quarantined_fit
    df_infected['deaths'] = death_fit
    df_infected.to_csv("Israel_infected_prediction_ablation3.csv", index=False)  # ***

    mape = functions.MAPECompute(quarantined_fit[data_points_fitting:], quarantined_prediction_truth)
    print("mape: ", mape)

    mape_death = functions.MAPECompute(death_fit[data_points_fitting:], total_deaths_truth[data_points_fitting:])
    print("mape (deaths): ", mape_death)
    # compute the transmission rates of the model
    # a constant ***
    betas_zero_one, betas_zero_two, betas_vac_one, betas_vac_two \
        = functions.transmissionRate(para_est[6], NPIs_no_vac, NPIs_vac, para_est[0], para_est[1], para_est[2],
                                     para_est[3], para_est[7: 10], para_est[10:14], para_est[14:18], para_est[18:21])
    trans_zero_one = betas_zero_one
    trans_zero_two = betas_zero_two
    trans_vac_one = []
    trans_vac_two = []
    for ii in range(data_points):
        trans_vac_one.append(betas_vac_one[ii] * (1 - vaccine_eff[ii]))
        trans_vac_two.append(betas_vac_two[ii] * (1 - vaccine_eff[ii]))

    # ******************* plot ************************
    time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***
    # ******************* prediction *******************
    fig, axs = plt.subplots(1, figsize=(9, 6), dpi=100)
    ax0 = axs.twinx()
    lns1, = axs.plot(time_scale, quarantined_fit, "-", linewidth=1, color='C3', label="quarantined (ablation3)")  # ***
    lns2, = ax0.plot(time_scale, death_fit, "-", linewidth=1, color='C8', label="total deaths (ablation3)")
    axs.axvline(x=time_scale[data_points_fitting - 1], linewidth=2, color='r')
    lns3 = axs.scatter(time_scale, quarantined_truth, marker="o", s=10, color='C4', label="actual quarantined")  # ***
    lns4 = ax0.scatter(time_scale, total_deaths_truth, marker="o", s=10, color='C9', label="actual total deaths")  # ***
    axs.annotate(r"", xy=(time_scale[data_points_fitting - 1], 2.5e4), xytext=(time_scale[data_points_fitting - 10], 2.5e4),
                 arrowprops=dict(arrowstyle="<-", color='r'), font={'size': 8}, color='r')
    axs.text(time_scale[data_points_fitting - 7], 2.6e4, s=r"Fitting", color='r', font={'size': 8})
    axs.annotate(r"", xy=(time_scale[data_points_fitting - 1], 5e4), xytext=(time_scale[data_points_fitting + 10], 5e4),
                 arrowprops=dict(arrowstyle="<-", color='r'), font={'size': 8}, color='r')
    axs.text(time_scale[data_points_fitting + 1], 5.1e4, s=r"Prediction", color='r', font={'size': 8})

    lns = [lns1, lns2, lns3, lns4]
    labels = [ll.get_label() for ll in lns]

    axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs.set_xlabel("Days")
    axs.set_ylim(0, 16e4)
    ax0.set_ylim(0, 3e3)
    axs.set_ylabel("Cases")
    ax0.set_ylabel("Deaths")
    axs.legend(lns, labels, loc='upper left', prop={"size": 8})  # ***
    plt.savefig("prediction_Israel_ablation3.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ************************ explanation ****************************
    # daily cases
    fig, axs = plt.subplots(4, figsize=(9, 6), dpi=100)
    axs[0].plot(time_scale, df_cases.iloc[0:, 9], "-", linewidth=1, color='g', label="daily cases")  # ***
    # booster
    axs[0].scatter(time_scale[2], 1e3, marker="P", color="black", s=8)
    axs[0].text(time_scale[3], 2e2, s="PI", color="black", font={'size': 6})
    # the first NPI
    axs[0].scatter(time_scale[5], 2.3e4, marker="v", color='r', s=8)  # for the unvaccinated
    axs[0].text(time_scale[6], 2.3e4, s=r"$C_{1}^{0}$", color='r', font={'size': 6})  # for the unvaccinated
    axs[0].scatter(time_scale[5], 1.9e4, marker="v", color='black', s=8)  # for the vaccinated
    axs[0].text(time_scale[6], 1.9e4, s=r"$C_{1}^{v}$", color='black', font={'size': 6})  # for the vaccinated
    # the second NPI
    axs[0].scatter(time_scale[19], 2.3e4, marker="v", color='r', s=8)  # for the unvaccinated
    axs[0].text(time_scale[20], 2.3e4, s=r"$C_{2}^{0}$", color='r', font={'size': 6})  # for the unvaccinated
    axs[0].scatter(time_scale[19], 1.9e4, marker="v", color='black', s=8)  # for the vaccinated
    axs[0].text(time_scale[20], 1.9e4, s=r"$C_{2}^{v}$", color='black', font={'size': 6})
    # the third NPI
    axs[0].scatter(time_scale[33], 2.3e4, marker="v", color='r', s=8)  # for the unvaccinated
    axs[0].text(time_scale[34], 2.3e4, s=r"$C_{3}^{0}$", color='r', font={'size': 6})  # for the unvaccinated
    # the fourth NPI
    axs[0].scatter(time_scale[65], 1.9e4, marker="v", color='black', s=8)  # for the vaccinated
    axs[0].text(time_scale[66], 1.9e4, s=r"$C_{3}^{v}$", color='black', font={'size': 6})  # for the vaccinated
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
    axs[1].scatter(time_scale[2], 0.05, marker="P", color="black", s=8)
    axs[1].text(time_scale[3], 0.01, s=r"start boosters", color="black", font={'size': 8})
    axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs[1].set_ylim(0, 1.0)
    axs[1].set_ylabel("Effectiveness \n or coverage")
    axs[1].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # basic transmission rates
    print("betas_zero_one", betas_zero_one)
    print("betas_zero_two", betas_zero_two)
    print("betas_vac_one", betas_vac_one)
    print("betas_vac_two", betas_vac_two)
    axs[2].plot(time_scale, betas_zero_one, "-", linewidth=1, color='r', label=r"$\beta_{base,1}^{0}$")  # ***
    axs[2].plot(time_scale, betas_zero_two, "-", linewidth=1, color='g', label=r"$\beta_{base,2}^{0}$")  # ***
    axs[2].plot(time_scale, betas_vac_one, "-", linewidth=1, color='b', label=r"$\beta_{base,1}^{v}$")  # ***
    axs[2].plot(time_scale, betas_vac_two, "-", linewidth=1, color='purple', label=r"$\beta_{base,2}^{v}$")  # ***

    # the first NPI
    axs[2].scatter(time_scale[5], 3.3, marker="v", color='r', s=8)  # for the unvaccinated
    axs[2].text(time_scale[6], 3.3, s=r"$C_{1}^{0}$", color='r', font={'size': 6})  # for the unvaccinated
    axs[2].scatter(time_scale[5], 2.9, marker="v", color='black', s=8)  # for the vaccinated
    axs[2].text(time_scale[6], 2.8, s=r"$C_{1}^{v}$", color='black', font={'size': 6})  # for the vaccinated
    # the second NPI
    axs[2].scatter(time_scale[19], 3.3, marker="v", color='r', s=8)  # for the unvaccinated
    axs[2].text(time_scale[20], 3.3, s=r"$C_{2}^{0}$", color='r', font={'size': 6})  # for the unvaccinated
    axs[2].scatter(time_scale[19], 2.9, marker="v", color='black', s=8)  # for the vaccinated
    axs[2].text(time_scale[20], 2.8, s=r"$C_{2}^{v}$", color='black', font={'size': 6})
    # the third NPI
    axs[2].scatter(time_scale[33], 3.3, marker="v", color='r', s=8)  # for the unvaccinated
    axs[2].text(time_scale[34], 3.3, s=r"$C_{3}^{0}$", color='r', font={'size': 6})  # for the unvaccinated
    # the fourth NPI
    axs[2].scatter(time_scale[65], 2.9, marker="v", color='black', s=8)  # for the vaccinated
    axs[2].text(time_scale[66], 2.9, s=r"$C_{3}^{v}$", color='black', font={'size': 6})  # for the vaccinated

    axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[2].set_xlabel("Days")
    axs[2].set_ylim(0, 4.0)
    axs[2].set_ylabel("Basic \n transmission \n rates")
    axs[2].legend(loc='upper left')  # ***
    axs[2].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # transmission rates
    axs[3].plot(time_scale, trans_zero_one, "-", linewidth=1, color='r', label=r"$\beta_{1}^{0}$")  # ***
    axs[3].plot(time_scale, trans_zero_two, "-", linewidth=1, color='g', label=r"$\beta_{2}^{0}$")  # ***
    axs[3].plot(time_scale, trans_vac_one, "-", linewidth=1, color='b', label=r"$\beta_{1}^{v}$")  # ***
    axs[3].plot(time_scale, trans_vac_two, "-", linewidth=1, color='purple', label=r"$\beta_{2}^{v}$")  # ***
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
    axs[3].set_ylim(0, 4.0)

    axs[3].set_ylabel("Effective \n transmission \n rates")
    # axs[3].legend(loc='upper left')  # ***
    axs[3].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})
    plt.savefig("transmissionrates_Israel_sensitivity.png", dpi=400, bbox_inches='tight')
    plt.show()
