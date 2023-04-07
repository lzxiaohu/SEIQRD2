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

    # timer
    start = time.time()
    # fix the seed
    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    # inputs
    # the information of COVID-19: time, data points of fitting, vaccination, NPIs, and virus variants, epidemiology
    district = "Greece_wave"  # ***
    df_cases = pd.read_csv("Greece_wave1_integrated_revised.csv")  # ***
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
    vaccination_rates_prediction = np.ones(prediction_length) * 0.15 / 100.0  # a constant of 0.15 ***
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
    # variants of virus: Delta (B.1.617.2)
    variant = "Delta"
    # Targets: cases
    daily_cases_fitting = df_cases.iloc[0:data_points_fitting, 12]

    daily_deaths_fitting_truth = df_cases.iloc[0:data_points_fitting, 13]
    daily_deaths_prediction_truth = df_cases.iloc[-1 * prediction_length:, 13]
    daily_deaths_truth = df_cases.iloc[0:, 13]
    total_deaths_truth = df_cases.iloc[0:, 6]

    daily_recovered_fitting_truth = df_cases.iloc[0:data_points_fitting, 14]
    daily_recovered_prediction_truth = df_cases.iloc[-1 * prediction_length:, 14]
    daily_recovered_truth = df_cases.iloc[0:, 14]
    total_recovered_truth = df_cases.iloc[0:, 7]

    quarantined_fitting_truth = df_cases.iloc[0:data_points_fitting, 16]
    quarantined_prediction_truth = df_cases.iloc[-1 * prediction_length:, 16]
    quarantined_truth = df_cases.iloc[0:, 16]
    # Parameters:
    # initial values of hyper parameters
    total_population = df_cases.iloc[0, 15]
    init_vaccinated = df_cases.iloc[0, 17] / 100.0 * total_population  # remember 100
    init_exposed_zero = 1775  # *** 1.7e-4 * total_population; 1775
    init_exposed_vac = 1044   # *** 1e-4 * total_population; 1044
    init_infected_zero = 100  # *** 1.7e-4 * total_population ; 1500
    init_infected_vac = 100   # *** 1e-4 * total_population ; 1000
    init_quarantined_zero = 20890  # *** total quarantined about 3e-3:  2e-3 * total_population; 20890
    init_quarantined_vac = 9940  # *** df_cases.iloc[0, 16] - init_quarantined_zero; 9940
    init_recovered_zero = 104453  # ***
    init_recovered_vac = df_cases.iloc[0, 7] - init_recovered_zero
    init_death_zero = 1044  # ***
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
    lams_zero_fitting = functions.recoveryRates(daily_recovered_fitting_truth, quarantined_fitting_truth)
    lams_vac_fitting = functions.recoveryRates(daily_recovered_fitting_truth, quarantined_fitting_truth)
    lams_prediction = functions.recoveryRates(daily_recovered_prediction_truth, quarantined_prediction_truth)
    phis_zero_fitting = functions.DeathRates(daily_deaths_fitting_truth, quarantined_fitting_truth)
    phis_vac_fitting = functions.DeathRates(daily_deaths_fitting_truth, quarantined_fitting_truth)
    phis_prediction = functions.DeathRates(daily_deaths_prediction_truth, quarantined_prediction_truth)
    pop_fitting = np.ones(data_points_fitting) * total_population
    # vaccination for fitting
    alphas_fitting = vaccination_rates_fitting
    # virus variants for fitting
    pi_fitting = variant
    # x_data_fitting: susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac, quarantined_zero
    # , quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac, alphas, lams_zero, lams_vac, phis_zero,
    # phis_vac, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    # x_data_fitting is considered as a hyperparameter set
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
    init_beta_zero_one = 0.2  # ***
    init_NPIs_eff_zero_one = []
    init_NPIs_eff_zero_one_upper_bounds = []
    init_NPIs_eff_zero_one_lower_bounds = []
    init_beta_zero_two = 0.1  # ***
    init_NPIs_eff_zero_two = []
    init_NPIs_eff_zero_two_upper_bounds = []
    init_NPIs_eff_zero_two_lower_bounds = []
    init_beta_vac_one = 0.1  # ***
    init_NPIs_eff_vac_one = []
    init_NPIs_eff_vac_one_upper_bounds = []
    init_NPIs_eff_vac_one_lower_bounds = []
    init_beta_vac_two = 0.4  # ***
    init_NPIs_eff_vac_two = []
    init_NPIs_eff_vac_two_upper_bounds = []
    init_NPIs_eff_vac_two_lower_bounds = []
    init_gamma = 0.2  # ***
    init_delta = 0.5  # ***
    init_NPIs_delay_days = 2  # *** 2
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
    # The variable initials are considered as a parameter set
    init_tempt = [init_beta_zero_one, init_beta_zero_two, init_beta_vac_one, init_beta_vac_two,
                  init_gamma, init_delta, init_NPIs_delay_days]
    init = np.concatenate((init_tempt, init_NPIs_eff_zero_one, init_NPIs_eff_zero_two,
                           init_NPIs_eff_vac_one, init_NPIs_eff_vac_two), axis=0)
    # set boundaries of init
    upper_bounds_2 = np.array([5, 1, 1, 5, 1, 1, 7])  # ***
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
    # para_est = [1.77194951e+00, 7.46598774e-01, 4.82084866e-01, 2.49210243e+00, 9.18269416e-01, 9.28694382e-01,
    #             1.83989552e+00, 7.20664020e-03, -1.07941393e-03, -6.98647334e-03, 7.97711792e-03, -6.59282376e-01,
    #             -1.14094442e-01, 3.25405758e-03, -4.85341609e-01, -4.36587027e-02, 5.21536063e-02]
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
    # lams_zero_prediction = np.ones(prediction_length) * ???  # ***
    # lams_zero_prediction = lams_prediction
    lams_vac_prediction = np.ones(prediction_length) * lams_vac_fitting[data_points_fitting - 1]
    # lams_vac_prediction = np.ones(prediction_length) * ???  # ***
    # lams_vac_prediction = lams_prediction
    phis_zero_prediction = np.ones(prediction_length) * phis_zero_fitting[data_points_fitting - 1]
    # phis_zero_prediction = phis_prediction
    phis_vac_prediction = np.ones(prediction_length) * phis_vac_fitting[data_points_fitting - 1]
    # phis_vac_prediction = phis_prediction
    pop_prediction = np.ones(prediction_length) * total_population
    # vaccination for prediction
    alphas_prediction = vaccination_rates_prediction
    # virus variants for prediction
    pi_prediction = variant
    # x_data_prediction: susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
    # quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac, alphas,
    # lams_zero, lams_vac, phis_zero, phis_vac, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    # x_data_prediction is considered as a hyperparameter
    print("lams_prediction: ", lams_prediction)
    print("phis_prediction: ", phis_prediction)
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
    quarantined_fit = quarantined_zero_fit + quarantined_vac_fit
    # save the result of prediction
    df_infected = pd.read_csv("Greece_infected.csv")  # ***
    df_infected['predictions'] = quarantined_fit
    df_infected.to_csv("Greece_infected_prediction.csv", index=False)

    # *********** Evaluation ******************
    mape = functions.MAPECompute(quarantined_fit[data_points_fitting:], quarantined_prediction_truth)
    print("mape: ", mape)

    # *********** Explanation & Analysis ******************
    # compute the transmission rates of the model
    # a constant ***
    betas_zero_one, betas_zero_two, betas_vac_one, betas_vac_two \
        = functions.transmissionRate(para_est[6], NPIs_no_vac, NPIs_vac, para_est[0], para_est[1], para_est[2],
                                     para_est[3], para_est[7: 10], para_est[10:13], para_est[13:16], para_est[16:17])
    trans_zero_one = betas_zero_one
    trans_zero_two = betas_zero_two
    trans_vac_one = []
    trans_vac_two = []
    for ii in range(data_points):
        trans_vac_one.append(betas_vac_one[ii] * (1 - vaccine_eff[ii]))
        trans_vac_two.append(betas_vac_two[ii] * (1 - vaccine_eff[ii]))

    # ******************* Plot & Display ************************
    time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***

    # the display of predictions
    fig, axs = plt.subplots(1, figsize=(9, 6), dpi=100)
    axs.plot(time_scale, quarantined_fit, "-", linewidth=1, color='g', label="quarantined (prediction)")  # ***
    axs.scatter(time_scale[0:data_points_fitting], quarantined_truth[0: data_points_fitting], marker="o", s=8,
                   color='r')  # ***
    axs.scatter(time_scale[data_points_fitting:], quarantined_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="quarantined (actual)")  # ***
    axs.set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs.set_ylim(0, 12e4)
    axs.set_ylabel("Cases")
    axs.legend(loc='upper right', prop={"size": 12})  # ***
    plt.savefig("prediction_Greece.png", dpi=400, bbox_inches='tight')
    plt.show()

    # the display of analysis
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
    axs[0].set_ylim(0, 12e3)
    axs[0].set_ylabel("Cases")
    axs[0].legend(loc='upper right', prop={"size": 8})  # ***

    # vaccination info
    file_name = "Greece_effectiveness_vaccine.csv"  # ***
    df_vaccine = pd.read_csv(file_name)  # ***
    total_vac = df_vaccine.iloc[243:335, 1] / 100
    # print(total_vac)
    total_booster = df_vaccine.iloc[243:335, 2] / 100
    axs[1].plot(time_scale, vaccine_eff, "-",linewidth=1, color='b', label="vacc eff")
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
    print("betas_zero_one", betas_zero_one)
    print("betas_zero_two", betas_zero_two)
    print("betas_vac_one", betas_vac_one)
    print("betas_vac_two", betas_vac_two)
    axs[2].plot(time_scale, betas_zero_one, "-", linewidth=1, color='r', label=r"$\beta_{base,1}^{0}$")  # ***
    axs[2].plot(time_scale, betas_zero_two, "-", linewidth=1, color='g', label=r"$\beta_{base,2}^{0}$")  # ***
    axs[2].scatter(time_scale[47], betas_zero_two[46]-0.05, marker="o", c='w', edgecolor="g", s=12)
    axs[2].scatter(time_scale[66], betas_zero_two[65], marker="o", c='w', edgecolor="g", s=12)
    axs[2].plot(time_scale, betas_vac_one, "-", linewidth=1, color='b', label=r"$\beta_{base,1}^{v}$")  # ***
    axs[2].scatter(time_scale[47], betas_vac_one[46]-0.05, marker="o", c='w', edgecolor="b", s=12)
    axs[2].plot(time_scale, betas_vac_two, "-", linewidth=1, color='purple', label=r"$\beta_{base,2}^{v}$")  # ***
    axs[2].scatter(time_scale[22], betas_vac_two[21]-0.05, marker="o", c='w', edgecolor="purple", s=12)
    axs[2].scatter(time_scale[20], 0.2, marker="^", color='black', s=16)  # for the vaccinated
    axs[2].annotate(r"$R_{1}^{v}$", xy=(time_scale[22], betas_vac_two[22]-0.1), xytext=(time_scale[21], 0.1),
                    arrowprops=dict(arrowstyle="->"), font={'size': 8})
    axs[2].scatter(time_scale[45], 2.7, marker="v", color='r', s=16)  # for the unvaccinated
    axs[2].annotate(r"$C_{1}^{v}$", xy=(time_scale[47], betas_zero_two[47]-0.1), xytext=(time_scale[46], 2.6),
                    arrowprops=dict(arrowstyle="->", color='r'), font={'size': 8}, color='r')
    axs[2].annotate(r"$C_{1}^{v}$", xy=(time_scale[47], betas_vac_one[47]-0.1), xytext=(time_scale[46], 2.6),
                    arrowprops=dict(arrowstyle="->", color='r'), font={'size': 8}, color='r')
    axs[2].scatter(time_scale[65], 2.7, marker="v", color='r', s=16)  # for the unvaccinated
    axs[2].annotate(r"$C_{2}^{v}$", xy=(time_scale[66], betas_zero_two[66]-0.1), xytext=(time_scale[66], 2.6),
                    arrowprops=dict(arrowstyle="->", color='r'), font={'size': 8}, color='r')
    axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[2].set_xlabel("Days")
    axs[2].set_ylim(0, 3.5)
    axs[2].set_ylabel("Basic \n transmission \n rates")
    axs[2].legend(loc='upper left')  # ***
    axs[2].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})

    # the display of impact of NPIs and vaccination
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
    axs[3].set_ylim(0, 2.5)

    axs[3].set_ylabel("Transmission \n rates")
    # axs[3].legend(loc='upper left')  # ***
    axs[3].legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0, prop={"size": 8})
    plt.savefig("transmissionrates_Greece.png", dpi=400, bbox_inches='tight')
    plt.show()

    # ************************ explanation ****************************
    # daily recovered and deaths
    recovered_fit = recovered_zero_fit + recovered_vac_fit
    death_fit = death_zero_fit + death_vac_fit
    lams_truth = functions.recoveryRates(daily_recovered_truth, quarantined_truth)
    lams_fitting = functions.recoveryRates(daily_recovered_fitting_truth, quarantined_fitting_truth)
    lams_prediction = np.ones(prediction_length) * lams_fitting[data_points_fitting - 1]
    lams_fit = np.concatenate((lams_fitting, lams_prediction), axis=0)

    phis_truth = functions.recoveryRates(daily_deaths_truth, quarantined_truth)
    phis_fitting = functions.recoveryRates(daily_deaths_fitting_truth, quarantined_fitting_truth)
    phis_prediction = np.ones(prediction_length) * phis_fitting[data_points_fitting - 1]
    phis_fit = np.concatenate((phis_fitting, phis_prediction), axis=0)

    fig, axs = plt.subplots(4, figsize=(9, 6), dpi=100)
    mape = functions.MAPECompute(recovered_fit[data_points_fitting:], total_recovered_truth[data_points_fitting:])
    print("mape(recovered): ", mape)
    print(total_recovered_truth[-2:])
    axs[0].plot(time_scale, recovered_fit, "-", linewidth=1, color='g', label="recovered cases(prediction)")  # ***
    axs[0].scatter(time_scale[0:data_points_fitting], total_recovered_truth[0: data_points_fitting], marker="o", s=8,
                   color='r')  # ***
    axs[0].scatter(time_scale[data_points_fitting:], total_recovered_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="recovered cases (actual)")  # ***

    axs[0].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs[0].set_ylim(0, 7e5)
    axs[0].set_ylabel("Cases")
    axs[0].legend(loc='upper right', prop={"size": 12})  # ***
    #

    # lambs
    axs[1].plot(time_scale, lams_fit, "-", linewidth=1, color='g', label="recovered rates(prediction)")  # ***
    axs[1].scatter(time_scale[0:data_points_fitting], lams_truth[0: data_points_fitting], marker="o", s=8,
                   color='r')  # ***
    axs[1].scatter(time_scale[data_points_fitting:], lams_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="recovered cases (actual)")  # ***

    axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs[1].set_ylim(0, 0.2)
    axs[1].set_ylabel("recovered rates")
    axs[1].legend(loc='upper right', prop={"size": 12})  # ***

    mape = functions.MAPECompute(death_fit[data_points_fitting:], total_deaths_truth[data_points_fitting:])
    print("mape(death): ", mape)
    print(total_deaths_truth[-2:])
    axs[2].plot(time_scale, death_fit, "-", linewidth=1, color='g', label="recovered cases(prediction)")  # ***
    axs[2].scatter(time_scale[0:data_points_fitting], total_deaths_truth[0: data_points_fitting], marker="o", s=8,
                   color='r')  # ***
    axs[2].scatter(time_scale[data_points_fitting:], total_deaths_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="recovered cases (actual)")  # ***

    axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs[2].set_ylim(0, 1e4)
    axs[2].set_ylabel("Cases")
    axs[2].legend(loc='upper right', prop={"size": 12})  # ***

    # phis
    axs[3].plot(time_scale, phis_fit, "-", linewidth=1, color='g', label="death rates(prediction)")  # ***
    axs[3].scatter(time_scale[0:data_points_fitting], phis_truth[0: data_points_fitting], marker="o", s=8,
                   color='r')  # ***
    axs[3].scatter(time_scale[data_points_fitting:], phis_truth[data_points_fitting:],
                marker="*", s=10, color='r', label="recovered cases (actual)")  # ***

    axs[3].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    # axs[0].set_xlabel("Days")
    axs[3].set_ylim(0, 0.002)
    axs[3].set_ylabel("death rates")
    axs[3].legend(loc='upper right', prop={"size": 12})  # ***

    plt.savefig("prediction_Greece_rates.png", dpi=400, bbox_inches='tight')
    plt.show()






