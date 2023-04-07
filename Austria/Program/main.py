import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import sys
import functions
import time
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # time counting
    start = time.time()
    # fix the seed
    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    # inputs
    district = "Austria_wave"  # ***
    df_cases = pd.read_csv("Austria_wave1_integrated.csv")  # ***
    # the period of time
    starting_time = datetime.strptime(df_cases.iloc[0, 0], "%d/%m/%Y")
    ending_time = datetime.strptime(df_cases.iloc[-1, 0], "%d/%m/%Y")
    # PIs: vaccination
    vaccination_rate = df_cases.iloc[0:, 1] / 100.0
    # NPIs: interventions
    NPIs_no_vac = df_cases.iloc[0:, 2]
    NPIs_vac = df_cases.iloc[0:, 3]
    # variants of virus: Alpha
    variant = "Delta"
    data_points = len(vaccination_rate)

    # Targets: cases
    daily_cases_average = df_cases.iloc[0:, 12]
    daily_deaths_average = df_cases.iloc[0:, 13]
    daily_recovered_average = df_cases.iloc[0:, 14]
    quarantined_average = df_cases.iloc[0:, 16]

    # other parameters
    total_population = df_cases.iloc[0, 15]
    # print(total_population, df_cases.iloc[0, 5])
    init_susceptible = total_population - df_cases.iloc[0, 5]
    susceptible = np.ones(data_points) * init_susceptible
    init_vaccinated = 1e+5  # df_cases.iloc[0, 1] / 100 * total_population  # ***
    vaccinated = np.ones(data_points) * init_vaccinated
    init_exposed_zero = 0.001 * init_susceptible
    exposed_zero = np.ones(data_points) * init_exposed_zero
    init_exposed_vac = 0.001 * init_vaccinated  # 0.01 * init_vaccinated
    exposed_vac = np.ones(data_points) * init_exposed_vac
    infected_zero = np.ones(data_points) * 0.2 * init_exposed_zero
    infected_vac = np.ones(data_points) * 200
    # print(infected_vac, infected_zero)
    quarantined_zero = np.ones(data_points) * quarantined_average[0]
    quarantined_vac = np.ones(data_points) * 1e+3
    recovered_zero = np.ones(data_points) * df_cases.iloc[0, 7]
    recovered_vac = np.ones(data_points) * 300
    death_zero = np.ones(data_points) * df_cases.iloc[0, 6]
    death_vac = np.ones(data_points) * 10
    pi = variant
    alphas = vaccination_rate
    lam_zeros = functions.RecoveryRates(daily_recovered_average, quarantined_average)
    lam_vacs = functions.RecoveryRates(daily_recovered_average, quarantined_average)
    phi_zeros = functions.DeathRates(daily_deaths_average, quarantined_average)
    phi_vacs = functions.DeathRates(daily_deaths_average, quarantined_average)
    pop = np.ones(data_points) * total_population

    init_beta_zero_one = 0.6
    init_NPIs_eff_zero_ones = []
    init_NPIs_eff_zero_ones_upper_bounds = []
    init_NPIs_eff_zero_ones_lower_bounds = []
    init_beta_zero_two = 0.5
    init_NPIs_eff_zero_twos = []
    init_NPIs_eff_zero_twos_upper_bounds = []
    init_NPIs_eff_zero_twos_lower_bounds = []
    init_beta_vac_one = 0.5
    init_NPIs_eff_vac_ones = []
    init_NPIs_eff_vac_ones_upper_bounds = []
    init_NPIs_eff_vac_ones_lower_bounds = []
    init_beta_vac_two = 0.5
    init_NPIs_eff_vac_twos = []
    init_NPIs_eff_vac_twos_upper_bounds = []
    init_NPIs_eff_vac_twos_lower_bounds = []
    init_gamma = 0.2
    init_delta = 0.5
    init_NPIs_delay_days = 2

    tt = [0, 0, 0, 0]
    # '1' stands for control; '2' stands for relaxation; '0' stands for unchanged
    for ii in range(len(NPIs_no_vac)):
        if NPIs_no_vac[ii] == 2 and NPIs_vac[ii] == 2:
            init_NPIs_eff_zero_ones.append(1e-6)
            init_NPIs_eff_zero_ones_upper_bounds.append(3)
            init_NPIs_eff_zero_ones_lower_bounds.append(0)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_twos.append(1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(3)
            init_NPIs_eff_zero_twos_lower_bounds.append(0)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(3)
            init_NPIs_eff_vac_ones_lower_bounds.append(0)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_twos.append(1e-6)
            init_NPIs_eff_vac_twos_upper_bounds.append(3)
            init_NPIs_eff_vac_twos_lower_bounds.append(0)
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] == 2 and NPIs_vac[ii] == 1:
            init_NPIs_eff_zero_ones.append(1e-6)
            init_NPIs_eff_zero_ones_upper_bounds.append(3)
            init_NPIs_eff_zero_ones_lower_bounds.append(0)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_twos.append(1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(3)
            init_NPIs_eff_zero_twos_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(3)
            init_NPIs_eff_vac_ones_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_twos.append(-1e-6)
            init_NPIs_eff_vac_twos_upper_bounds.append(0)
            init_NPIs_eff_vac_twos_lower_bounds.append(-3)
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] == 2 and NPIs_vac[ii] == 0:
            init_NPIs_eff_zero_ones.append(1e-6)
            init_NPIs_eff_zero_ones_upper_bounds.append(3)
            init_NPIs_eff_zero_ones_lower_bounds.append(0)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_twos.append(1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(3)
            init_NPIs_eff_zero_twos_lower_bounds.append(0)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(3)
            init_NPIs_eff_vac_ones_lower_bounds.append(0)
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 1 and NPIs_vac[ii] == 2:
            init_NPIs_eff_zero_ones.append(-1e-6)
            init_NPIs_eff_zero_ones_upper_bounds.append(0)
            init_NPIs_eff_zero_ones_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_twos.append(1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(3)
            init_NPIs_eff_zero_twos_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(3)
            init_NPIs_eff_vac_ones_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_twos.append(1e-6)
            init_NPIs_eff_vac_twos_upper_bounds.append(3)
            init_NPIs_eff_vac_twos_lower_bounds.append(0)
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] == 1 and NPIs_vac[ii] == 1:
            init_NPIs_eff_zero_ones.append(-1e-6)
            init_NPIs_eff_zero_ones_upper_bounds.append(0)
            init_NPIs_eff_zero_ones_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_twos.append(-1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(0)
            init_NPIs_eff_zero_twos_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(-1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(0)
            init_NPIs_eff_vac_ones_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_twos.append(-1e-6)
            init_NPIs_eff_vac_twos_upper_bounds.append(0)
            init_NPIs_eff_vac_twos_lower_bounds.append(-3)
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] == 1 and NPIs_vac[ii] == 0:
            init_NPIs_eff_zero_ones.append(-1e-6)
            init_NPIs_eff_zero_ones_upper_bounds.append(0)
            init_NPIs_eff_zero_ones_lower_bounds.append(-3)
            tt[0] = tt[0] + 1
            init_NPIs_eff_zero_twos.append(-1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(0)
            init_NPIs_eff_zero_twos_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(-1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(0)
            init_NPIs_eff_vac_ones_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] == 2:
            init_NPIs_eff_zero_twos.append(1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(3)
            init_NPIs_eff_zero_twos_lower_bounds.append(0)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(3)
            init_NPIs_eff_vac_ones_lower_bounds.append(0)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_twos.append(1e-6)
            init_NPIs_eff_vac_twos_upper_bounds.append(3)
            init_NPIs_eff_vac_twos_lower_bounds.append(0)
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] == 1:
            init_NPIs_eff_zero_twos.append(-1e-6)
            init_NPIs_eff_zero_twos_upper_bounds.append(0)
            init_NPIs_eff_zero_twos_lower_bounds.append(-3)
            tt[1] = tt[1] + 1
            init_NPIs_eff_vac_ones.append(-1e-6)
            init_NPIs_eff_vac_ones_upper_bounds.append(0)
            init_NPIs_eff_vac_ones_lower_bounds.append(-3)
            tt[2] = tt[2] + 1
            init_NPIs_eff_vac_twos.append(-1e-6)
            init_NPIs_eff_vac_twos_upper_bounds.append(0)
            init_NPIs_eff_vac_twos_lower_bounds.append(-3)
            tt[3] = tt[3] + 1

    # print(init_NPIs_eff_zero_twos)
    x_data = np.stack((susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero,
                       infected_vac, quarantined_zero,
                       quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac,
                       alphas, lam_zeros, lam_vacs,
                       phi_zeros, phi_vacs, NPIs_no_vac, NPIs_vac, pop), axis=0)
    print("initial susceptible:", x_data[0, 1])
    print("initial vaccinated:", x_data[1, 1])
    print("initial exposed_zero:", x_data[2, 1])
    print("initial exposed_vac:", x_data[3, 1])
    print("initial infected_zero:", x_data[4, 1])
    print("initial infected_vac:", x_data[5, 1])
    #print("lambda: ", lam_zeros)
    init_tempt = [init_beta_zero_one, init_beta_zero_two, init_beta_vac_one, init_beta_vac_two,
                  init_gamma, init_delta, init_NPIs_delay_days]
    init = np.concatenate((init_tempt, init_NPIs_eff_zero_ones, init_NPIs_eff_zero_twos,
                           init_NPIs_eff_vac_ones, init_NPIs_eff_vac_twos), axis=0)

    upper_bounds_2 = np.array([3, 3, 3, 3, 1, 1, 7])
    lower_bounds_2 = np.array([0, 0, 0, 0, 0, 0, 0])
    upper_bounds_3 = np.concatenate((init_NPIs_eff_zero_ones_upper_bounds, init_NPIs_eff_zero_twos_upper_bounds,
                           init_NPIs_eff_vac_ones_upper_bounds, init_NPIs_eff_vac_twos_upper_bounds), axis=0)
    lower_bounds_3 = np.concatenate((init_NPIs_eff_zero_ones_lower_bounds, init_NPIs_eff_zero_twos_lower_bounds,
                           init_NPIs_eff_vac_ones_lower_bounds, init_NPIs_eff_vac_twos_lower_bounds), axis=0)
    upper_bounds = np.concatenate((upper_bounds_2, upper_bounds_3), axis=0)
    lower_bounds = np.concatenate((lower_bounds_2, lower_bounds_3), axis=0)
    bounds = (lower_bounds, upper_bounds)
    # print(init_tempt, init_tempt)
    #print(init)
    #print("tt: ", tt)
    #output = functions.SVEIIQRDFit(x_data, *init)
    #print("quarantined: ", output)
    y_data = quarantined_average
    #para_est = functions.fit(init, x_data, y_data, bounds)
    #print("para_est: ", para_est)
    #print("it took", time.time() - start, "seconds.")
    para_est = [3.73888244e-01,  3.00000000e+00,  3.00000000e+00,  3.00000000e+00,
                1.00000000e+00,  6.11591122e-01,  2.10775041e+00, -4.23851778e-24,
                -3.72863707e-22, -4.96985379e-16, -5.71679596e-10, -1.41679445e-16,
                -1.07238727e-01,  1.03372335e-34,  4.02350774e-31, -1.08049422e-20,
                -5.56452842e-19, -1.32950204e-10, -7.12923747e-01, -2.02557656e-13,
                -6.29723521e-01,  1.33947741e-35,  3.09999544e-26, -3.26334333e-19,
                -2.36011084e-17, -5.27234327e-01, -2.10330812e-14, -9.72127939e-01,
                -1.64482916e-13,  2.04248310e-34,  2.60841071e-24, -4.46803475e-06,
                -2.51828178e-05, -2.99999588e+00, -2.09635830e-05,  2.34584091e-36,
                3.73727141e-23]

    susceptible_fit, vaccinated_fit, exposed_zero_fit, exposed_vac_fit, infected_zero_fit, infected_vac_fit, \
    quarantined_zero_fit, quarantined_vac_fit, recovered_zero_fit, recovered_vac_fit, death_zero_fit, death_vac_fit = \
        functions.SVEIIQRDSimulation(x_data, *para_est)
    # print(quarantined_vac_fit + quarantined_zero_fit)

    quarantined_fit = quarantined_zero_fit + quarantined_vac_fit
    mape = functions.MAPECompute(quarantined_fit, quarantined_average)
    print("mape: ", mape)
    # ***
    beta_zero_ones, beta_zero_twos, beta_vac_ones, beta_vac_twos \
        = functions.Betas(para_est[6], NPIs_no_vac, NPIs_vac, para_est[0], para_est[1], para_est[2], para_est[3],
                          para_est[7: 15], para_est[15:23], para_est[23:31], para_est[31:37])
    #print("beta_vac_ones: ", beta_vac_ones)
    time_scale = [starting_time + timedelta(days=i) for i in range(data_points)]  # ***
    fig, axs = plt.subplots(3)
    axs[0].plot(time_scale, quarantined_fit, "o", markersize=2, color='g', label="Quarantined_fit")  # ***
    axs[0].plot(time_scale, quarantined_average, "o", markersize=2, color='b', label="Quarantined_average")  # ***
    axs[0].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Cases")
    axs[0].legend(loc='upper left')  # ***
    #axs[1].plot(susceptible_fit, "o", markersize=2, color='g')
    #axs[1].plot(vaccinated_fit, "o", markersize=2, color='g')
    #axs[1].plot(exposed_zero_fit, "o", markersize=2, color='g')
    axs[1].plot(time_scale, exposed_vac_fit, "o", markersize=2, color='g', label="exposed_vac_fit")    # ***
    #axs[1].plot(exposed_zero_fit+exposed_vac_fit, "o", markersize=2, color='g') ?
    #axs[1].plot(infected_zero_fit, "o", markersize=2, color='g')
    #axs[1].plot(infected_vac_fit, "o", markersize=2, color='g') ?
    #axs[1].plot(infected_zero_fit+infected_vac_fit, "o", markersize=2, color='g')
    #axs[1].plot(quarantined_zero_fit, "o", markersize=2, color='g')
    #axs[1].plot(quarantined_vac_fit, "o", markersize=2, color='g') ?
    #axs[1].plot(quarantined_zero_fit+quarantined_vac_fit, "o", markersize=2, color='g')
    #axs[1].plot(recovered_zero_fit, "o", markersize=2, color='g')
    #axs[1].plot(recovered_vac_fit, "o", markersize=2, color='g')
    #axs[1].plot(recovered_zero_fit+recovered_zero_fit, "o", markersize=2, color='g')
    axs[1].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Cases")
    axs[1].legend(loc='upper left')  # ***
    #
    axs[2].plot(time_scale, beta_zero_ones, "-", linewidth=2, color='r', label="beta_zero_ones")  # ***
    axs[2].plot(time_scale, beta_zero_twos, "-", linewidth=2, color='g', label="beta_zero_twos")  # ***
    axs[2].plot(time_scale, beta_vac_ones, "-", linewidth=2, color='b', label="beta_vac_ones")  # ***
    axs[2].plot(time_scale, beta_vac_twos, "-", linewidth=2, color='purple', label="beta_vac_twos")  # ***
    axs[2].set_xlim(datetime.date(starting_time), datetime.date(ending_time))  # ***
    axs[2].set_xlabel("Days")
    axs[2].set_ylabel("Transmission rates")
    axs[2].legend(loc='upper left')  # ***

    plt.show()
