import numpy as np
from scipy import optimize
import string
import matplotlib.pyplot as plt
import pandas as pd

global HuCount
global YDATA
HuCount = 0

def SVEIIQRD(init_susceptible, init_vaccinated, init_exposed_zero, init_exposed_vac, init_infected_zero, init_infected_vac, init_quarantined_zero,
             init_quarantined_vac, init_recovered_zero, init_recovered_vac, init_death_zero, init_death_vac, pi, alpha,
             beta_zero_one, beta_zero_two, beta_vac_one, beta_vac_two, gamma, delta, lam_zero, lam_vac,
             phi_zero, phi_vac, pop):

    """
    implement deterministic version of SVEIIQRD
    :parameter:
    --------
    init_susceptible: number
        initial value of susceptible

    init_vaccinated: number
        initial value of vaccinated

    init_exposed: number
        initial value of exposed with unvaccinated

    init_exposed_vac: number
        initial value of exposed with vaccinated

    init_infected: number
        initial value of infected with unvaccinated

    init_infected_vac: number
        initial value of infected with vaccination

    init_quarantined: number
        initial value of reported cases with unvaccinated

    init_quarantined_vac: number
        initial value of reported cases with vaccination

    init_recovered: number
        initial value of recovered cases without vaccinated

    init_recovered_vac: number
        initial value of recovered cases with vaccinated

    init_death: number
        initial value of death cases without vaccinated

    init_death_vac: number
        initial value of death cases with vaccinated

    pi: string
        variant of COVID-19 virus

    alpha: array
        daily 2_doses vaccination rate (daily)

    beta_zero_one: array
        transmission rate of COVID-19 between susceptible and infected with unvaccinated

    beta_zero_two: array
        transmission rate of COVID-19 between susceptible and infected with vaccinated

    beta_vac_one: array
        transmission rate of COVID-19 between vaccinated and infected with unvaccinated

    beta_vac_two: array
        transmission rate of COVID-19 between vaccinated and infected with vaccinated

    gamma: number
        transition rate of COVID-19 between exposed and infected with unvaccinated


    gamma_vac: number (optional)
        transition rate of COVID-19 between exposed and infected with vaccinated

    delta: number
        confirmation rate of COVID-19 between infected and confirmed cases without vaccinated

    delta_vac: number (optional)
        confirmation rate of COVID-19 between infected and confirmed cases with vaccinated

    lamb: number
        the recovery rate of COVID-19 without vaccinated

    lamb_vac: number
        the recovery rate of COVID-19 with vaccinated

    phi: number
        the death rate of COVID-19 without vaccinated

    phi_vac: number
        the death rate of COVID-19 with vaccinated

    pop: number
        the total population of a specific zone or contry

    Notes: betas is dependent on variants, vaccination, and NPIs


    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed: array
        the number of exposed without vaccinated (active)

    exposed_vac: array
        the number of exposed with vaccinated (active)

    infected: array
        the number of infected with unvaccinated (active)

    infected_vac: array
        the number of infected with vaccinated  (active)

    quarantined: array
        the number of quarantined without vaccinated (active)

    quarantined_vac: array
        the number of quarantined with vaccinated (active)

    recovered: array
        the accumulated number of recovered without vaccinated (accumulated)

    recovered_vac: array
        the accumulated number of recovered with vaccinated (accumulated)

    death: array
        the accumulated number of death without vaccinated (accumulated)

    death_vac: array
        the accumulated number of death with vaccinated (accumulated)

    Notes:

    """

    data_points = len(alpha) + 1
    susceptible = np.zeros(data_points)
    vaccinated = np.zeros(data_points)
    exposed_zero = np.zeros(data_points)
    exposed_vac = np.zeros(data_points)
    total_exposed = np.zeros(data_points)
    infected_zero = np.zeros(data_points)
    infected_vac = np.zeros(data_points)
    total_infected = np.zeros(data_points)
    quarantined_zero = np.zeros(data_points)
    quarantined_vac = np.zeros(data_points)
    total_quarantined = np.zeros(data_points)
    recovered_zero = np.zeros(data_points)
    recovered_vac = np.zeros(data_points)
    total_recovered = np.zeros(data_points)
    death_zero = np.zeros(data_points)
    death_vac = np.zeros(data_points)
    total_death = np.zeros(data_points)

    # Initialization
    susceptible[0] = init_susceptible
    vaccinated[0] = init_vaccinated
    exposed_zero[0] = init_exposed_zero
    exposed_vac[0] = init_exposed_vac
    # total_exposed[0] = init_exposed_zero + init_exposed_vac
    infected_zero[0] = init_infected_zero
    infected_vac[0] = init_infected_vac
    # total_infected[0] = init_infected_zero + init_infected_vac
    quarantined_zero[0] = init_quarantined_zero
    quarantined_vac[0] = init_quarantined_vac
    # total_quarantined[0] = init_quarantined_zero + init_quarantined_vac
    recovered_zero[0] = init_recovered_zero
    recovered_vac[0] = init_recovered_vac
    # total_recovered[0] = init_recovered_zero + init_recovered_vac
    death_zero[0] = init_death_zero
    death_vac[0] = init_death_vac
    # total_death[0] = init_death_zero + init_death_vac

    for ii in range(1, data_points):
        # susceptible
        susceptible[ii] = susceptible[ii-1] + -1 * beta_zero_one[ii-1] * susceptible[ii-1] * infected_zero[ii-1] / pop + \
            -1 * beta_zero_two[ii-1] * susceptible[ii-1] * infected_vac[ii-1] / pop + -1 * alpha[ii-1] * pop
        if susceptible[ii] < 0:
            break
        # vaccinated
        vaccinated[ii] = vaccinated[ii-1] + -1 * beta_vac_one[ii-1] * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
            -1 * beta_vac_two[ii-1] * vaccinated[ii-1] * infected_vac[ii-1] / pop + alpha[ii-1] * pop
        if vaccinated[ii] < 0:
            vaccinated[ii] = 0
        # exposed_zero
        exposed_zero[ii] = exposed_zero[ii-1] + beta_zero_one[ii-1] * susceptible[ii-1] * infected_zero[ii-1] / pop + \
            beta_zero_two[ii-1] * susceptible[ii-1] * infected_vac[ii-1] / pop + -1 * gamma * exposed_zero[ii-1]
        if exposed_zero[ii] < 0:
            exposed_zero[ii] = 0
        # exposed_vac
        exposed_vac[ii] = exposed_vac[ii-1] + beta_vac_one[ii-1] * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
            beta_vac_two[ii-1] * vaccinated[ii-1] * infected_vac[ii-1] / pop + -1 * gamma * exposed_vac[ii - 1]
        if exposed_vac[ii] < 0:
            exposed_vac[ii] = 0
        # infected_zero
        infected_zero[ii] = infected_zero[ii - 1] + gamma * exposed_zero[ii-1] + -1 * delta * infected_zero[ii-1]
        if infected_zero[ii] < 0:
            infected_zero[ii] = 0
        # infected_vac
        infected_vac[ii] = infected_vac[ii - 1] + gamma * exposed_vac[ii - 1] + -1 * delta * infected_vac[ii - 1]
        if infected_vac[ii] < 0:
            infected_vac[ii] = 0
        # quarantined_zero
        quarantined_zero[ii] = quarantined_zero[ii-1] + delta * infected_zero[ii-1] + -1 * lam_zero * quarantined_zero[ii-1] + \
            -1 * phi_zero * quarantined_zero[ii-1]
        if quarantined_zero[ii] < 0:
            quarantined_zero[ii] = 0
        # quarantined_vac
        quarantined_vac[ii] = quarantined_vac[ii - 1] + delta * infected_vac[ii - 1] + \
            -1 * lam_vac * quarantined_vac[ii - 1] + -1 * phi_vac * quarantined_vac[ii - 1]
        if quarantined_vac[ii] < 0:
            quarantined_vac[ii] = 0
        # recovered_zero
        recovered_zero[ii] = recovered_zero[ii-1] + lam_zero * quarantined_zero[ii-1]
        if recovered_zero[ii] < 0:
            recovered_zero[ii] = 0
        # recovered_vac
        recovered_vac[ii] = recovered_vac[ii - 1] + lam_vac * quarantined_vac[ii - 1]
        if recovered_vac[ii] < 0:
            recovered_vac[ii] = 0
        # death_zero
        death_zero[ii] = death_zero[ii-1] + phi_zero * quarantined_zero[ii-1]
        if death_zero[ii] < 0:
            death_zero[ii] = 0
        # death_vac
        death_vac[ii] = death_vac[ii - 1] + phi_vac * quarantined_vac[ii - 1]
        if death_vac[ii] < 0:
            death_vac[ii] = 0

    """
    total_exposed = exposed + exposed_vac
    total_infected[ii] = infected[ii] + infected_vac[ii]
    total_quarantined[ii] = quarantined[ii] + quarantined_vac[ii]
    total_recovered[ii] = recovered[ii] + recovered_vac[ii]
    total_death[ii] = death[ii] + death_vac[ii]
    """

    return np.concatenate((susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
                           quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac))


def SVEIIQRDFit(x_data, *args):

    """
    implement deterministic version of SVEIIQRD
    :parameter:
    --------
    susceptible: array
        values of susceptible

    vaccinated: array
        values of vaccinated

    exposed_zero: array
        values of exposed with unvaccinated

    exposed_vac: array
        values of exposed with vaccinated

    infected_zero: array
        values of infected with unvaccinated

    infected_vac: array
        values of infected with vaccination

    quarantined: array
        values of reported cases with unvaccinated

    quarantined_vac: array
        values of reported cases with vaccination

    recovered_zero: array
        values of recovered cases without vaccinated

    recovered_vac: array
        values of recovered cases with vaccinated

    death_zero: array
        values of death cases without vaccinated

    death_vac: array
        values of death cases with vaccinated

    pi: string
        variant of COVID-19 virus

    alpha: array
        daily 2_doses vaccination rate (daily)

    init_beta_zero_one: number
        initial transmission rate of COVID-19 between susceptible and infected with unvaccinated

    init_NPIs_eff_zero_ones: array
        initial transmission rates of COVID-19 between susceptible and infected with unvaccinated caused by NPIs

    init_beta_zero_two: number
        initial transmission rate of COVID-19 between susceptible and infected with vaccinated

    init_NPIs_eff_zero_twos: array
        initial transmission rates of COVID-19 between susceptible and infected with vaccinated caused by NPIs

    init_beta_vac_one: number
        initial transmission rate of COVID-19 between vaccinated and infected with unvaccinated

    init_NPIs_eff_vac_ones: array
        initial transmission rates of COVID-19 between vaccinated and infected with unvaccinated caused by NPIs

    init_beta_vac_two: number
        initial transmission rate of COVID-19 between vaccinated and infected with vaccinated

    init_NPIs_eff_vac_twos: array
        initial transmission rates of COVID-19 between vaccinated and infected with vaccinated caused by NPIs

    init_gamma: number
        transition rate of COVID-19 between exposed and infected with unvaccinated

    init_gamma_vac: number (optional)
        transition rate of COVID-19 between exposed and infected with vaccinated

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases without vaccinated

    init_delta_vac: number (optional)
        confirmation rate of COVID-19 between infected and confirmed cases with vaccinated

    lamb_zeros: array
        the recovery rates of COVID-19 without vaccinated

    lamb_vacs: array
        the recovery rates of COVID-19 with vaccinated

    phi_zeros: array
        the death rates of COVID-19 without vaccinated

    phi_vacs: array
        the death rates of COVID-19 with vaccinated

    pop: number
        the total population of a specific zone or contry

    Notes: betas is dependent on variants, vaccination, and NPIs


    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed: array
        the number of exposed without vaccinated (active)

    exposed_vac: array
        the number of exposed with vaccinated (active)

    infected: array
        the number of infected with unvaccinated (active)

    infected_vac: array
        the number of infected with vaccinated  (active)

    quarantined: array
        the number of quarantined without vaccinated (active)

    quarantined_vac: array
        the number of quarantined with vaccinated (active)

    recovered: array
        the accumulated number of recovered without vaccinated (accumulated)

    recovered_vac: array
        the accumulated number of recovered with vaccinated (accumulated)

    death: array
        the accumulated number of death without vaccinated (accumulated)

    death_vac: array
        the accumulated number of death with vaccinated (accumulated)

    Notes:

    """

    susceptible = x_data[0]
    vaccinated = x_data[1]
    exposed_zero = x_data[2]
    exposed_vac = x_data[3]
    infected_zero = x_data[4]
    infected_vac = x_data[5]
    quarantined_zero = x_data[6]
    quarantined_vac = x_data[7]
    recovered_zero = x_data[8]
    recovered_vac = x_data[9]
    death_zero = x_data[10]
    death_vac = x_data[11]
    alphas = x_data[12]
    lam_zeros = x_data[13]
    lam_vacs = x_data[14]
    phi_zeros = x_data[15]
    phi_vacs = x_data[16]
    NPIs_no_vac = x_data[17]
    NPIs_vac = x_data[18]
    pop = x_data[19]

    data_points = len(alphas)
    #susceptible = np.zeros(data_points)
    # vaccinated = np.zeros(data_points)
    #exposed_zero = np.zeros(data_points)
    #exposed_vac = np.zeros(data_points)
    #total_exposed = np.zeros(data_points)
    #infected_zero = np.zeros(data_points)
    #infected_vac = np.zeros(data_points)
    #total_infected = np.zeros(data_points)
    #quarantined_zero = np.zeros(data_points)
    #quarantined_vac = np.zeros(data_points)
    #total_quarantined = np.zeros(data_points)
    #recovered_zero = np.zeros(data_points)
    #recovered_vac = np.zeros(data_points)
    #total_recovered = np.zeros(data_points)
    #death_zero = np.zeros(data_points)
    #death_vac = np.zeros(data_points)
    #total_death = np.zeros(data_points)
    #print(args)
    #tt = args[0:4]
    #print(tt)
    #print(len(args))
    init_beta_zero_one = args[0]
    init_beta_zero_two = args[1]
    init_beta_vac_one = args[2]
    init_beta_vac_two = args[3]
    init_gamma = args[4]
    init_delta = args[5]
    init_NPIs_delay_days = args[6]
    init_NPIs_eff_zero_ones = args[7: 13]  # ***
    init_NPIs_eff_zero_twos = args[13: 22]  # ***
    init_NPIs_eff_vac_ones = args[22: 31]  # ***
    init_NPIs_eff_vac_twos = args[31: 40]  # ***
    #print(init_NPIs_eff_vac_twos)

    beta_zero_ones = np.zeros(data_points) + init_beta_zero_one
    beta_zero_twos = np.zeros(data_points) + init_beta_zero_two
    beta_vac_ones = np.zeros(data_points) + init_beta_vac_one
    beta_vac_twos = np.zeros(data_points) + init_beta_vac_two
    pop = pop[0]

    # Initialization
    # susceptible[0] = init_susceptible
    # vaccinated[0] = init_vaccinated
    #exposed_zero[0] = init_exposed_zero
    #exposed_vac[0] = init_exposed_vac
    # total_exposed[0] = init_exposed_zero + init_exposed_vac
    #infected_zero[0] = init_infected_zero
    #infected_vac[0] = init_infected_vac
    # total_infected[0] = init_infected_zero + init_infected_vac
    #quarantined_zero[0] = init_quarantined_zero
    #quarantined_vac[0] = init_quarantined_vac
    # total_quarantined[0] = init_quarantined_zero + init_quarantined_vac
    #recovered_zero[0] = init_recovered_zero
    #recovered_vac[0] = init_recovered_vac
    # total_recovered[0] = init_recovered_zero + init_recovered_vac
    #death_zero[0] = init_death_zero
    #death_vac[0] = init_death_vac
    # total_death[0] = init_death_zero + init_death_vac
    #print("beta_zero_ones", beta_zero_ones)
    tt = [0, 0, 0, 0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] != 0:
            # beta_zero_ones[ii+init_NPIs_delay_days:] = beta_zero_ones[ii] * (1 + init_NPIs_eff_zero_ones[tt[0]])
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            # beta_vac_twos[ii+init_NPIs_delay_days:] = beta_vac_twos[ii] * (1 + init_NPIs_eff_vac_twos[tt[3]])
            # print("ii + init_NPIs_delay_days: ", init_NPIs_eff_vac_twos)
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            # beta_zero_ones[ii+init_NPIs_delay_days:] = beta_zero_ones[ii] * (1 + init_NPIs_eff_zero_ones[tt[0]])
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            # beta_vac_twos[ii+init_NPIs_delay_days:] = beta_vac_twos[ii] * (1 + init_NPIs_eff_vac_twos[tt[3]])
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1

    #print("beta_zero_ones", beta_zero_ones)

    for ii in range(1, data_points):
        # susceptible
        susceptible[ii] = susceptible[ii-1] + -1 * beta_zero_ones[ii-1] * susceptible[ii-1] * infected_zero[ii-1] / pop + \
            -1 * beta_zero_twos[ii-1] * susceptible[ii-1] * infected_vac[ii-1] / pop + -1 * alphas[ii-1] * pop
        if susceptible[ii] < 0:
            print(ii, susceptible[ii])
            print("beta_zero_ones[ii-1]", beta_zero_ones[ii-1])
            print("susceptible[ii-1]", susceptible[ii-1])
            print("infected_zero[ii-1]", infected_zero[ii-1])
            print("beta_zero_twos[ii-1]", beta_zero_twos[ii-1])
            print("alphas[ii-1]", alphas[ii-1])
            print("pop", pop)
            break
        # vaccinated
        vaccinated[ii] = vaccinated[ii-1] + -1 * beta_vac_ones[ii-1] * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
            -1 * beta_vac_twos[ii-1] * vaccinated[ii-1] * infected_vac[ii-1] / pop + alphas[ii-1] * pop
        # exposed_zero
        exposed_zero[ii] = exposed_zero[ii-1] + beta_zero_ones[ii-1] * susceptible[ii-1] * infected_zero[ii-1] / pop + \
            beta_zero_twos[ii-1] * susceptible[ii-1] * infected_vac[ii-1] / pop + -1 * init_gamma * exposed_zero[ii-1]
        # exposed_vac
        exposed_vac[ii] = exposed_vac[ii-1] + beta_vac_ones[ii-1] * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
            beta_vac_twos[ii-1] * vaccinated[ii-1] * infected_vac[ii-1] / pop + -1 * init_gamma * exposed_vac[ii - 1]
        # infected_zero
        infected_zero[ii] = infected_zero[ii - 1] + init_gamma * exposed_zero[ii-1] + -1 * init_delta * infected_zero[ii-1]
        # infected_vac
        infected_vac[ii] = infected_vac[ii - 1] + init_gamma * exposed_vac[ii - 1] + -1 * init_delta * infected_vac[ii-1]
        # quarantined_zero
        quarantined_zero[ii] = quarantined_zero[ii-1] + init_delta * infected_zero[ii-1] + -1 * lam_zeros[ii-1] * quarantined_zero[ii-1] + \
            -1 * phi_zeros[ii-1] * quarantined_zero[ii-1]
        # quarantined_vac
        quarantined_vac[ii] = quarantined_vac[ii - 1] + init_delta * infected_vac[ii - 1] + \
            -1 * lam_vacs[ii-1] * quarantined_vac[ii - 1] + -1 * phi_vacs[ii-1] * quarantined_vac[ii - 1]
        # recovered_zero
        recovered_zero[ii] = recovered_zero[ii-1] + lam_zeros[ii-1] * quarantined_zero[ii-1]
        # recovered_vac
        recovered_vac[ii] = recovered_vac[ii - 1] + lam_vacs[ii-1] * quarantined_vac[ii-1]
        # death_zero
        death_zero[ii] = death_zero[ii-1] + phi_zeros[ii-1] * quarantined_zero[ii-1]
        # death_vac
        death_vac[ii] = death_vac[ii-1] + phi_vacs[ii-1] * quarantined_vac[ii-1]

    """
    total_exposed = exposed + exposed_vac
    total_infected[ii] = infected[ii] + infected_vac[ii]
    total_quarantined[ii] = quarantined[ii] + quarantined_vac[ii]
    total_recovered[ii] = recovered[ii] + recovered_vac[ii]
    total_death[ii] = death[ii] + death_vac[ii]
    """
    global HuCount
    HuCount = HuCount + 1

    if HuCount % 5000 == 0:
        print("HuCount: ", HuCount)

    with open("../iter.txt", "a+") as file_object:
        msd = 0.5 * sum((quarantined_zero+quarantined_vac - YDATA)**2)
        file_object.write(str(msd)+"\n")
    return quarantined_zero+quarantined_vac
    #return np.concatenate((susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
    #                       quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac))

def SVEIIQRDSimulation(x_data, *args):

    """
    implement deterministic version of SVEIIQRD
    :parameter:
    --------
    susceptible: array
        values of susceptible

    vaccinated: array
        values of vaccinated

    exposed_zero: array
        values of exposed with unvaccinated

    exposed_vac: array
        values of exposed with vaccinated

    infected_zero: array
        values of infected with unvaccinated

    infected_vac: array
        values of infected with vaccination

    quarantined: array
        values of reported cases with unvaccinated

    quarantined_vac: array
        values of reported cases with vaccination

    recovered_zero: array
        values of recovered cases without vaccinated

    recovered_vac: array
        values of recovered cases with vaccinated

    death_zero: array
        values of death cases without vaccinated

    death_vac: array
        values of death cases with vaccinated

    pi: string
        variant of COVID-19 virus

    alpha: array
        daily 2_doses vaccination rate (daily)

    init_beta_zero_one: number
        initial transmission rate of COVID-19 between susceptible and infected with unvaccinated

    init_NPIs_eff_zero_ones: array
        initial transmission rates of COVID-19 between susceptible and infected with unvaccinated caused by NPIs

    init_beta_zero_two: number
        initial transmission rate of COVID-19 between susceptible and infected with vaccinated

    init_NPIs_eff_zero_twos: array
        initial transmission rates of COVID-19 between susceptible and infected with vaccinated caused by NPIs

    init_beta_vac_one: number
        initial transmission rate of COVID-19 between vaccinated and infected with unvaccinated

    init_NPIs_eff_vac_ones: array
        initial transmission rates of COVID-19 between vaccinated and infected with unvaccinated caused by NPIs

    init_beta_vac_two: number
        initial transmission rate of COVID-19 between vaccinated and infected with vaccinated

    init_NPIs_eff_vac_twos: array
        initial transmission rates of COVID-19 between vaccinated and infected with vaccinated caused by NPIs

    init_gamma: number
        transition rate of COVID-19 between exposed and infected with unvaccinated

    init_gamma_vac: number (optional)
        transition rate of COVID-19 between exposed and infected with vaccinated

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases without vaccinated

    init_delta_vac: number (optional)
        confirmation rate of COVID-19 between infected and confirmed cases with vaccinated

    lamb_zeros: array
        the recovery rates of COVID-19 without vaccinated

    lamb_vacs: array
        the recovery rates of COVID-19 with vaccinated

    phi_zeros: array
        the death rates of COVID-19 without vaccinated

    phi_vacs: array
        the death rates of COVID-19 with vaccinated

    pop: number
        the total population of a specific zone or contry

    Notes: betas is dependent on variants, vaccination, and NPIs


    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed: array
        the number of exposed without vaccinated (active)

    exposed_vac: array
        the number of exposed with vaccinated (active)

    infected: array
        the number of infected with unvaccinated (active)

    infected_vac: array
        the number of infected with vaccinated  (active)

    quarantined: array
        the number of quarantined without vaccinated (active)

    quarantined_vac: array
        the number of quarantined with vaccinated (active)

    recovered: array
        the accumulated number of recovered without vaccinated (accumulated)

    recovered_vac: array
        the accumulated number of recovered with vaccinated (accumulated)

    death: array
        the accumulated number of death without vaccinated (accumulated)

    death_vac: array
        the accumulated number of death with vaccinated (accumulated)

    Notes:

    """

    susceptible = x_data[0]
    vaccinated = x_data[1]
    exposed_zero = x_data[2]
    exposed_vac = x_data[3]
    infected_zero = x_data[4]
    infected_vac = x_data[5]
    quarantined_zero = x_data[6]
    quarantined_vac = x_data[7]
    recovered_zero = x_data[8]
    recovered_vac = x_data[9]
    death_zero = x_data[10]
    death_vac = x_data[11]
    alphas = x_data[12]
    lam_zeros = x_data[13]
    lam_vacs = x_data[14]
    phi_zeros = x_data[15]
    phi_vacs = x_data[16]
    NPIs_no_vac = x_data[17]
    NPIs_vac = x_data[18]
    pop = x_data[19]

    data_points = len(alphas)
    #susceptible = np.zeros(data_points)
    # vaccinated = np.zeros(data_points)
    #exposed_zero = np.zeros(data_points)
    #exposed_vac = np.zeros(data_points)
    #total_exposed = np.zeros(data_points)
    #infected_zero = np.zeros(data_points)
    #infected_vac = np.zeros(data_points)
    #total_infected = np.zeros(data_points)
    #quarantined_zero = np.zeros(data_points)
    #quarantined_vac = np.zeros(data_points)
    #total_quarantined = np.zeros(data_points)
    #recovered_zero = np.zeros(data_points)
    #recovered_vac = np.zeros(data_points)
    #total_recovered = np.zeros(data_points)
    #death_zero = np.zeros(data_points)
    #death_vac = np.zeros(data_points)
    #total_death = np.zeros(data_points)
    #print(args)
    #tt = args[0:4]
    #print(tt)
    #print(len(args))
    init_beta_zero_one = args[0]
    init_beta_zero_two = args[1]
    init_beta_vac_one = args[2]
    init_beta_vac_two = args[3]
    init_gamma = args[4]
    init_delta = args[5]
    init_NPIs_delay_days = args[6]
    init_NPIs_eff_zero_ones = args[7: 13]  # ***
    init_NPIs_eff_zero_twos = args[13: 22] # ***
    init_NPIs_eff_vac_ones = args[22: 31]  # ***
    init_NPIs_eff_vac_twos = args[31: 40]  # ***
    #print(init_NPIs_eff_vac_twos)

    beta_zero_ones = np.zeros(data_points) + init_beta_zero_one
    beta_zero_twos = np.zeros(data_points) + init_beta_zero_two
    beta_vac_ones = np.zeros(data_points) + init_beta_vac_one
    beta_vac_twos = np.zeros(data_points) + init_beta_vac_two
    pop = pop[0]

    # Initialization
    # susceptible[0] = init_susceptible
    # vaccinated[0] = init_vaccinated
    #exposed_zero[0] = init_exposed_zero
    #exposed_vac[0] = init_exposed_vac
    # total_exposed[0] = init_exposed_zero + init_exposed_vac
    #infected_zero[0] = init_infected_zero
    #infected_vac[0] = init_infected_vac
    # total_infected[0] = init_infected_zero + init_infected_vac
    #quarantined_zero[0] = init_quarantined_zero
    #quarantined_vac[0] = init_quarantined_vac
    # total_quarantined[0] = init_quarantined_zero + init_quarantined_vac
    #recovered_zero[0] = init_recovered_zero
    #recovered_vac[0] = init_recovered_vac
    # total_recovered[0] = init_recovered_zero + init_recovered_vac
    #death_zero[0] = init_death_zero
    #death_vac[0] = init_death_vac
    # total_death[0] = init_death_zero + init_death_vac

    tt = [0, 0, 0, 0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] != 0:
            # beta_zero_ones[ii+init_NPIs_delay_days:] = beta_zero_ones[ii] * (1 + init_NPIs_eff_zero_ones[tt[0]])
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            # beta_vac_twos[ii+init_NPIs_delay_days:] = beta_vac_twos[ii] * (1 + init_NPIs_eff_vac_twos[tt[3]])
            # print("ii + init_NPIs_delay_days: ", init_NPIs_eff_vac_twos)
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            # beta_zero_ones[ii+init_NPIs_delay_days:] = beta_zero_ones[ii] * (1 + init_NPIs_eff_zero_ones[tt[0]])
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            # beta_vac_twos[ii+init_NPIs_delay_days:] = beta_vac_twos[ii] * (1 + init_NPIs_eff_vac_twos[tt[3]])
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1

    for ii in range(1, data_points):
        # susceptible
        susceptible[ii] = susceptible[ii-1] + -1 * beta_zero_ones[ii-1] * susceptible[ii-1] * infected_zero[ii-1] / pop + \
            -1 * beta_zero_twos[ii-1] * susceptible[ii-1] * infected_vac[ii-1] / pop + -1 * alphas[ii-1] * pop
        # vaccinated
        vaccinated[ii] = vaccinated[ii-1] + -1 * beta_vac_ones[ii-1] * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
            -1 * beta_vac_twos[ii-1] * vaccinated[ii-1] * infected_vac[ii-1] / pop + alphas[ii-1] * pop
        # exposed_zero
        exposed_zero[ii] = exposed_zero[ii-1] + beta_zero_ones[ii-1] * susceptible[ii-1] * infected_zero[ii-1] / pop + \
            beta_zero_twos[ii-1] * susceptible[ii-1] * infected_vac[ii-1] / pop + -1 * init_gamma * exposed_zero[ii-1]
        # exposed_vac
        exposed_vac[ii] = exposed_vac[ii-1] + beta_vac_ones[ii-1] * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
            beta_vac_twos[ii-1] * vaccinated[ii-1] * infected_vac[ii-1] / pop + -1 * init_gamma * exposed_vac[ii - 1]
        # infected_zero
        infected_zero[ii] = infected_zero[ii - 1] + init_gamma * exposed_zero[ii-1] + -1 * init_delta * infected_zero[ii-1]
        # infected_vac
        infected_vac[ii] = infected_vac[ii - 1] + init_gamma * exposed_vac[ii - 1] + -1 * init_delta * infected_vac[ii - 1]
        # quarantined_zero
        quarantined_zero[ii] = quarantined_zero[ii-1] + init_delta * infected_zero[ii-1] + -1 * lam_zeros[ii-1] * quarantined_zero[ii-1] + \
            -1 * phi_zeros[ii-1] * quarantined_zero[ii-1]
        # quarantined_vac
        quarantined_vac[ii] = quarantined_vac[ii - 1] + init_delta * infected_vac[ii - 1] + \
            -1 * lam_vacs[ii-1] * quarantined_vac[ii - 1] + -1 * phi_vacs[ii-1] * quarantined_vac[ii - 1]
        # recovered_zero
        recovered_zero[ii] = recovered_zero[ii-1] + lam_zeros[ii-1] * quarantined_zero[ii-1]
        # recovered_vac
        recovered_vac[ii] = recovered_vac[ii - 1] + lam_vacs[ii-1] * quarantined_vac[ii - 1]
        # death_zero
        death_zero[ii] = death_zero[ii-1] + phi_zeros[ii-1] * quarantined_zero[ii-1]
        # death_vac
        death_vac[ii] = death_vac[ii - 1] + phi_vacs[ii-1] * quarantined_vac[ii - 1]

    """
    total_exposed = exposed + exposed_vac
    total_infected[ii] = infected[ii] + infected_vac[ii]
    total_quarantined[ii] = quarantined[ii] + quarantined_vac[ii]
    total_recovered[ii] = recovered[ii] + recovered_vac[ii]
    total_death[ii] = death[ii] + death_vac[ii]
    """
    return susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac, \
           quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac
    #return np.concatenate((susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
    #                       quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac))

def fit(init, x_data, y_data, bounds):
    x_data = x_data
    #self.ydata = ydata
    #time_full = range(data.shape[0])
    #print(f'x_data: {x_data}')
    global YDATA
    YDATA = y_data
    popt, pcov = optimize.curve_fit(SVEIIQRDFit, x_data, y_data, p0=init, bounds=bounds,
                                            maxfev=1e+7)
    return popt



def RecoveryRates(daily_recovered, infected):
    return np.divide(daily_recovered, infected)


def DeathRates(daily_deaths, infected):
    return np.divide(daily_deaths, infected)


def MAPECompute(x, y):
    return np.mean(np.abs((x - y))/y) * 100


def Betas(init_NPIs_delay_days, NPIs_no_vac, NPIs_vac, init_beta_zero_one, init_beta_zero_two, init_beta_vac_one,
          init_beta_vac_two, init_NPIs_eff_zero_ones, init_NPIs_eff_zero_twos, init_NPIs_eff_vac_ones,
          init_NPIs_eff_vac_twos):
    tt = [0, 0, 0, 0]
    data_points = len(NPIs_no_vac)
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    beta_zero_ones = np.zeros(data_points) + init_beta_zero_one
    beta_zero_twos = np.zeros(data_points) + init_beta_zero_two
    beta_vac_ones = np.zeros(data_points) + init_beta_vac_one
    beta_vac_twos = np.zeros(data_points) + init_beta_vac_two
    #print("init_NPIs_eff_zero_ones:", init_NPIs_eff_zero_ones)
    for ii in range(data_points):
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] != 0:
            # beta_zero_ones[ii+init_NPIs_delay_days:] = beta_zero_ones[ii] * (1 + init_NPIs_eff_zero_ones[tt[0]])
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            # beta_vac_twos[ii+init_NPIs_delay_days:] = beta_vac_twos[ii] * (1 + init_NPIs_eff_vac_twos[tt[3]])
            # print("ii + init_NPIs_delay_days: ", init_NPIs_eff_vac_twos)
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            # beta_zero_ones[ii+init_NPIs_delay_days:] = beta_zero_ones[ii] * (1 + init_NPIs_eff_zero_ones[tt[0]])
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            # beta_zero_twos[ii+init_NPIs_delay_days:] = beta_zero_twos[ii] * (1 + init_NPIs_eff_zero_twos[tt[1]])
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            # beta_vac_ones[ii+init_NPIs_delay_days:] = beta_vac_ones[ii] * (1 + init_NPIs_eff_vac_ones[tt[2]])
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            # beta_vac_twos[ii+init_NPIs_delay_days:] = beta_vac_twos[ii] * (1 + init_NPIs_eff_vac_twos[tt[3]])
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1

    #print("beta_zero_ones: ", beta_zero_ones)
    return beta_zero_ones, beta_zero_twos, beta_vac_ones, beta_vac_twos


def plot_optimizer(filename):
    #filename = "iter1.txt"
    data = pd.read_table(filename)
    data_length = len(data)
    idx_end = int(3e3)
    idx_start = int(5e+4)

    plt.plot(data[idx_start: idx_end])
    plt.xlabel("Iteration")
    plt.ylabel("MSD")
    plt.show()
    return 0


