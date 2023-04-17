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

    lam: number
        the recovery rate of COVID-19 without vaccinated

    lam_vac: number
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


def SEIQRD2Fit(x_data, *args):
    """
    implement deterministic version of SEIQRD^2
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

    alphas: array
        daily 2_doses vaccination rate (daily)

    lams_zero: array
        the recovery rates of COVID-19 without vaccinated

    lams_vac: array
        the recovery rates of COVID-19 with vaccinated

    phis_zero: array
        the death rates of COVID-19 without vaccinated

    phis_vac: array
        the death rates of COVID-19 with vaccinated

    NPIs_no_vac: array
        the list of NPIs for the unvaccinated

    NPIs_vac: array
        the list of NPIs for the vaccinated

    vaccine_eff: array
        the waning of vaccine in its effectiveness

    pop: number
        the total population of a specific zone or contry

    init_beta_zero_one: number
        initial transmission rate of COVID-19 between susceptible and infected with unvaccinated

    init_beta_zero_two: number
        initial transmission rate of COVID-19 between susceptible and infected with vaccinated

    init_beta_vac_one: number
        initial transmission rate of COVID-19 between vaccinated and infected with unvaccinated

    init_beta_vac_two: number
        initial transmission rate of COVID-19 between vaccinated and infected with vaccinated

    init_gamma: number
        transition rate of COVID-19 between exposed and infected with unvaccinated

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases without vaccinated

    init_NPIs_delay_days: number
        the delay of NPIs in effect

    init_NPIs_eff_zero_one: array
        initial transmission rates of COVID-19 between susceptible and infected with unvaccinated caused by NPIs

    init_NPIs_eff_zero_two: array
        initial transmission rates of COVID-19 between susceptible and infected with vaccinated caused by NPIs

    init_NPIs_eff_vac_one: array
        initial transmission rates of COVID-19 between vaccinated and infected with unvaccinated caused by NPIs

    init_NPIs_eff_vac_two: array
        initial transmission rates of COVID-19 between vaccinated and infected with vaccinated caused by NPIs

    Notes: betas is dependent on variants, vaccination, and NPIs

    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed_zero: array
        the number of exposed without vaccinated (active)

    exposed_vac: array
        the number of exposed with vaccinated (active)

    infected_zero: array
        the number of infected with unvaccinated (active)

    infected_vac: array
        the number of infected with vaccinated  (active)

    quarantined_zero: array
        the number of quarantined without vaccinated (active)

    quarantined_vac: array
        the number of quarantined with vaccinated (active)

    recovered_zero: array
        the accumulated number of recovered without vaccinated (accumulated)

    recovered_vac: array
        the accumulated number of recovered with vaccinated (accumulated)

    death_zero: array
        the accumulated number of death without vaccinated (accumulated)

    death_vac: array
        the accumulated number of death with vaccinated (accumulated)

    Notes:

    """
    # define variables
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
    lams_zero = x_data[13]
    lams_vac = x_data[14]
    phis_zero = x_data[15]
    phis_vac = x_data[16]
    NPIs_no_vac = x_data[17]
    NPIs_vac = x_data[18]
    vaccine_eff = x_data[19]
    pop = x_data[20]
    data_points = len(alphas)
    # set initial values of parameters
    init_beta_zero_one = args[0]
    init_beta_zero_two = args[1]
    init_beta_vac_one = args[2]
    init_beta_vac_two = args[3]
    init_gamma = args[4]
    init_delta = args[5]
    init_NPIs_delay_days = args[6]
    init_NPIs_eff_zero_one = args[7: 11]  # ***
    init_NPIs_eff_zero_two = args[11: 16]  # ***
    init_NPIs_eff_vac_one = args[16: 21]  # ***
    init_NPIs_eff_vac_two = args[21: 22]  # ***
    # print(init_NPIs_eff_vac_two)
    betas_zero_one = np.zeros(data_points) + init_beta_zero_one
    betas_zero_two = np.zeros(data_points) + init_beta_zero_two
    betas_vac_one = np.zeros(data_points) + init_beta_vac_one
    betas_vac_two = np.zeros(data_points) + init_beta_vac_two
    pop = pop[0]
    tt = [0, 0, 0, 0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        # establish NPIs as change points
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] != 0:
            # betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii] + init_NPIs_eff_zero_one[tt[0]]
            betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_one[tt[0]]
            tt[0] = tt[0] + 1
            # betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii] + init_NPIs_eff_zero_two[tt[1]]
            betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii] + init_NPIs_eff_vac_one[tt[2]]
            betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii + init_NPIs_delay_days] + \
                                                        init_NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
            # betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii] + init_NPIs_eff_vac_two[tt[3]]
            betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_two[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            # betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii] + init_NPIs_eff_zero_one[tt[0]]
            betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_one[tt[0]]
            tt[0] = tt[0] + 1
            # betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii] + init_NPIs_eff_zero_two[tt[1]]
            betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii + init_NPIs_delay_days -1 ] + \
                                                         init_NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii] + init_NPIs_eff_vac_one[tt[2]]
            betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            # betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii] + init_NPIs_eff_zero_two[tt[1]]
            betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii] + init_NPIs_eff_vac_one[tt[2]]
            betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
            # betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii] + init_NPIs_eff_vac_two[tt[3]]
            betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_two[tt[3]]
            tt[3] = tt[3] + 1

        # rule out negative values
        if betas_zero_one[ii] <= 0:
            betas_zero_one[ii] = 0
        if betas_zero_two[ii] <= 0:
            betas_zero_two[ii] = 0
        if betas_vac_one[ii] <= 0:
            betas_vac_one[ii] = 0
        if betas_vac_two[ii] <= 0:
            betas_vac_two[ii] = 0



    for ii in range(1, data_points):
        # compute Ordinary Differential Equations of SEIQRD^2
        # susceptible
        susceptible[ii] = susceptible[ii - 1] + \
                          -1 * betas_zero_one[ii - 1] * susceptible[ii - 1] * infected_zero[ii - 1] / pop + \
                          - 1 * betas_zero_two[ii - 1] * susceptible[ii - 1] * infected_vac[ii - 1] / pop + \
                          -1 * alphas[ii - 1] * pop
        # if susceptible[ii] < 0:
        #     print(ii, susceptible[ii])
        #     print("betas_zero_one[ii-1]", betas_zero_one[ii - 1])
        #     print("susceptible[ii-1]", susceptible[ii - 1])
        #     print("infected_zero[ii-1]", infected_zero[ii - 1])
        #     print("betas_zero_two[ii-1]", betas_zero_two[ii - 1])
        #     print("alphas[ii-1]", alphas[ii - 1])
        #     print("pop", pop)
        #     break
        # # vaccinated
        vaccinated[ii] = vaccinated[ii - 1] - \
                         betas_vac_one[ii-1] * (1-vaccine_eff[ii - 1]) * vaccinated[ii-1] * infected_zero[ii - 1] / pop\
                         - betas_vac_two[ii-1] * (1-vaccine_eff[ii - 1]) * vaccinated[ii-1] * infected_vac[ii-1] / pop \
                         + alphas[ii - 1] * pop
        # exposed_zero
        exposed_zero[ii] = exposed_zero[ii - 1] + \
                           betas_zero_one[ii - 1] * susceptible[ii - 1] * infected_zero[ii - 1] / pop + \
                           betas_zero_two[ii - 1] * susceptible[ii - 1] * infected_vac[ii - 1] / pop + \
                           -1 * init_gamma * exposed_zero[ii - 1]
        # exposed_vac
        exposed_vac[ii] = exposed_vac[ii - 1] + \
                          betas_vac_one[ii-1] * (1-vaccine_eff[ii-1]) * vaccinated[ii-1] * infected_zero[ii - 1] / pop \
                          + betas_vac_two[ii-1] * (1-vaccine_eff[ii-1]) * vaccinated[ii-1] * infected_vac[ii-1] / pop \
                          - init_gamma * exposed_vac[ii - 1]
        # infected_zero
        infected_zero[ii] = infected_zero[ii - 1] + init_gamma * exposed_zero[ii - 1] + \
                            -1 * init_delta * infected_zero[ii - 1]
        # infected_vac
        infected_vac[ii] = infected_vac[ii - 1] + init_gamma * exposed_vac[ii - 1] + \
                           -1 * init_delta * infected_vac[ii - 1]
        # quarantined_zero
        quarantined_zero[ii] = quarantined_zero[ii - 1] + init_delta * infected_zero[ii - 1] + \
                               -1 * lams_zero[ii - 1] * quarantined_zero[ii - 1] + \
                               -1 * phis_zero[ii - 1] * quarantined_zero[ii - 1]
        # quarantined_vac
        quarantined_vac[ii] = quarantined_vac[ii - 1] + init_delta * infected_vac[ii - 1] + \
                              -1 * lams_vac[ii - 1] * quarantined_vac[ii - 1] + \
                              -1 * phis_vac[ii - 1] * quarantined_vac[ii - 1]
        # recovered_zero
        recovered_zero[ii] = recovered_zero[ii - 1] + lams_zero[ii - 1] * quarantined_zero[ii - 1]
        # recovered_vac
        recovered_vac[ii] = recovered_vac[ii - 1] + lams_vac[ii - 1] * quarantined_vac[ii - 1]
        # death_zero
        death_zero[ii] = death_zero[ii - 1] + phis_zero[ii - 1] * quarantined_zero[ii - 1]
        # death_vac
        death_vac[ii] = death_vac[ii - 1] + phis_vac[ii - 1] * quarantined_vac[ii - 1]

    global HuCount
    HuCount = HuCount + 1

    if HuCount % 5000 == 0:
        print("HuCount: ", HuCount)

    with open("iter.txt", "a+") as file_object:
        msd = 0.5 * sum((quarantined_zero + quarantined_vac - YDATA) ** 2)
        file_object.write(str(msd) + "\n")
    return quarantined_zero + quarantined_vac

def SEIQRD2Simulation(x_data, *args):
    """
    implement deterministic version of SEIQRD^2
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

    alphas: array
        daily 2_doses vaccination rate (daily)

    lams_zero: array
        the recovery rates of COVID-19 without vaccinated

    lams_vac: array
        the recovery rates of COVID-19 with vaccinated

    phis_zero: array
        the death rates of COVID-19 without vaccinated

    phis_vac: array
        the death rates of COVID-19 with vaccinated

    NPIs_no_vac: array
        the list of NPIs for the unvaccinated

    NPIs_vac: array
        the list of NPIs for the vaccinated

    vaccine_eff: array
        the waning of vaccine in its effectiveness against infection

    pop: number
        the total population of a specific zone or contry

    init_beta_zero_one: number
        initial transmission rate of COVID-19 between susceptible and infected with unvaccinated

    init_beta_zero_two: number
        initial transmission rate of COVID-19 between susceptible and infected with vaccinated

    init_beta_vac_one: number
        initial transmission rate of COVID-19 between vaccinated and infected with unvaccinated

    init_beta_vac_two: number
        initial transmission rate of COVID-19 between vaccinated and infected with vaccinated

    init_gamma: number
        transition rate of COVID-19 between exposed and infected with unvaccinated

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases without vaccinated

    init_NPIs_delay_days: number
        the delay of NPIs in effect

    init_NPIs_eff_zero_one: array
        initial transmission rates of COVID-19 between susceptible and infected with unvaccinated caused by NPIs

    init_NPIs_eff_zero_two: array
        initial transmission rates of COVID-19 between susceptible and infected with vaccinated caused by NPIs

    init_NPIs_eff_vac_one: array
        initial transmission rates of COVID-19 between vaccinated and infected with unvaccinated caused by NPIs

    init_NPIs_eff_vac_two: array
        initial transmission rates of COVID-19 between vaccinated and infected with vaccinated caused by NPIs

    Notes: betas is dependent on variants, vaccination, and NPIs


    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed_zero: array
        the number of exposed without vaccinated (active)

    exposed_vac: array
        the number of exposed with vaccinated (active)

    infected_zero: array
        the number of infected with unvaccinated (active)

    infected_vac: array
        the number of infected with vaccinated  (active)

    quarantined_zero: array
        the number of quarantined without vaccinated (active)

    quarantined_vac: array
        the number of quarantined with vaccinated (active)

    recovered_zero: array
        the accumulated number of recovered without vaccinated (accumulated)

    recovered_vac: array
        the accumulated number of recovered with vaccinated (accumulated)

    death_zero: array
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
    lams_zero = x_data[13]
    lams_vac = x_data[14]
    phis_zero = x_data[15]
    phis_vac = x_data[16]
    NPIs_no_vac = x_data[17]
    NPIs_vac = x_data[18]
    vaccine_eff = x_data[19]
    pop = x_data[20]
    data_points = len(alphas)
    # values of optimal parameters
    init_beta_zero_one = args[0]
    init_beta_zero_two = args[1]
    init_beta_vac_one = args[2]
    init_beta_vac_two = args[3]
    init_gamma = args[4]
    init_delta = args[5]
    init_NPIs_delay_days = args[6]
    init_NPIs_eff_zero_one = args[7: 11]  # ***
    init_NPIs_eff_zero_two = args[11: 16]  # ***
    init_NPIs_eff_vac_one = args[16: 21]  # ***
    init_NPIs_eff_vac_two = args[21: 22]  # ***
    betas_zero_one = np.zeros(data_points) + init_beta_zero_one
    betas_zero_two = np.zeros(data_points) + init_beta_zero_two
    betas_vac_one = np.zeros(data_points) + init_beta_vac_one
    betas_vac_two = np.zeros(data_points) + init_beta_vac_two
    pop = pop[0]
    tt = [0, 0, 0, 0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        # establish NPIs as change points
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] != 0:
            # betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii] + init_NPIs_eff_zero_one[tt[0]]
            betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_one[tt[0]]
            tt[0] = tt[0] + 1
            # betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii] + init_NPIs_eff_zero_two[tt[1]]
            betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii] + init_NPIs_eff_vac_one[tt[2]]
            betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii + init_NPIs_delay_days] + \
                                                        init_NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
            # betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii] + init_NPIs_eff_vac_two[tt[3]]
            betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_two[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            # betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii] + init_NPIs_eff_zero_one[tt[0]]
            betas_zero_one[ii + init_NPIs_delay_days:] = betas_zero_one[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_one[tt[0]]
            tt[0] = tt[0] + 1
            # betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii] + init_NPIs_eff_zero_two[tt[1]]
            betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii] + init_NPIs_eff_vac_one[tt[2]]
            betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            # betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii] + init_NPIs_eff_zero_two[tt[1]]
            betas_zero_two[ii + init_NPIs_delay_days:] = betas_zero_two[ii + init_NPIs_delay_days - 1] + \
                                                         init_NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii] + init_NPIs_eff_vac_one[tt[2]]
            betas_vac_one[ii + init_NPIs_delay_days:] = betas_vac_one[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
            # betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii] + init_NPIs_eff_vac_two[tt[3]]
            betas_vac_two[ii + init_NPIs_delay_days:] = betas_vac_two[ii + init_NPIs_delay_days - 1] + \
                                                        init_NPIs_eff_vac_two[tt[3]]
            tt[3] = tt[3] + 1

        # rule out negative values
        if betas_zero_one[ii] <= 0:
            betas_zero_one[ii] = 0
        if betas_zero_two[ii] <= 0:
            betas_zero_two[ii] = 0
        if betas_vac_one[ii] <= 0:
            betas_vac_one[ii] = 0
        if betas_vac_two[ii] <= 0:
            betas_vac_two[ii] = 0

    for ii in range(1, data_points):
        # compute ODE of SEIQRD^2
        # susceptible
        susceptible[ii] = susceptible[ii - 1] + \
                          -1 * betas_zero_one[ii - 1] * susceptible[ii - 1] * infected_zero[ii - 1] / pop + \
                          -1 * betas_zero_two[ii - 1] * susceptible[ii - 1] * infected_vac[ii - 1] / pop + \
                          -1 * alphas[ii - 1] * pop
        # vaccinated
        vaccinated[ii] = vaccinated[ii - 1] - \
                         betas_vac_one[ii-1] * (1-vaccine_eff[ii - 1]) * vaccinated[ii-1] * infected_zero[ii-1] / pop \
                         - betas_vac_two[ii-1] * (1-vaccine_eff[ii-1]) * vaccinated[ii-1] * infected_vac[ii-1] / pop + \
                         alphas[ii - 1] * pop
        # exposed_zero
        exposed_zero[ii] = exposed_zero[ii - 1] + \
                           betas_zero_one[ii - 1] * susceptible[ii - 1] * infected_zero[ii - 1] / pop + \
                           betas_zero_two[ii - 1] * susceptible[ii - 1] * infected_vac[ii - 1] / pop + \
                           -1 * init_gamma * exposed_zero[ii - 1]
        # exposed_vac
        exposed_vac[ii] = exposed_vac[ii - 1] + \
                          betas_vac_one[ii-1] * (1-vaccine_eff[ii-1]) * vaccinated[ii-1] * infected_zero[ii-1] / pop + \
                          betas_vac_two[ii-1] * (1-vaccine_eff[ii-1]) * vaccinated[ii-1] * infected_vac[ii-1] / pop + \
                          -1 * init_gamma * exposed_vac[ii - 1]
        # infected_zero
        infected_zero[ii] = infected_zero[ii - 1] + init_gamma * exposed_zero[ii - 1] + \
                            -1 * init_delta * infected_zero[ii - 1]
        # infected_vac
        infected_vac[ii] = infected_vac[ii - 1] + init_gamma * exposed_vac[ii - 1] + \
                           -1 * init_delta * infected_vac[ii - 1]
        # quarantined_zero
        quarantined_zero[ii] = quarantined_zero[ii - 1] + init_delta * infected_zero[ii - 1] + \
                               -1 * lams_zero[ii - 1] * quarantined_zero[ii - 1] + \
                               -1 * phis_zero[ii - 1] * quarantined_zero[ii - 1]
        # quarantined_vac
        quarantined_vac[ii] = quarantined_vac[ii - 1] + init_delta * infected_vac[ii - 1] + \
                              -1 * lams_vac[ii - 1] * quarantined_vac[ii - 1] + \
                              -1 * phis_vac[ii - 1] * quarantined_vac[ii - 1]
        # recovered_zero
        recovered_zero[ii] = recovered_zero[ii - 1] + lams_zero[ii - 1] * quarantined_zero[ii - 1]
        # recovered_vac
        recovered_vac[ii] = recovered_vac[ii - 1] + lams_vac[ii - 1] * quarantined_vac[ii - 1]
        # death_zero
        death_zero[ii] = death_zero[ii - 1] + phis_zero[ii - 1] * quarantined_zero[ii - 1]
        # death_vac
        death_vac[ii] = death_vac[ii - 1] + phis_vac[ii - 1] * quarantined_vac[ii - 1]

    return susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac, \
           quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac

def fit(init, x_data, y_data, bounds):
    """
    :param init: is a set of initial values of parameters
                  init_beta_zero_one, init_beta_zero_two, init_beta_vac_one, init_beta_vac_two,
                  init_gamma, init_delta, init_NPIs_delay_days,
                  init_NPIs_eff_zero_one, init_NPIs_eff_zero_two,init_NPIs_eff_vac_one, init_NPIs_eff_vac_two
    :param x_data: is a set of hyper parameters
                    susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
                    quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac,
                    alphas, lams_zero, lams_vac, phis_zero, phis_vac, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    :param y_data: is a target
                    quarantined_average
    :param bounds: is the bounds of parameters
    :return: a set of parameters
    """
    x_data = x_data
    global YDATA
    YDATA = y_data
    # print("bounds: ", bounds)
    # print("init:", init)
    popt, pcov = optimize.curve_fit(SEIQRD2Fit, x_data, y_data, p0=init, bounds=bounds, maxfev=1e+7)
    return popt

def SPEIQRDFit(x_data, *args):
    """
    implement deterministic version of SPEIQRD: ablation study
    :parameter:
    --------
    susceptible: array
        values of susceptible

    protected: array
        values of vaccinated, considering the effectiveness

    exposed: array
        values of exposed

    infected: array
        values of infected

    quarantined: array
        values of reported cases

    recovered: array
        values of recovered cases

    death: array
        values of death cases

    pi: string
        variant of COVID-19 virus

    alphas: array
        daily protection rates

    lams: array
        the recovery rates of COVID-19

    phis: array
        the death rates of COVID-19

    NPIs_fitting: array
        the list of NPIs

    pop: number
        the total population of a specific zone or contry

    init_beta: number
        initial transmission rate of COVID-19 between susceptible and infected

    init_gamma: number
        transition rate of COVID-19 between exposed and infected with unvaccinated

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases without vaccinated

    init_NPIs_delay_days: number
        the delay of NPIs in effect

    init_NPIs_eff: array
        initial transmission rates of COVID-19 between vaccinated and infected caused by NPIs

    Notes: betas is dependent on variants, vaccination, and NPIs

    :return:
    susceptible: array
        the number of susceptible (active)

    protected: array
        the accumulated number of vaccinated (active)

    exposed: array
        the number of exposed without vaccinated (active)

    infected: array
        the number of infected with unvaccinated (active)

    quarantined: array
        the number of quarantined without vaccinated (active)

    recovered: array
        the accumulated number of recovered without vaccinated (accumulated)

    death: array
        the accumulated number of death without vaccinated (accumulated)

    Notes:

    """
    # define variables
    susceptible = x_data[0]
    protected = x_data[1]
    exposed = x_data[2]
    infected = x_data[3]
    quarantined = x_data[4]
    recovered = x_data[5]
    death = x_data[6]
    alphas = x_data[7]
    lams = x_data[8]
    phis = x_data[9]
    NPIs = x_data[10]
    pop = x_data[11]
    data_points = len(alphas)
    # set initial values of parameters
    init_beta = args[0]
    init_gamma = args[1]
    init_delta = args[2]
    init_NPIs_delay_days = args[3]
    init_NPIs_eff = args[4: 8]  # ***
    betas = np.zeros(data_points) + init_beta
    pop = pop[0]
    tt = [0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        # establish NPIs as change points
        if NPIs[ii] != 0:
            betas[ii + init_NPIs_delay_days:] = betas[ii] + init_NPIs_eff[tt[0]]
            tt[0] = tt[0] + 1
    for ii in range(1, data_points):
        # compute Ordinary Differential Equations of SPEIQRD
        # susceptible
        susceptible[ii] = susceptible[ii - 1] + -1 * betas[ii - 1] * susceptible[ii - 1] * infected[ii - 1] / pop + \
                          -1 * alphas[ii - 1] * pop
        if susceptible[ii] < 0:
            break
        # protected
        protected[ii] = protected[ii - 1] + alphas[ii - 1] * pop
        # exposed
        exposed[ii] = exposed[ii - 1] + betas[ii - 1] * susceptible[ii - 1] * infected[ii - 1] / pop + \
                      -1 * init_gamma * exposed[ii - 1]
        # infected
        infected[ii] = infected[ii - 1] + init_gamma * exposed[ii - 1] + -1 * init_delta * infected[ii - 1]
        # quarantined
        quarantined[ii] = quarantined[ii - 1] + init_delta * infected[ii - 1] + -1 * lams[ii - 1] * quarantined[ii - 1]\
                          - phis[ii - 1] * quarantined[ii - 1]
        # recovered
        recovered[ii] = recovered[ii - 1] + lams[ii - 1] * quarantined[ii - 1]
        # death
        death[ii] = death[ii - 1] + phis[ii - 1] * quarantined[ii - 1]
    # document the progress of the program
    global HuCount
    HuCount = HuCount + 1
    if HuCount % 5000 == 0:
        print("HuCount: ", HuCount)
    with open("iter.txt", "a+") as file_object:
        msd = 0.5 * sum((quarantined - YDATA) ** 2)
        file_object.write(str(msd) + "\n")
    return quarantined

def SPEIQRDSimulation(x_data, *args):

    """
    implement deterministic version of SPEIQRD
    :parameter:
    --------
    susceptible: array
        values of susceptible

    protected: array
        values of vaccinated, considering the effectiveness of vaccine

    exposed: array
        values of exposed

    infected: array
        values of infected

    quarantined: array
        values of reported cases

    recovered: array
        values of recovered cases

    death: array
        values of death cases

    pi: string
        variant of COVID-19 virus

    alphas: array
        daily protection rate

    lams: array
        the recovery rates of COVID-19

    phis: array
        the death rates of COVID-19 without vaccinated

    NPIs_fitting: array
        the list of NPIs

    pop: number
        the total population of a specific zone or country

    init_beta: number
        initial transmission rate of COVID-19 between susceptible and infected

    init_gamma: number
        transition rate of COVID-19 between exposed and infected

    init_delta: number
        confirmation rate of COVID-19 between infected and quarantine cases

    init_NPIs_delay_days: number
        the delay of NPIs in effect

    init_NPIs_eff: array
        initial transmission rates of COVID-19 between susceptible and infected caused by NPIs

    Notes: betas is dependent on variants, and NPIs


    :return:
    susceptible: array
        the number of susceptible

    protected: array
        values of vaccinated, considering the effectiveness of vaccine

    exposed: array
        the number of exposed

    infected: array
        the number of infected

    quarantined: array
        the number of quarantined

    recovered: array
        the accumulated number of recovered

    death: array
        the accumulated number of death

    Notes:

    """

    susceptible = x_data[0]
    protected = x_data[1]
    exposed = x_data[2]
    infected = x_data[3]
    quarantined = x_data[4]
    recovered = x_data[5]
    death = x_data[6]
    alphas = x_data[7]
    lams = x_data[8]
    phis = x_data[9]
    NPIs = x_data[10]
    pop = x_data[11]
    data_points = len(alphas)
    # values of optimal parameters
    init_beta = args[0]
    init_gamma = args[1]
    init_delta = args[2]
    init_NPIs_delay_days = args[3]
    init_NPIs_eff = args[4: 8]  # ***

    betas = np.zeros(data_points) + init_beta
    pop = pop[0]
    # record the number of NPIs
    tt = [0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        if NPIs[ii] != 0:
            betas[ii + init_NPIs_delay_days:] = betas[ii] + init_NPIs_eff[tt[0]]
            tt[0] = tt[0] + 1

    for ii in range(1, data_points):
        # compute ODE of SPEIQRD
        # susceptible
        susceptible[ii] = susceptible[ii-1] + -1 * betas[ii-1] * susceptible[ii-1] * infected[ii-1] / pop + \
                          -1 * alphas[ii-1] * pop
        # protected
        protected[ii] = protected[ii-1] + alphas[ii-1] * pop
        # exposed
        exposed[ii] = exposed[ii-1] + betas[ii-1] * susceptible[ii-1] * infected[ii-1] / pop + \
                      -1 * init_gamma * exposed[ii-1]
        # infected
        infected[ii] = infected[ii - 1] + init_gamma * exposed[ii-1] + -1 * init_delta * infected[ii-1]
        # quarantined
        quarantined[ii] = quarantined[ii-1] + init_delta * infected[ii-1] + -1 * lams[ii-1] * quarantined[ii-1] + \
            -1 * phis[ii-1] * quarantined[ii-1]
        # recovered
        recovered[ii] = recovered[ii-1] + lams[ii-1] * quarantined[ii-1]
        # death
        death[ii] = death[ii-1] + phis[ii-1] * quarantined[ii-1]

    return susceptible, protected, exposed, infected, quarantined, recovered, death

def ablationFit(init, x_data, y_data, bounds):
    """
    :param init: is a set of initial values of parameters
                  init_beta, init_gamma, init_delta, init_NPIs_delay_days, init_NPIs_eff
    :param x_data: is a set of hyperparameters
                    susceptible, protected, exposed, infected, quarantined, recovered, death,
                    alphas, lams, phis, NPIs, pop
    :param y_data: is the target
                    quarantined_truth
    :param bounds: is the bounds of parameters
    :return: a set of parameters

    """
    x_data = x_data
    global YDATA
    YDATA = y_data
    # print("bounds: ", bounds)
    # print("init:", init)
    print("optimizing the parameters ......")
    popt, pcov = optimize.curve_fit(SPEIQRDFit, x_data, y_data, p0=init, bounds=bounds, maxfev=1e+7)
    return popt

def SVEIQRD2Fit(x_data, *args):
    """
    implement deterministic version of SVEIQRD
    :parameter:
    --------
    susceptible: array
        values of susceptible

    vaccinated: array
        values of vaccinated

    exposed: array
        values of exposed

    infected: array
        values of infected

    quarantined: array
        values of reported cases

    recovered: array
        values of recovered cases

    death: array
        values of death cases

    pi: string
        variant of COVID-19 virus

    alphas: array
        daily 2_doses vaccination rate (daily)

    lams: array
        the recovery rates of COVID-19

    phis: array
        the death rates of COVID-19

    NPIs_no_vac: array
        the list of NPIs for the unvaccinated

    NPIs_vac: array
        the list of NPIs for the vaccinated

    vaccine_eff: array
        the waning of vaccine in its effectiveness

    pop: number
        the total population of a specific zone or contry

    init_beta_zero: number
        initial transmission rate of COVID-19 between the susceptible and infected

    init_beta_vac: number
        initial transmission rate of COVID-19 between the vaccinated and infected

    init_gamma: number
        transition rate of COVID-19 between exposed and infected

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases

    init_NPIs_delay_days: number
        the delay of NPIs in effect

    init_NPIs_eff_zero: array
        initial transmission rates of COVID-19 between the susceptible and infected caused by NPIs

    init_NPIs_eff_vac: array
        initial transmission rates of COVID-19 between the vaccinated and infected caused by NPIs

    Notes: betas is dependent on variants, vaccination, and NPIs

    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed: array
        the number of exposed (active)

    infected: array
        the number of infected (active)

    quarantined: array
        the number of quarantined (active)

    recovered: array
        the accumulated number of recovered (accumulated)

    death: array
        the accumulated number of death (accumulated)

    Notes:

    """
    # define variables
    susceptible = x_data[0]
    vaccinated = x_data[1]
    exposed = x_data[2]
    infected = x_data[3]
    quarantined = x_data[4]
    recovered = x_data[5]
    death = x_data[6]
    alphas = x_data[7]
    lams = x_data[8]
    phis = x_data[9]
    NPIs_no_vac = x_data[10]
    NPIs_vac = x_data[11]
    vaccine_eff = x_data[12]
    pop = x_data[13]

    data_points = len(alphas)

    # set initial values of parameters
    init_beta_zero = args[0]
    init_beta_vac = args[1]
    init_gamma = args[2]
    init_delta = args[3]
    init_NPIs_delay_days = args[4]
    init_NPIs_eff_zero = args[5: 9]  # ***
    init_NPIs_eff_vac = args[9: 10]  # ***

    betas_zero = np.zeros(data_points) + init_beta_zero
    betas_vac = np.zeros(data_points) + init_beta_vac
    pop = pop[0]

    tt = [0, 0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        # establish NPIs as change points
        if NPIs_no_vac[ii] != 0:
            betas_zero[ii + init_NPIs_delay_days:] = betas_zero[ii] + init_NPIs_eff_zero[tt[0]]
            tt[0] = tt[0] + 1
        if NPIs_vac[ii] != 0:
            betas_vac[ii + init_NPIs_delay_days:] = betas_vac[ii] + init_NPIs_eff_vac[tt[1]]
            tt[1] = tt[1] + 1

    for ii in range(1, data_points):
        # compute Ordinary Differential Equations of SVEIQRD
        # susceptible
        susceptible[ii] = susceptible[ii - 1] + -1 * betas_zero[ii - 1] * susceptible[ii - 1] * infected[
            ii - 1] / pop + -1 * alphas[ii - 1] * pop
        if susceptible[ii] < 0:
            print(ii, susceptible[ii])
            print("betas_zero[ii-1]", betas_zero[ii - 1])
            print("susceptible[ii-1]", susceptible[ii - 1])
            print("infected[ii-1]", infected[ii - 1])
            print("alphas[ii-1]", alphas[ii - 1])
            print("pop", pop)
            break
        # vaccinated
        vaccinated[ii] = vaccinated[ii - 1] - \
                         betas_vac[ii - 1] * (1 - vaccine_eff[ii - 1]) * vaccinated[ii - 1] * infected[ii - 1] / pop + \
                         alphas[ii - 1] * pop
        # exposed
        exposed[ii] = exposed[ii - 1] + betas_zero[ii - 1] * susceptible[ii - 1] * infected[ii - 1] / pop + \
                      betas_vac[ii - 1] * (1 - vaccine_eff[ii - 1]) * vaccinated[ii - 1] * infected[ii - 1] / pop + \
                      -1 * init_gamma * exposed[ii - 1]
        # infected
        infected[ii] = infected[ii - 1] + init_gamma * exposed[ii - 1] + -1 * init_delta * infected[ii - 1]
        # quarantined
        quarantined[ii] = quarantined[ii - 1] + init_delta * infected[ii - 1] + \
                          -1 * lams[ii - 1] * quarantined[ii - 1] + -1 * phis[ii - 1] * quarantined[ii - 1]
        # recovered
        recovered[ii] = recovered[ii - 1] + lams[ii - 1] * quarantined[ii - 1]
        # death
        death[ii] = death[ii - 1] + phis[ii - 1] * quarantined[ii - 1]

    global HuCount
    HuCount = HuCount + 1

    if HuCount % 5000 == 0:
        print("HuCount: ", HuCount)

    with open("iter.txt", "a+") as file_object:
        msd = 0.5 * sum((quarantined - YDATA) ** 2)
        file_object.write(str(msd) + "\n")
    return quarantined

def SVEIQRDSimulation(x_data, *args):
    """
    implement deterministic version of SVEIQRD
    :parameter:
    --------
    susceptible: array
        values of susceptible

    vaccinated: array
        values of vaccinated

    exposed: array
        values of exposed

    infected: array
        values of infected

    quarantined: array
        values of reported cases

    recovered: array
        values of recovered cases

    death: array
        values of death cases

    pi: string
        variant of COVID-19 virus

    alphas: array
        daily 2_doses vaccination rate (daily)

    lams: array
        the recovery rates of COVID-19

    phis: array
        the death rates of COVID-19

    NPIs_no_vac: array
        the list of NPIs for the unvaccinated

    NPIs_vac: array
        the list of NPIs for the vaccinated

    vaccine_eff: array
        the waning of vaccine in its effectiveness against infection

    pop: number
        the total population of a specific zone or contry

    init_beta_zero: number
        initial transmission rate of COVID-19 between the susceptible and infected

    init_beta_vac: number
        initial transmission rate of COVID-19 between the vaccinated and infected

    init_gamma: number
        transition rate of COVID-19 between exposed and infected

    init_delta: number
        confirmation rate of COVID-19 between infected and confirmed cases

    init_NPIs_delay_days: number
        the delay of NPIs in effect

    init_NPIs_eff_zero: array
        initial transmission rates of COVID-19 between the susceptible and infected caused by NPIs

    init_NPIs_eff_vac: array
        initial transmission rates of COVID-19 between the vaccinated and infected caused by NPIs

    Notes: betas is dependent on variants, vaccination, and NPIs


    :return:
    susceptible: array
        the number of susceptible (active)

    vaccinated: array
        the accumulated number of vaccinated (active)

    exposed: array
        the number of exposed (active)

    infected: array
        the number of infected (active)

    quarantined: array
        the number of quarantined (active)

    recovered: array
        the accumulated number of recovered (accumulated)

    death: array
        the accumulated number of death (accumulated)

    Notes:

    """

    susceptible = x_data[0]
    vaccinated = x_data[1]
    exposed = x_data[2]
    infected = x_data[3]
    quarantined = x_data[4]
    recovered = x_data[5]
    death = x_data[6]
    alphas = x_data[7]
    lams = x_data[8]
    phis = x_data[9]
    NPIs_no_vac = x_data[10]
    NPIs_vac = x_data[11]
    vaccine_eff = x_data[12]
    pop = x_data[13]

    data_points = len(alphas)
    # values of optimal parameters
    init_beta_zero = args[0]
    init_beta_vac = args[1]
    init_gamma = args[2]
    init_delta = args[3]
    init_NPIs_delay_days = args[4]
    init_NPIs_eff_zero = args[5: 9]  # ***
    init_NPIs_eff_vac = args[9: 10]  # ***

    betas_zero = np.zeros(data_points) + init_beta_zero
    betas_vac = np.zeros(data_points) + init_beta_vac
    pop = pop[0]

    tt = [0, 0]
    init_NPIs_delay_days = int(np.ceil(init_NPIs_delay_days))
    for ii in range(data_points):
        if NPIs_no_vac[ii] != 0:
            betas_zero[ii + init_NPIs_delay_days:] = betas_zero[ii] + init_NPIs_eff_zero[tt[0]]
            tt[0] = tt[0] + 1
        if NPIs_vac[ii] != 0:
            betas_vac[ii + init_NPIs_delay_days:] = betas_vac[ii] + init_NPIs_eff_vac[tt[1]]
            tt[1] = tt[1] + 1

    for ii in range(1, data_points):
        # compute ODE of SVEIQRD
        # susceptible
        susceptible[ii] = susceptible[ii - 1] + \
                          -1 * betas_zero[ii - 1] * susceptible[ii - 1] * infected[ii - 1] / pop + \
                          -1 * alphas[ii - 1] * pop
        # vaccinated
        vaccinated[ii] = vaccinated[ii - 1] + \
                         -1 * betas_vac[ii - 1] * (1 - vaccine_eff[ii - 1]) * vaccinated[ii - 1] * infected[
                             ii - 1] / pop + alphas[ii - 1] * pop
        # exposed
        exposed[ii] = exposed[ii - 1] + betas_zero[ii - 1] * susceptible[ii - 1] * infected[ii - 1] / pop + \
                      betas_vac[ii - 1] * (1 - vaccine_eff[ii - 1]) * vaccinated[ii - 1] * infected[
                          ii - 1] / pop + -1 * init_gamma * exposed[ii - 1]
        # infected
        infected[ii] = infected[ii - 1] + init_gamma * exposed[ii - 1] + \
                            -1 * init_delta * infected[ii - 1]
        # quarantined
        quarantined[ii] = quarantined[ii - 1] + init_delta * infected[ii - 1] + \
                               -1 * lams[ii - 1] * quarantined[ii - 1] + \
                               -1 * phis[ii - 1] * quarantined[ii - 1]
        # recovered
        recovered[ii] = recovered[ii - 1] + lams[ii - 1] * quarantined[ii - 1]
        # death
        death[ii] = death[ii - 1] + phis[ii - 1] * quarantined[ii - 1]

    return susceptible, vaccinated, exposed, infected, quarantined, recovered, death

def ablationFit2(init, x_data, y_data, bounds):
    """
    :param init: is the set of initial values of parameters
                  init_beta_zero_one, init_beta_zero_two, init_beta_vac_one, init_beta_vac_two,
                  init_gamma, init_delta, init_NPIs_delay_days,
                  init_NPIs_eff_zero_ones, init_NPIs_eff_zero_twos,init_NPIs_eff_vac_ones, init_NPIs_eff_vac_twos
    :param x_data: is the set of hyper parameters
                    susceptible, vaccinated, exposed_zero, exposed_vac, infected_zero, infected_vac,
                    quarantined_zero, quarantined_vac, recovered_zero, recovered_vac, death_zero, death_vac,
                    alphas, lam_zeros, lam_vacs, phi_zeros, phi_vacs, NPIs_no_vac, NPIs_vac, vaccine_eff, pop
    :param y_data: is the target
                    quarantined_average
    :param bounds: is the bounds of parameters
    :return: set of parameters

    """
    x_data = x_data
    global YDATA
    YDATA = y_data
    popt, pcov = optimize.curve_fit(SVEIQRD2Fit, x_data, y_data, p0=init, bounds=bounds,
                                            maxfev=1e+7)
    return popt

def recoveryRates(daily_recovered, infected):
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
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            beta_zero_ones[ii + init_NPIs_delay_days:] = beta_zero_ones[ii] + init_NPIs_eff_zero_ones[tt[0]]
            tt[0] = tt[0] + 1
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            beta_zero_twos[ii + init_NPIs_delay_days:] = beta_zero_twos[ii] + init_NPIs_eff_zero_twos[tt[1]]
            tt[1] = tt[1] + 1
            beta_vac_ones[ii + init_NPIs_delay_days:] = beta_vac_ones[ii] + init_NPIs_eff_vac_ones[tt[2]]
            tt[2] = tt[2] + 1
            beta_vac_twos[ii + init_NPIs_delay_days:] = beta_vac_twos[ii] + init_NPIs_eff_vac_twos[tt[3]]
            tt[3] = tt[3] + 1

    return beta_zero_ones, beta_zero_twos, beta_vac_ones, beta_vac_twos


def transmissionRate(NPIs_delay_days, NPIs_no_vac, NPIs_vac, init_beta_zero_one, init_beta_zero_two, init_beta_vac_one,
          init_beta_vac_two, NPIs_eff_zero_one, NPIs_eff_zero_two, NPIs_eff_vac_one,
          NPIs_eff_vac_two):
    # document the events of NPIs
    tt = [0, 0, 0, 0]
    data_points = len(NPIs_no_vac)
    NPIs_delay_days = int(np.ceil(NPIs_delay_days))
    # define the variable of transmission rates
    trans_zero_one = np.zeros(data_points) + init_beta_zero_one
    trans_zero_two = np.zeros(data_points) + init_beta_zero_two
    trans_vac_one = np.zeros(data_points) + init_beta_vac_one
    trans_vac_two = np.zeros(data_points) + init_beta_vac_two
    #print("NPIs_eff_zero_one:", NPIs_eff_zero_one)
    for ii in range(data_points):
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] != 0:
            # trans_zero_one[ii + NPIs_delay_days:] = trans_zero_one[ii] + NPIs_eff_zero_one[tt[0]]
            trans_zero_one[ii + NPIs_delay_days:] = trans_zero_one[ii + NPIs_delay_days - 1] + NPIs_eff_zero_one[tt[0]]
            tt[0] = tt[0] + 1
            # trans_zero_two[ii + NPIs_delay_days:] = trans_zero_two[ii] + NPIs_eff_zero_two[tt[1]]
            trans_zero_two[ii + NPIs_delay_days:] = trans_zero_two[ii + NPIs_delay_days - 1] + NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # trans_vac_one[ii + NPIs_delay_days:] = trans_vac_one[ii] + NPIs_eff_vac_one[tt[2]]
            trans_vac_one[ii + NPIs_delay_days:] = trans_vac_one[ii + NPIs_delay_days - 1] + NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
            # trans_vac_two[ii + NPIs_delay_days:] = trans_vac_two[ii] + NPIs_eff_vac_two[tt[3]]
            trans_vac_two[ii + NPIs_delay_days:] = trans_vac_two[ii + NPIs_delay_days - 1] + NPIs_eff_vac_two[tt[3]]
            tt[3] = tt[3] + 1
        if NPIs_no_vac[ii] != 0 and NPIs_vac[ii] == 0:
            # trans_zero_one[ii + NPIs_delay_days:] = trans_zero_one[ii] + NPIs_eff_zero_one[tt[0]]
            trans_zero_one[ii + NPIs_delay_days:] = trans_zero_one[ii + NPIs_delay_days - 1] + NPIs_eff_zero_one[tt[0]]
            tt[0] = tt[0] + 1
            # trans_zero_two[ii + NPIs_delay_days:] = trans_zero_two[ii] + NPIs_eff_zero_two[tt[1]]
            trans_zero_two[ii + NPIs_delay_days:] = trans_zero_two[ii + NPIs_delay_days - 1] + NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # trans_vac_one[ii + NPIs_delay_days:] = trans_vac_one[ii] + NPIs_eff_vac_one[tt[2]]
            trans_vac_one[ii + NPIs_delay_days:] = trans_vac_one[ii + NPIs_delay_days - 1] + NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
        if NPIs_no_vac[ii] == 0 and NPIs_vac[ii] != 0:
            # trans_zero_two[ii + NPIs_delay_days:] = trans_zero_two[ii] + NPIs_eff_zero_two[tt[1]]
            trans_zero_two[ii + NPIs_delay_days:] = trans_zero_two[ii + NPIs_delay_days - 1] + NPIs_eff_zero_two[tt[1]]
            tt[1] = tt[1] + 1
            # trans_vac_one[ii + NPIs_delay_days:] = trans_vac_one[ii] + NPIs_eff_vac_one[tt[2]]
            trans_vac_one[ii + NPIs_delay_days:] = trans_vac_one[ii + NPIs_delay_days - 1] + NPIs_eff_vac_one[tt[2]]
            tt[2] = tt[2] + 1
            # trans_vac_two[ii + NPIs_delay_days:] = trans_vac_two[ii] + NPIs_eff_vac_two[tt[3]]
            trans_vac_two[ii + NPIs_delay_days:] = trans_vac_two[ii + NPIs_delay_days] + NPIs_eff_vac_two[tt[3]]
            tt[3] = tt[3] + 1

        if trans_zero_one[ii] <= 0:
            trans_zero_one[ii] = 0
        if trans_zero_two[ii] <= 0:
            trans_zero_two[ii] = 0
        if trans_vac_one[ii] <= 0:
            trans_vac_one[ii] = 0
        if trans_vac_two[ii] <= 0:
            trans_vac_two[ii] = 0

    return trans_zero_one, trans_zero_two, trans_vac_one, trans_vac_two

def transmissionRateSPEIQRD(NPIs_delay_days, NPIs, init_beta, NPIs_eff):
    # document the events of NPIs
    tt = [0]
    data_points = len(NPIs)
    NPIs_delay_days = int(np.ceil(NPIs_delay_days))
    # define the variable of transmission rates
    transmission_rate = np.zeros(data_points) + init_beta
    #print("NPIs_eff:", NPIs_eff)
    for ii in range(data_points):
        if NPIs[ii] != 0:
            transmission_rate[ii + NPIs_delay_days:] = transmission_rate[ii] + NPIs_eff[tt[0]]
            tt[0] = tt[0] + 1
    return transmission_rate

def transmissionRateSVEIQRD(NPIs_delay_days, NPIs_no_vac, NPIs_vac, init_beta_zero, init_beta_vac, NPIs_eff_zero,
                            NPIs_eff_vac):
    # document the events of NPIs
    tt = [0, 0]
    data_points = len(NPIs_no_vac)
    NPIs_delay_days = int(np.ceil(NPIs_delay_days))
    # define the variable of transmission rates
    trans_zero = np.zeros(data_points) + init_beta_zero
    trans_vac = np.zeros(data_points) + init_beta_vac
    #print("NPIs_eff:", NPIs_eff)
    for ii in range(data_points):
        if NPIs_no_vac[ii] != 0:
            trans_zero[ii + NPIs_delay_days:] = trans_zero[ii + NPIs_delay_days - 1] + NPIs_eff_zero[tt[0]]
            tt[0] = tt[0] + 1
        if NPIs_vac[ii] != 0:
            trans_vac[ii + NPIs_delay_days:] = trans_vac[ii + NPIs_delay_days - 1] + NPIs_eff_vac[tt[1]]
            tt[1] = tt[1] + 1
    return trans_zero, trans_vac

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

def computeEff(file_path):
    """
        compute the effectivenss of vaccine, taking the waning of effectiveness and boosters into account
        :parameter:
        --------
        the path of file.csv
            file content format: Date, people_fully_vaccinated_per_hundred, total_boosters_per_hundred,
            daily_vaccinated_per_hundred, daily_boosters_per_hundred

        vaccinated: array
            values of vaccinated

        exposed_zero: array

        :return:
        a csv file:
            file content format: Date, people_fully_vaccinated_per_hundred, total_boosters_per_hundred,
            daily_vaccinated_per_hundred, daily_boosters_per_hundred, effectiveness

        usage:
        file_path = "Austria_info_vaccination.csv"
        functions.computeEff(file_path)
"""
    # read file
    df_cases = pd.read_csv(file_path)
    # obtain information about vaccination
    total_vaccination = df_cases.iloc[:, 1]
    total_boosters = df_cases.iloc[:, 2]
    daily_vaccination = df_cases.iloc[:, 3]
    daily_boosters = df_cases.iloc[:, 4]
    coeff = [0.775, 0.732, 0.696, 0.517, 0.225, 0.173]  # the waning curve in efficacy of vaccine
    # define the length of time in days
    length_days = len(df_cases.iloc[:, 0])

    eff_vacc = np.zeros(length_days)  # initiate values of effectiveness

    # compute the effectiveness of vaccine
    for ii in range(2, length_days):
        tempt_eff = 0.0
        for jj in range(ii):
            # considering the waning effectiveness and boosters
            if ii - jj <= 30:
                tempt_eff = tempt_eff + (daily_vaccination[jj] + daily_boosters[jj]) * coeff[0]
            elif ii - jj <= 60:
                tempt_eff = tempt_eff + (daily_vaccination[jj] + daily_boosters[jj]) * coeff[1]
            elif ii - jj <= 90:
                tempt_eff = tempt_eff + (daily_vaccination[jj] + daily_boosters[jj]) * coeff[2]
            elif ii - jj <= 120:
                tempt_eff = tempt_eff + (daily_vaccination[jj] + daily_boosters[jj]) * coeff[3]
            elif ii - jj <= 150:
                tempt_eff = tempt_eff + (daily_vaccination[jj] + daily_boosters[jj]) * coeff[4]
            else:
                tempt_eff = tempt_eff + (daily_vaccination[jj] + daily_boosters[jj]) * coeff[5]
        # Assumption that people with being fully vaccinated more than 180 will get booster. The overlap therefore is
        # removed.
        tempt_eff = tempt_eff - total_boosters[ii] * coeff[5]
        if total_vaccination[ii - 1] == 0:
            eff_vacc[ii] = 0
        else:
            eff_vacc[ii] = tempt_eff / total_vaccination[ii - 1]

    df_cases['effectiveness'] = eff_vacc
    df_cases.to_csv("effectiveness_vaccine.csv", index=False)
    return "effectiveness_vaccine.csv"




