import numpy as np
import matplotlib.pyplot as plt

title_checkMSM = "          Explicit comparison of the entries of the different transition matrices \n\n\
The exponent $m$ in M$(n \Delta t)^m$ is chosen for each $n$ and $n_{fixed}$ in the way that \
$n \cdot m = n_{fixed}$\n "


def check_MSM(lags, transition_matrices):
    """ function to check for Markov criteria by comparing the matrix entries 
    for different lag times

    Parameters
    ----------
    lags : list of int
        list containing integer values. These lag times are then used 
        for the calculations
    transition_matrices : dict of numpy arrays
        dictionary containing the transition matrices of different lag
        times
    """
    # array that stores the trajectory of M(n Δt)^m for all lag times
    matrix_trajectory = np.zeros([2, 2, len(lags), len(lags)])
    # array that contains the transition matrix M(n_fixed Δt) for all 
    # lag times n_fixed Δt
    reference_matrix = np.zeros([2, 2, len(lags)])

    # loop over all lag times n_fixed Δt and store M(n_fixed Δt) as well as 
    # the corresponding matrix products M(n Δt)^m with m*n=n_fixed
    for i in range(len(lags)):
        n_fixed = lags[i]
        # loop over all other lag times and determine the required
        # factor/exponent m to get n*m = n_fixed
        for j in range(len(lags)):
            n = lags[j]
            if ((n_fixed > n) and (n_fixed % n == 0)):
                m = int(n_fixed / n)
                Mn_powered = \
                    np.linalg.matrix_power(transition_matrices[str(n)], m)
                matrix_trajectory[:, :, j, i] = Mn_powered
            else:
                # set nan so that it won't appear in the plot
                matrix_trajectory[:, :, j, i] = \
                    np.zeros([2, 2]).fill(float('nan')) 

        reference_matrix[:, :, i] = transition_matrices[str(n_fixed)]

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    axs = axs.flatten()
    axs[0].set_ylabel(r"P(-1$\rightarrow$ -1)")
    axs[1].set_ylabel(r"P(-1$\rightarrow$ 1)")
    axs[2].set_ylabel(r"P(1$\rightarrow$ -1)")
    axs[3].set_ylabel(r"P(1$\rightarrow$ 1)")
    axs[0].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")
    axs[1].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")
    axs[2].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")
    axs[3].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")    

    cm = plt.get_cmap('nipy_spectral')
    # plot the reference matrix M(n_fixed Δt) = M(n_fixed Δt)
    axs[0].plot(lags, reference_matrix[0, 0, :], label=r"M$(n_{fixed})$",
                color=cm(-1), marker="+", markersize=10)
    axs[3].plot(lags, reference_matrix[1, 1, :], color=cm(-1),
                marker="+", markersize=10)
    axs[1].plot(lags, reference_matrix[0, 1, :], color=cm(-1),
                marker="+", markersize=10)
    axs[2].plot(lags, reference_matrix[1, 0, :], color=cm(-1),
                marker="+", markersize=10)

    # now loop over all n and plot the curve of M(n Δt)^m
    for i in range(0, len(lags), 1):
        # if the trajectory contains only nan, don't plot at all
        if np.isnan(matrix_trajectory[0, 0, i, :]).all():
            continue

        c = cm(10+i*20)
        axs[0].plot(lags, matrix_trajectory[0, 0, i, :], ls="--",
                    marker="o", label=r"M$(%i \Delta t)^m$" % (lags[i]),
                    color=c)
        axs[1].plot(lags, matrix_trajectory[0, 1, i, :], ls="--",
                    marker="o", color=c)
        axs[2].plot(lags, matrix_trajectory[1, 0, i, :], ls="--",
                    marker="o", color=c)
        axs[3].plot(lags, matrix_trajectory[1, 1, i, :], ls="--",
                    marker="o", color=c)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               title=title_checkMSM)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, wspace=0.3, hspace=0.25) 
    plt.show()


title_limMSM = "    Comparison of the entries of the different transition matrices for long lag times\n\n\
The exponent $m$ in M$(n \Delta t)^m$ is chosen for each $n$ and $n_{fixed}$ in the way that \
$n \cdot m = n_{fixed}$\n "


def plot_M_limes(lags, transition_matrices, u01, u02, max_n_index=5, x_axis_start_n=0):
    """ function that plots the behaviour of the matrix elements in the
        limit of large lag times

    Parameters
    ----------
    lags : list of int
        list containing integer values. These lag times are then used 
        for the calculations
    transition_matrices : dict
        dictionary that contains the transition matrices of the different
        lag times
    u01 : float
        1st entry of the stationary state (eigenvector of eigenvalue 1)
    u02 : float
        2nd entry of the stationary state (eigenvector of eigenvalue 1)
    max_n_indices : int, optional
        index of maximum lag time n in lags for which M(nΔt)^m should be 
        plotted, default is 5
    x_axis_start_n : int, optional
        index of first lag time n_fixed that should be included in the plot
        should be chosen appropriately to see the relevant part of the graph
    """
    # array that stores the trajectory of M(n Δt)^m for all lag times
    matrix_trajectory = np.zeros([2, 2, len(lags), len(lags)])
    # array that contains the transition matrix M(n_fixed Δt) for all 
    # lag times n_fixed Δt
    reference_matrix = np.zeros([2, 2, len(lags)])

    # loop over all lag times n_fixed Δt and store M(n_fixed Δt) as well as 
    # the corresponding matrix products M(n Δt)^m with m*n=n_fixed
    for i in range(len(lags)):
        n_fixed = lags[i]
        # loop over all other lag times and determine the required
        # factor/exponent m to get n*m = n_fixed
        for j in range(len(lags)):

            if i < x_axis_start_n:
                # set nan so that it won't appear in the plot
                matrix_trajectory[:, :, j, i] = \
                    np.zeros([2, 2]).fill(float('nan')) 
                continue

            n = lags[j]
            if ((n_fixed > n) and (n_fixed % n == 0)):
                m = int(n_fixed / n)
                Mn_powered = \
                    np.linalg.matrix_power(transition_matrices[str(n)], m)
                matrix_trajectory[:, :, j, i] = Mn_powered
            else:
                # set nan so that it won't appear in the plot
                matrix_trajectory[:, :, j, i] = \
                    np.zeros([2, 2]).fill(float('nan')) 

        reference_matrix[:, :, i] = transition_matrices[str(n_fixed)]
        if i < x_axis_start_n:
            reference_matrix[:, :, i] = \
                    np.zeros([2, 2]).fill(float('nan'))

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    axs = axs.flatten()
    axs[0].set_ylabel(r"P(-1$\rightarrow$ -1)")
    axs[1].set_ylabel(r"P(-1$\rightarrow$ 1)")
    axs[2].set_ylabel(r"P(1$\rightarrow$ -1)")
    axs[3].set_ylabel(r"P(1$\rightarrow$ 1)")
    axs[0].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")
    axs[1].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")
    axs[2].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")
    axs[3].set_xlabel(r"lag time $n_{fixed} ~[\Delta t]$ ")    

    cm = plt.get_cmap('nipy_spectral')
    # plot the reference matrix M(n_fixed Δt) = M(n_fixed Δt)
    axs[0].semilogx(lags, reference_matrix[0, 0, :], label=r"M$(n_{fixed})$",
                    color=cm(-1), marker="+", markersize=10, ls="")
    axs[3].semilogx(lags, reference_matrix[1, 1, :], color=cm(-1),
                    marker="+", markersize=10, ls="")
    axs[1].semilogx(lags, reference_matrix[0, 1, :], color=cm(-1),
                    marker="+", markersize=10, ls="")
    axs[2].semilogx(lags, reference_matrix[1, 0, :], color=cm(-1),
                    marker="+", markersize=10, ls="")

    # now loop over all n and plot the curve of M(n Δt)^m
    for i in range(0, max_n_index, 1):
        # if the trajectory contains only nan, don't plot at all
        if np.isnan(matrix_trajectory[0, 0, i, :]).all():
            continue

        colorfactor = 20
        c = cm(10+i*colorfactor)
        axs[0].semilogx(lags, matrix_trajectory[0, 0, i, :], ls="--",
                        marker="o", label=r"M$(%i \Delta t)^m$" % (lags[i]),
                        color=c)
        axs[1].semilogx(lags, matrix_trajectory[0, 1, i, :], ls="--",
                        marker="o", color=c)
        axs[2].semilogx(lags, matrix_trajectory[1, 0, i, :], ls="--",
                        marker="o", color=c)
        axs[3].semilogx(lags, matrix_trajectory[1, 1, i, :], ls="--",
                        marker="o", color=c)

    axs[0].axhline(u01, label="%.2f and %.2f" % (u01, u02))
    axs[2].axhline(u01)
    axs[1].axhline(u02)
    axs[3].axhline(u02)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, title=title_limMSM)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, wspace=0.3, hspace=0.25) 
    plt.show()


def implied_timescale_evolution(trans_mats, lags, delta_t):
    """
    Produces a plot of the evolution of the implied timescale as a function of
    the lag time

    Parameters
    ----------
    trans_mats : dict
        dictionary of the transition matrices corresponding to the different
        lag times
    lags : list of ints
        list of the lag times that are supposed to be included in the plot
    """
    t_values = []  # list to store the calculated values for t
    lambdas = {}  # store the eigenvalues lambda_1

    for n in lags:
        # diagonalise the transition matrix and get the smaller eigenvalue
        eig_vals, eig_vecs = np.linalg.eig(trans_mats[str(n)])
        lambda_1 = eig_vals.min()
        # calculate and save the implied timescale and the eigenvalue
        t = - n * delta_t / np.log(lambda_1)
        t_values.append(t)
        lambdas.update({str(n): lambda_1})

    # plot the result
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_title("Evolution of the implied timescale")
    ax.set_xlabel(r"lag time $n$ $[\Delta t]$")
    ax.set_ylabel(r"Implied time scale $t$ [ps]")
    n_0 = 3000
    ax.axhline(-delta_t * n_0 / np.log(lambdas[str(n_0)]), label=r"$t\,(n_0 \,\Delta t)$")
    ax.axvline(n_0, ls="--", color="r", label=r"$n_0$", alpha=0.5)
    ax.plot(lags, t_values, ls="--", marker="o", color="g")
    ax.legend()
    plt.show()
