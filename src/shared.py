from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# modified system of ordinary differential equations
def get_time_points(start, end, num_points):
    return np.linspace(start, end, num_points)

def odes(x, t, c, r, lambda_):
    S = x[0]
    I = x[1]            # noqa  # I is reserved
    R = x[2]
    dSdt = -c * I * S + lambda_ * R
    dIdt = c * I * S - r * I
    dRdt = r * I - lambda_ * R
    return [dSdt,dIdt,dRdt]

def solve_one(x0, t, c, r, lambda_):
    x = odeint(odes, x0, t, args=(c, r, lambda_))  # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html about args
    return x

def unpack_solution(x):
    x_S = x[:,0]
    x_I = x[:,1]
    x_R = x[:,2]
    return x_S, x_I, x_R

def plot_solution(t, x_S, x_I, x_R, save: bool = False):
    plt.plot(t,x_S,label='S')
    plt.plot(t,x_I,label='I')
    plt.plot(t,x_R,label='R')
    plt.legend()
    plt.xlabel('time')
    if save:
        plt.savefig('SIR_solution.pdf')
    plt.show()


def solve_and_plot_continuous_SIR(
    x0,
    c,
    r,
    lambda_,
    num_points,
    **kwargs,
):
    """
    Solve the continuous SIR model and plot the results.

    Args:
        x0: Initial conditions for the SIR model.
        c: Contact rate.
        r: Recovery rate.
        lambda_: Immunity loss rate.
        num_points: Number of time points to solve the model at.
        **kwargs: Additional keyword arguments.
            start: Start time.
            end: End time.
            verbose: Whether to print the results.
            plot: Whether to plot the results.
            save: Whether to save the plot.

    Returns:
        x_S: Susceptible population.
        x_I: Infected population.
        x_R: Recovered population.
    """

    start = kwargs.get("start", 0)
    end = kwargs.get("end", 1000)
    verbose = kwargs.get("verbose", False)
    plot = kwargs.get("plot", True)
    save = kwargs.get("save", False)

    t = get_time_points(start, end, num_points)
    x = solve_one(x0, t, c, r, lambda_)
    x_S, x_I, x_R = unpack_solution(x)
    if plot:
        plot_solution(t, x_S, x_I, x_R, save)
    if verbose:
        print(f"S(0) = {x_S[0]}, I(0) = {x_I[0]}, R(0) = {x_R[0]}")
        print(f"S(t) = {x_S[-1]}, I(t) = {x_I[-1]}, R(t) = {x_R[-1]}")
    return x_S, x_I, x_R
