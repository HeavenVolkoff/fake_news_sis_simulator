# Internal
import typing as T

# External
import numpy as np
import scipy.integrate as spi


def edo_sis_k1(
    t0_infectious: float,
    transmission_rate: float,
    recovery_rate: float,
    total_time: int,
) -> T.Any:
    """Epidemic SIS model where K=1.

    Arguments:
        t0_infectious: Initial rate of infected
        transmission_rate: Transmission rate for the infection
        recovery_rate: User recovery rate
        total_time: Total time to run the model
    """
    # Susceptible population rate
    t0_susceptible = 1.0 - t0_infectious
    time_points = (t0_susceptible, t0_infectious)

    def diff_eqs(time_point: T.Tuple[float, float], _: T.Any) -> np.ndarray:
        y = np.zeros(2)
        susceptible = time_point[0]
        infectious = time_point[1]
        y[0] = -transmission_rate * susceptible * infectious + recovery_rate * infectious
        y[1] = transmission_rate * susceptible * infectious - recovery_rate * infectious
        return y  # For odeint

    t_start = 0.0
    t_end = total_time  # Apenas mudei de 1000 para 100
    t_inc = 1
    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    return spi.odeint(diff_eqs, time_points, t_range)


def edo_sis_k2(
    t0_infectious_1: float,
    t0_infectious2: float,
    transmission_rate_1: float,
    transmission_rate_2: float,
    recovery_rate: float,
    total_time: int,
) -> T.Any:
    """Epidemic SIS model where K=2.

    Arguments:
        t0_infectious_1: Initial rate of single slot infected.
        t0_infectious2: Initial rate of dual slot infected.
        transmission_rate_1: Transmission rate for single slot infected.
        transmission_rate_2: Transmission rate for dual slot infected.
        recovery_rate: Recovery rate
        total_time:
    """
    # Susceptible population rate
    t0_susceptible = 1.0 - t0_infectious_1 - t0_infectious2
    time_points = (t0_susceptible, t0_infectious_1, t0_infectious2)

    def diff_eqs(time_point: T.Tuple[float, float, float], _: T.Any) -> np.ndarray:
        y = np.zeros(3)
        v = time_point
        susceptible = (
            recovery_rate * v[1] - (transmission_rate_1 * v[1] + transmission_rate_2 * v[2]) * v[0]
        )
        one_post_infected = (
            (transmission_rate_1 * v[1] + transmission_rate_2 * v[2]) * v[0]
            + recovery_rate * v[2]
            - (
                recovery_rate * v[1]
                + (transmission_rate_1 * v[1] + transmission_rate_2 * v[2]) * v[1]
            )
        )
        two_post_infected = (transmission_rate_1 * v[1] + transmission_rate_2 * v[2]) * v[
            1
        ] - recovery_rate * v[2]
        y[0] = susceptible
        y[1] = one_post_infected
        y[2] = two_post_infected
        return y  # For odeint

    t_start = 0.0
    t_inc = 1.0
    t_range = np.arange(t_start, total_time + t_inc, t_inc)
    return spi.odeint(diff_eqs, time_points, t_range)


__all__ = ("edo_sis_k1", "edo_sis_k2")
