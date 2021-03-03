# Internal
import typing as T

# External
import matplotlib.pyplot as pl

# Project
from ..edo import edo_sis_k1, edo_sis_k2


def show_plot(data: T.Any) -> None:
    pl.plot(data[:, 0], "-g", label="Susceptibles")
    pl.title("EDO - SIS - K=1")
    pl.xlabel("Time")
    pl.plot(data[:, 1], "-r", label="Infectious")
    pl.show()

    # Último valor RES[:, 0]
    print("Suscetíveis ao final:", data[-1, 0])

    # Último valor RES[:, 1]
    print("Infectados ao final:", data[-1, 1])


def show_plot_k2(data: T.Any) -> None:
    pl.plot(data[:, 1], "orangered", label="Infected-1", linestyle="dashed")
    pl.plot(data[:, 2], "darkred", label="Infected-2", linestyle="dashed")
    pl.plot(data[:, 0], "-g", label="Susceptible")
    pl.plot([x + y for x, y in zip(data[:, 1], data[:, 2])], "-r", label="Infectious")
    pl.legend()

    pl.xlabel("Time")
    pl.ylabel("Population %")
    pl.show()

    print("Suscetíveis ao final:", data[-1, 0])
    print("Infectados com um post ao final:", data[-1, 1])
    print("Infectados com dois posts ao final:", data[-1, 2])


def main() -> None:
    res = edo_sis_k2(0.2, 0.0, 0.1, 1.0, 0.15, 50)
    show_plot_k2(res)

    res = edo_sis_k1(0.01, 0.5, 0.15, 200)

    # Plotting
    show_plot(res)

    res = edo_sis_k1(0.5, 0.2, 0.15, 200)

    # Plotting
    show_plot(res)


if __name__ == "__main__":
    main()

__all__ = ("main",)
