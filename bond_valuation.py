import numpy as np
import plotly.graph_objects as go
from scipy.linalg import solve_banded


class FiniteDiffBond:
    def __init__(self, params):
        self.params = params
        self.dr = params["max_int_rate"] / params["n_int_rate_steps"]
        self.dt = params["bond_maturity"] / params["n_time_steps"]

        self.old_bond_values = np.full(
            params["n_int_rate_steps"] + 1, self.params["bond_face_value"]
        )
        self.boundary_condition = "Dirichlet"

    def construct_bond_matrix(self, time_index):
        A_banded = np.zeros(shape=(3, self.params["n_int_rate_steps"] + 1))
        d = np.zeros(self.params["n_int_rate_steps"] + 1)
        l_and_u = (1, 1)

        # Boundary condition at the first node
        A_banded[1][0] = 1 / self.dt + (1 / self.dr) * self.params[
            "kappa"
        ] * self.params["theta"] * np.exp(
            self.params["mu"] * (time_index + 0.5) * self.dt
        )
        A_banded[0][1] = (
            -1
            / self.dr
            * self.params["kappa"]
            * self.params["theta"]
            * np.exp(self.params["mu"] * (time_index + 0.5) * self.dt)
        )

        # Populate middle rows
        for j in range(1, self.params["n_int_rate_steps"]):
            const_1 = (
                self.params["sigma"] ** 2
                * pow(j, 2 * self.params["beta"])
                * pow(self.dr, 2 * (self.params["beta"] - 1))
            )
            const_2 = (
                (1 / self.dr)
                * self.params["kappa"]
                * self.params["theta"]
                * np.exp(self.params["mu"] * (time_index + 0.5) * self.dt)
            )

            A_banded[2][j - 1] = 0.25 * (const_1 - const_2 + self.params["kappa"] * j)
            A_banded[1][j] = -(1 / self.dt + 0.5 * const_1 + j * self.dr / 2)
            A_banded[0][j + 1] = 0.25 * (const_1 + const_2 - self.params["kappa"] * j)

            d_1 = 0.25 * (-const_1 + const_2 - self.params["kappa"] * j)
            d_2 = -(1 / self.dt) + 0.5 * const_1 + j * self.dr / 2
            d_3 = 0.25 * (-const_1 - const_2 + self.params["kappa"] * j)
            d[j] = (
                d_1 * self.old_bond_values[j - 1]
                + d_2 * self.old_bond_values[j]
                + d_3 * self.old_bond_values[j + 1]
                - self.params["C"] * np.exp(-self.params["alpha"] * (time_index + 0.5) * self.dt)
            )

        # Boundary condition at the last node
        if self.boundary_condition == "Neumann":
            A_banded[2][self.params["n_int_rate_steps"] - 1] = -1 / self.dr
            A_banded[1][self.params["n_int_rate_steps"]] = 1 / self.dr
        else:
            A_banded[2][self.params["n_int_rate_steps"] - 1] = 0.0
            A_banded[1][self.params["n_int_rate_steps"]] = 1.0

        d[0] = 1 / self.dt * self.old_bond_values[0] + self.params["C"] * np.exp(
            -self.params["alpha"] * (time_index + 0.5) * self.dt
        )
        d[self.params["n_int_rate_steps"]] = 0.0

        return A_banded, d, l_and_u

    def run(self):
        n_r = self.params["n_int_rate_steps"] + 1
        n_t = self.params["n_time_steps"] + 1
        interest_rates = np.array([j * self.dr for j in range(n_r)])

        # Initialize 2D array to store B(r,t)
        bond_grid = np.zeros((n_r, n_t))
        bond_grid[:, -1] = self.old_bond_values  # final time = maturity

        # Backward iteration over time
        for i in range(self.params["n_time_steps"] - 1, -1, -1):
            A_banded, d, l_and_u = self.construct_bond_matrix(i)
            bond_values_new = solve_banded(l_and_u, A_banded, d)
            self.old_bond_values = np.copy(bond_values_new)
            bond_grid[:, i] = bond_values_new

        times = np.linspace(0, self.params["bond_maturity"], n_t)
        return interest_rates, times, bond_grid


def plot_surface(interest_rates, times, bond_grid):
    fig = go.Figure(
        data=[
            go.Surface(
                z=bond_grid.T,  # transpose so that x=interest rate, y=time
                x=interest_rates,
                y=times,
                colorscale="Viridis",
            )
        ]
    )

    fig.update_layout(
        title="Bond Price Surface B(r, t)",
        scene=dict(
            xaxis_title="Interest Rate (r)",
            yaxis_title="Time (t)",
            zaxis_title="Bond Value B(r,t)",
        ),
        template="plotly_white",
    )

    fig.show()


def main():
    params = {
        "bond_maturity": 3,
        "option_maturity": 0.9839,
        "strike_price": 49.6,
        "bond_face_value": 50,
        "initial_int_rate": 0.0275,
        "n_time_steps": 100,
        "n_int_rate_steps": 100,
        "max_int_rate": 1.0,
        "sigma": 0.251,
        "kappa": 0.08116,
        "theta": 0.0409,
        "mu": -0.0222,
        "alpha": 0.01,
        "beta": 0.653,
        "C": 1.07,
    }

    fin_dif_bond = FiniteDiffBond(params)
    interest_rates, times, bond_grid = fin_dif_bond.run()
    plot_surface(interest_rates, times, bond_grid)


if __name__ == "__main__":
    main()
