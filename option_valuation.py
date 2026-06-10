import numpy as np
import plotly.graph_objects as go
from scipy.linalg import solve_banded


class FiniteDiffAmericanOption:
    def __init__(self, params):

        self.params = params

        self.dr = params["max_int_rate"] / params["n_int_rate_steps"]
        self.dt = params["bond_maturity"] / params["n_time_steps"]

        self.boundary_condition = "Neumann"

        # grids
        self.rate_steps = params["n_int_rate_steps"]
        self.time_steps = params["n_time_steps"]

        self.interest_rates = np.array(
            [j * self.dr for j in range(self.rate_steps + 1)]
        )

        # initialise bond + option
        self.bond_values = np.full(self.rate_steps + 1, params["bond_face_value"])

        self.option_values = np.zeros(self.rate_steps + 1)

        # index of option maturity
        self.T1_index = int(params["option_maturity"] / self.dt)

    def construct_bond_matrix(self, time_index, bond_old):

        A_banded = np.zeros((3, self.rate_steps + 1))
        d = np.zeros(self.rate_steps + 1)
        l_and_u = (1, 1)

        t_mid = (time_index + 0.5) * self.dt

        drift_term = (
            (1 / self.dr)
            * self.params["kappa"]
            * self.params["theta"]
            * np.exp(self.params["mu"] * t_mid)
        )

        # r = 0 boundary
        A_banded[1][0] = 1 / self.dt + drift_term
        A_banded[0][1] = -drift_term

        for j in range(1, self.rate_steps):

            const_1 = (
                self.params["sigma"] ** 2
                * j ** (2 * self.params["beta"])
                * self.dr ** (2 * (self.params["beta"] - 1))
            )

            const_2 = drift_term

            A_banded[2][j - 1] = 0.25 * (const_1 - const_2 + self.params["kappa"] * j)
            A_banded[1][j] = -(1 / self.dt + 0.5 * const_1 + j * self.dr / 2)
            A_banded[0][j + 1] = 0.25 * (const_1 + const_2 - self.params["kappa"] * j)

            d_1 = 0.25 * (-const_1 + const_2 - self.params["kappa"] * j)
            d_2 = -(1 / self.dt) + 0.5 * const_1 + j * self.dr / 2
            d_3 = 0.25 * (-const_1 - const_2 + self.params["kappa"] * j)

            d[j] = (
                d_1 * bond_old[j - 1]
                + d_2 * bond_old[j]
                + d_3 * bond_old[j + 1]
                - self.params["C"] * np.exp(-self.params["alpha"] * t_mid)
            )

        # boundary at r_max
        if self.boundary_condition == "Neumann":
            A_banded[2][self.rate_steps - 1] = -1 / self.dr
            A_banded[1][self.rate_steps] = 1 / self.dr
        else:
            A_banded[2][self.rate_steps - 1] = 0.0
            A_banded[1][self.rate_steps] = 1.0

        d[0] = (1 / self.dt) * bond_old[0] + self.params["C"] * np.exp(
            -self.params["alpha"] * t_mid
        )
        d[self.rate_steps] = 0.0

        return A_banded, d, l_and_u

    def construct_option_matrix(self, time_index, option_old):

        A_banded = np.zeros((3, self.rate_steps + 1))
        d = np.zeros(self.rate_steps + 1)
        l_and_u = (1, 1)

        t_mid = (time_index + 0.5) * self.dt

        drift_term = (
            (1 / self.dr)
            * self.params["kappa"]
            * self.params["theta"]
            * np.exp(self.params["mu"] * t_mid)
        )

        # r = 0 boundary
        A_banded[1][0] = 1 / self.dt + drift_term
        A_banded[0][1] = -drift_term

        for j in range(1, self.rate_steps):

            const_1 = (
                self.params["sigma"] ** 2
                * j ** (2 * self.params["beta"])
                * self.dr ** (2 * (self.params["beta"] - 1))
            )

            const_2 = drift_term

            A_banded[2][j - 1] = 0.25 * (const_1 - const_2 + self.params["kappa"] * j)
            A_banded[1][j] = -(1 / self.dt + 0.5 * const_1 + j * self.dr / 2)
            A_banded[0][j + 1] = 0.25 * (const_1 + const_2 - self.params["kappa"] * j)

            d_1 = 0.25 * (-const_1 + const_2 - self.params["kappa"] * j)
            d_2 = -(1 / self.dt) + 0.5 * const_1 + j * self.dr / 2
            d_3 = 0.25 * (-const_1 - const_2 + self.params["kappa"] * j)

            d[j] = (
                d_1 * option_old[j - 1] + d_2 * option_old[j] + d_3 * option_old[j + 1]
            )

        # boundary
        if self.boundary_condition == "Neumann":
            A_banded[2][self.rate_steps - 1] = -1 / self.dr
            A_banded[1][self.rate_steps] = 1 / self.dr
        else:
            A_banded[2][self.rate_steps - 1] = 0.0
            A_banded[1][self.rate_steps] = 1.0

        d[0] = (1 / self.dt) * option_old[0]
        d[self.rate_steps] = 0.0

        return A_banded, d, l_and_u

    def apply_penalty_method(self, option_new, bond_new):

        rho = 1e8
        tol = 1e-8
        maxiter = 50

        for _ in range(maxiter):

            intrinsic = np.maximum(
                self.params["strike_price"] - bond_new,
                0.0,
            )

            penalty = rho * np.maximum(intrinsic - option_new, 0.0)

            if np.linalg.norm(penalty, 1) < tol:
                break

            option_new += penalty / rho

        return option_new

    def run(self, use_penalty=False, return_surface=False):
        n_r = self.params["n_int_rate_steps"] + 1
        n_t = self.params["n_time_steps"] + 1

        # Initialize grids to store option and bond values
        option_grid = np.zeros((n_r, n_t))
        bond_grid = np.zeros((n_r, n_t))

        bond_old = np.copy(self.bond_values)
        bond_new = np.copy(bond_old)

        # Step 1: Solve bond from T → T1
        for i in range(self.time_steps - 1, self.T1_index - 1, -1):
            A, d, lu = self.construct_bond_matrix(i, bond_old)
            bond_new = solve_banded(lu, A, d)
            bond_old = np.copy(bond_new)
            bond_grid[:, i] = bond_new

        # Step 2: Initialise option at T1
        option_old = np.maximum(self.params["strike_price"] - bond_old, 0.0)
        option_new = np.copy(option_old)
        option_grid[:, self.T1_index] = option_new

        # Step 3: Backward solve to t=0
        for i in range(self.T1_index - 1, -1, -1):
            # update bond
            A_b, d_b, lu_b = self.construct_bond_matrix(i, bond_old)
            bond_new = solve_banded(lu_b, A_b, d_b)

            # update option
            A_o, d_o, lu_o = self.construct_option_matrix(i, option_old)
            option_new = solve_banded(lu_o, A_o, d_o)

            if use_penalty:
                option_new = self.apply_penalty_method(option_new, bond_new)
            else:
                intrinsic = np.maximum(self.params["strike_price"] - bond_new, 0.0)
                option_new = np.maximum(option_new, intrinsic)

            option_old = np.copy(option_new)
            bond_old = np.copy(bond_new)

            option_grid[:, i] = option_new
            bond_grid[:, i] = bond_new

        # Time vector
        times = np.linspace(0, self.params["bond_maturity"], n_t)

        # Return either the surface grids or just t=0 values
        if return_surface:
            return self.interest_rates, times, option_grid, bond_grid
        else:
            return (
                self.interest_rates,
                option_new,
                option_grid[:, self.T1_index],
                bond_new,
            )


def plot_option_surface(rates, times, option_grid):
    fig = go.Figure(
        data=[
            go.Surface(
                z=option_grid.T,  # transpose so x=rates, y=times
                x=rates,
                y=times,
                colorscale="Viridis",
            )
        ]
    )

    fig.update_layout(
        title="American Put Option Value Surface B(r, t)",
        scene=dict(
            xaxis_title="Interest Rate r",
            yaxis_title="Time t",
            zaxis_title="Option Value",
        ),
        template="plotly_white",
    )

    fig.write_html("american_put_surface.html")
    print("Saved surface plot to american_put_surface.html. Open in Firefox to view.")
    fig.show()


def main():

    params = {
        "kappa": 0.1,
        "theta": 0.05,
        "sigma": 0.02,
        "beta": 0.5,
        "mu": 0.0,
        "C": 2.0,
        "alpha": 0.0,
        "bond_face_value": 100.0,
        "strike_price": 95.0,
        "bond_maturity": 5.0,
        "option_maturity": 2.0,
        "max_int_rate": 0.2,
        "n_int_rate_steps": 200,
        "n_time_steps": 500,
    }

    # instantiate solver
    solver = FiniteDiffAmericanOption(params)

    # run model
    rates, times, option_grid, bond_grid = solver.run(
        return_surface=True, use_penalty=False
    )
    plot_option_surface(rates, times, option_grid)


if __name__ == "__main__":
    main()
