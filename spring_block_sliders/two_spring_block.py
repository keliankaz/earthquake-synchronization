# %%
import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
import matplotlib.pyplot as plt

class CoupledSpringBlock:
    def __init__(
        self,
        Y: list = [-9., 0., np.log(1.0001), 0.],  # state vector
        k1: float = -1,  # how fault loads 1 and 2
        k2: float = 0.3,  # how 2 affects 1
        k3: float = 0.3,  # how 1 affects 2
        Vpl: callable = lambda t: 1e-9,  # plate loading velocity in m/s
        sigma: np.array = np.array([10, 9]),  # normal stress in MPa
        a: float = 1e-2,  # friction parameters
        b: float = 1e-2 + 5e-3,
        dc: float = 2e-4,  # slip weakening distance in meters
        eta: float = 6,  # radiation damping in MPa-s/m
    ):

        self.Y = np.array(Y)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.sigma = sigma
        self.a = a
        self.b = b
        self.dc = dc
        self.eta = eta
        self.Vpl = Vpl

        self.kernel = np.array([[k1, k2], [k3, k1]])

        self.sol = None

    def check_sol(func):
        def wrapper(self, *args, **kwargs):
            if self.sol is None:
                raise ValueError(
                    "The solution has not been computed yet. Please run the simulation first."
                )
            return func(self, *args, **kwargs)

        return wrapper

    @property
    @check_sol
    def time(self):
        return self.sol.t

    @property
    @check_sol
    def velocity(self):
        return np.array([self.Vpl(t) for t in self.time]) * np.exp(
            self.sol.y[[0, 2], :]
        )

    @property
    @check_sol
    def friction(self):

        y = self.sol.y

        return self.a * y[[0, 2], :] + self.b * y[[1, 3], :]

    @property
    @check_sol
    def slip(self):
        return cumtrapz(self.velocity, self.time, initial=0, axis=1)

    @property
    @check_sol
    def slip_deficit(self):
        Vpl_values = np.array([self.Vpl(t) for t in self.time])
        Vpl_integral = cumtrapz(Vpl_values, self.time, initial=0)
        return -self.slip + Vpl_integral

    @property
    @check_sol
    def stress(self):
        return (
            (0.6 + self.a * self.sol.y[[0, 2], :] + self.b * self.sol.y[[1, 3], :])
            * self.sigma.reshape(-1, 1)
            * np.ones_like(self.time)
        )
        
    def apply_stress_step(self, step):
        self.Y[[0,2]] += step/(self.a * self.sigma)

    def ode_spring_slider(self, t, Y):

        Vpl_t = self.Vpl(t)

        V = Vpl_t * np.exp(Y[[0, 2]])

        th = Y[[1, 3]]

        Yp = np.zeros_like(Y)  # initialize change in state

        dth = (Vpl_t * np.exp(-th) - V) / self.dc  # change in RS state (theta)
        Yp[[1, 3]] = dth


        # calculate the change in velocity
        kv = self.kernel @ (V - Vpl_t)
        Yp[[0, 2]] = (kv - self.b * self.sigma * dth) / (
            self.a * self.sigma + self.eta * V
        )

        return Yp

    def simulate(self, T):
        """Uses solve_ivp to simulate the system"""
        options = {"rtol": 1e-6, "atol": 1e-6, "max_step": 1e8}
        
        if self.sol is not None:
            T0 = self.time[-1]
        else:
            T0 = 0
        
        solution = solve_ivp(
            self.ode_spring_slider, [T0, T0+T], self.Y, method="RK45", **options
        )
        
        # irresponsible concatenation of solutions
        if self.sol is None:
            self.sol = solution
        else:
            self.sol.t = np.concatenate((self.sol.t, solution.t))
            self.sol.y = np.concatenate((self.sol.y, solution.y), axis=1)
        
        return self.sol

    def plot_velocity(self, ax=None):
        """Plots the velocity of the simulation"""

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.time, self.velocity[0, :], label="V1", color="mediumpurple")
        ax.plot(self.time, self.velocity[1, :], label="V2", color="darkseagreen")
        ax.set(
            xlabel="Time",
            ylabel="Velocity",
            yscale="log",
        )

        return ax

    def plot_slip(self, ax=None):
        """Plots the slip of the simulation"""

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.time, self.slip[0, :], label="Slip 1")
        ax.plot(self.time, self.slip[1, :], label="Slip 2")
        ax.set(
            xlabel="Time",
            ylabel="Slip",
            yscale="log",
        )

        return ax

    def plot_stress(self, ax=None):
        """Plots the stress of the simulation"""

        if ax is None:
            fig, ax = plt.subplots(dpi=300, figsize=(6, 2))

        ax.plot(
            self.time, self.stress[0, :], label="Stress 1", color="mediumpurple", lw=0.5
        )
        ax.plot(
            self.time, self.stress[1, :], label="Stress 2", color="darkseagreen", lw=0.5
        )
        ax.set(
            xlabel="Time",
            ylabel="Stress (MPa)",
        )

        return ax

    def plot_results(self, ax=None):
        """Plots the results of the simulation"""

        if ax is None:
            fig, ax = plt.subplots()

        self.plot_velocity(ax)

        return ax

    def plot_orbits(self, ax=None):
        """Plots the orbits of the simulation"""

        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.slip_deficit[0, :],
            self.slip_deficit[1, :],
            s=0.1,
            c=self.time,
            cmap="viridis",
        )
        ax.set_xlabel("Slip Deficit 1")
        ax.set_ylabel("Slip Deficit 2")

        return ax

    def _get_event_times(self, velocity, Veq=1e-3):
        """Returns the times and indices of the events"""

        events = np.where(velocity > Veq)[0]  # Get indices where velocity > Veq

        if len(events) == 0:
            return [], []  # Return empty lists if no events found

        # Find where the events are not consecutive
        split_indices = np.where(np.diff(events) > 1)[0] + 1
        event_groups = np.split(events, split_indices)

        event_times = []
        event_indices = []

        for group in event_groups:
            max_index = group[np.argmax(velocity[group])]
            event_indices.append(max_index)
            event_times.append(self.time[max_index])

        return event_times, event_indices

    def get_events(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the times of the events"""

        return (
            self._get_event_times(self.velocity[0, :])[0],
            self._get_event_times(self.velocity[1, :])[0],
        )


if __name__ == "__main__":

    def time_dependent_Vpl(t):
        return 1e-9 * (1 + 0.5 * np.sin(2 * np.pi * t / (10 * 3.15e7)))

    fig, axs = plt.subplots(1, 2, figsize=(5, 2), sharex=True, sharey=True)
    for i in range(2):
        model = CoupledSpringBlock(
            Vpl=time_dependent_Vpl,
            sigma=np.array([10, 8]),
            Y=[
                np.random.uniform(-10, 0),
                0,
                np.random.uniform(-10, 0),
                0,
            ],  # perturb the initial state
        )

        model.simulate(100 * 3.15e7)
        model.plot_orbits(ax=axs[i])

    plt.tight_layout()

# %%
