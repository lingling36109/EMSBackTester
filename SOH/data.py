import pybamm
import random
import pandas as pd

def generate_constant_current_cycle():
    hours = random.uniform(0.5, 1.5)
    duration = f"{hours:.2f} hours"
    return [
        f"Discharge at 1 A for {duration}",
        "Rest for 30 minutes",
        f"Charge at 1 A for {duration}",
        "Rest for 30 minutes",
    ]


if __name__ == '__main__':
    model = pybamm.lithium_ion.DFN(
        {
            "SEI": "solvent-diffusion limited",
            "SEI porosity change": "true",
            "lithium plating": "partially reversible",
            "lithium plating porosity change": "true",  # alias for "SEI porosity change"
            "particle mechanics": ("swelling and cracking", "swelling only"),
            "SEI on cracks": "true",
            "loss of active material": "stress-driven",
        }
    )
    param = pybamm.ParameterValues("OKane2022")
    var_pts = {
        "x_n": 5,  # negative electrode
        "x_s": 5,  # separator
        "x_p": 5,  # positive electrode
        "r_n": 30,  # negative particle
        "r_p": 30,  # positive particle
    }

    cycle_number = 1000
    exp = pybamm.Experiment(
        [
            "Hold at 4.2 V until C/100",
            "Rest for 4 hours",
            "Discharge at 0.1C until 2.5 V",
            "Charge at 0.1C until 4.2 V",
            "Hold at 4.2 V until C/100",
        ]
        + [
            (
                "Discharge at 1C until 3.5 V",
                "Charge at 1C until 4.2 V",
            )
        ]
        * cycle_number
        + ["Discharge at 0.1C until 2.5 V"],  # final capacity check
    )
    solver = pybamm.IDAKLUSolver()
    sim = pybamm.Simulation(
        model, parameter_values=param, experiment=exp, solver=solver, var_pts=var_pts
    )
    # sim = pybamm.Simulation(
    #     model, parameter_values=param, experiment=exp, solver=solver
    # )
    sol = sim.solve()

    output_variables = ['Current [A]', 'Terminal voltage [V]', 'Discharge capacity [A.h]', 'Total capacity lost to side reactions [A.h]']
    sim.plot(output_variables=output_variables)

    solution = sim.solution
    data = {'Time [s]': solution.t}
    for name in output_variables:
        try:
            y = solution[name].entries
            if y.ndim > 1 and y.shape[1] == 1:
                y = y[:, 0]
            data[name] = y
        except KeyError:
            print(f"Variable {name} not found in solution.")

    df = pd.DataFrame(data)
    df.to_csv("simulation_output.csv", index=False)

    print("Simulation output saved to simulation_output.csv")
