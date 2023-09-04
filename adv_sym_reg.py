import pprint

import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor


def target_f(x):
    return 1/x



if __name__ =='__main__':
    grid = np.linspace(start=1.0,stop=2.0,num=10000).reshape(10000,1)
    y_truth = target_f(grid)

    model = PySRRegressor(
        procs=8,
        populations=16,
        # ^ 2 populations per core, so one is always running.
        population_size=50,
        ncyclesperiteration=500,
        # ^ Generations between migrations.
        niterations=50,  # Run forever
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6"
            # Stop early if we find a good and simple equation
        ),
        timeout_in_seconds=60 * 60 * 24,
        # ^ Alternatively, stop after 24 hours have passed.
        maxsize=10,
        # ^ Allow greater complexity.
        maxdepth=10,
        # ^ But, avoid deep nesting.
        binary_operators=["*", "+", "-"],
        unary_operators=["square", "cube", "exp"],
        # constraints={
        #     "exp": 9,
        # },
        # ^ Limit the complexity within each argument.
        # "inv": (-1, 9) states that the numerator has no constraint,
        # but the denominator has a max complexity of 9.
        # "exp": 9 simply states that `exp` can only have
        # an expression of complexity 9 as input.
        # ^ Nesting constraints on operators. For example,
        # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
        complexity_of_operators={"*": 0,"+":0,"-":0, "exp": 1,"square":0,"cube":0},
        # ^ Custom complexity of particular operators.
        complexity_of_constants=2,
        # ^ Punish constants more than variables
        select_k_features=4,
        # ^ Train on only the 4 most important features
        progress=True,
        # ^ Can set to false if printing to a file.
        weight_randomize=0.1,
        # ^ Randomize the tree much more frequently
        cluster_manager=None,
        # ^ Can be set to, e.g., "slurm", to run a slurm
        # cluster. Just launch one script from the head node.
        precision=64,
        # ^ Higher precision calculations.
        warm_start=True,
        # ^ Start from where left off.
        turbo=True,
        # ^ Faster evaluation (experimental)
        julia_project=None,
        # ^ Can set to the path of a folder containing the
        # "SymbolicRegression.jl" repo, for custom modifications.
        update=False,
        # ^ Don't update Julia packages
        # extra_torch_mappings={sympy.cos: torch.cos},
        # ^ Not needed as cos already defined, but this
        # is how you define custom torch operators.
        # extra_jax_mappings={sympy.cos: "jnp.cos"},
        # ^ For JAX, one passes a string.
    )
    model.fit(grid, y_truth)
    print(model.latex_table())
    torch_model = model.pytorch()
    print(torch_model)
    predict = model.predict(grid)
    fig, ax = plt.subplots(1, 1)
    ax.plot(grid, predict, label=r'predict')
    ax.plot(grid, y_truth, label=r'true')
    ax.legend()
    print('max abs error {}'.format(np.max(np.abs(y_truth-predict))))

    plt.show()
