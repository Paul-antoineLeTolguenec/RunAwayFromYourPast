import numpy as np
from evojax.obs_norm import ObsNormalizer
from evojax.sim_mgr import SimManager
from evojax.task.brax_task import BraxTask
from evojax.policy import MLPPolicy

from evosax import Strategies
from evosax.utils.evojax_wrapper import Evosax2JAX_Wrapper
def get_brax_task(
    env_name = "ant",
    hidden_dims = [32, 32, 32, 32],
):
    train_task = BraxTask(env_name, test=False, task_kwargs={"num_steps": 1000})
    test_task = BraxTask(env_name, test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=hidden_dims,
    )
    return train_task, test_task, policy
train_task, test_task, policy = get_brax_task("ant")
solver = Evosax2JAX_Wrapper(
    Strategies["OpenES"],
    param_size=policy.num_params,
    pop_size=256,
    es_config={"maximize": True,
               "centered_rank": True,
               "lrate_init": 0.01,
               "lrate_decay": 0.999,
               "lrate_limit": 0.001},
    es_params={"sigma_init": 0.05,
    "sigma_decay": 0.999,
    "sigma_limit": 0.01},
    seed=0,
)
obs_normalizer = ObsNormalizer(
    obs_shape=train_task.obs_shape, dummy=not True
)
sim_mgr = SimManager(
    policy_net=policy,
    train_vec_task=train_task,
    valid_vec_task=test_task,
    seed=0,
    obs_normalizer=obs_normalizer,
    pop_size=256,
    use_for_loop=False,
    n_repeats=16,
    test_n_repeats=1,
    n_evaluations=128
)

print(f"START EVOLVING {policy.num_params} PARAMS.")
# Run ES Loop.
for gen_counter in range(1000):
    params = solver.ask()
    scores, _ = sim_mgr.eval_params(params=params, test=False)
    solver.tell(fitness=scores)
    if gen_counter == 0 or (gen_counter + 1) % 50 == 0:
        test_scores, _ = sim_mgr.eval_params(
            params=solver.best_params, test=True
        )
        print(
            {
                "num_gens": gen_counter + 1,
            },
            {
                "train_perf": float(np.nanmean(scores)),
                "test_perf": float(np.nanmean(test_scores)),
            },
        )