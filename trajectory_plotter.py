from model_scripts import plot_trajectories as pt
from common_scripts.RFP_initialization import create_rfp
from model_scripts.RFP_operational_environment import RFPOperationalEnv
import numpy as np
import matplotlib.pyplot as plt

rfp = create_rfp()
planning_horizon = 4 * 24
decision_horizon = 24
env = RFPOperationalEnv(rfp=rfp, decision_horizon=decision_horizon, planning_horizon=planning_horizon)
experiment_name = "test_short_planning_target"
trajectories = pt.load_trajectories(experiment_name)
plot_name = experiment_name + '_ammonia_state'


trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'state',
                                **{'plot_mask': np.array([0,1,1,1]).astype(bool),}
                                )
plt.ylabel(r"tons NH$_3$")
plt.xlabel("Days")
plt.xlim(0, len(trajectories[0].reward))
# plt.axhline(env.state_space.high[2], label="Ammonia contract quantity")
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}.png')
plt.close()

plot_name = experiment_name + '_hydrogen_state'
trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'state',
                                **{'plot_mask': np.array([1,0,0,0]).astype(bool),}
                                )
plt.ylabel(r"tons H$_2$")
plt.xlabel("Days")
plt.xlim(0, len(trajectories[0].reward))
plt.axhline(env.state_space.high[0], label=r"H$_2$ storage capacity", color='black')
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}_1.png')
plt.close()


plot_name = experiment_name + '_action'
trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'action',
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}.png')
plt.close()

trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'action',
                                **{'plot_mask': np.array([1,0,0,0,0]).astype(bool),}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}_grid_sale.png')
plt.close()


trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'action',
                                **{'plot_mask': np.array([0,1,0,0,0]).astype(bool),}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}_elec_power.png')
plt.close()

trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'action',
                                **{'plot_mask': np.array([0,0,1,0,0]).astype(bool),}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}_pipeline_flow.png')
plt.close()

plot_name = experiment_name + '_balancing'
trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'env_info',
                                **{'env_info_keys': ['balancing'],}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}.png')
plt.close()

plot_name = experiment_name + '_soc_h2'
trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'env_info',
                                **{'env_info_keys': ['soc_h2'],}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}.png')
plt.close()

plot_name = experiment_name + '_technical_violation_cost'
trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'env_info',
                                **{'env_info_keys': ['technical_violation_cost'],}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}.png')
plt.close()

plot_name = experiment_name + '_electricity_price'
trajectory = pt.plot_trajectory(env,
                                trajectories[0],
                                'env_info',
                                **{'env_info_keys': ['electricity_price'],}
                                )
plt.legend()
plt.savefig(f'trajectory_plots/{plot_name}.png')
plt.close()