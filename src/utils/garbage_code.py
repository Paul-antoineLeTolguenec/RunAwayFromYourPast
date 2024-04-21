# UPDATE RHO
# idx_row, idx_column = torch.where(dones)
# idx_row, idx_column = idx_row.cpu().numpy(), idx_column.cpu().numpy()
# nb_ep = 0
# idx_per_env = np.zeros(args.num_envs, dtype=int)
# for (i,j) in zip(idx_row, idx_column):
#     idx_start = idx_per_env[j]
#     idx_end = i+1
#     obs_rho_terminated.append(obs[idx_start:idx_end,j].unsqueeze(1))
#     actions_rho_terminated.append(actions[idx_start:idx_end,j].unsqueeze(1))
#     logprobs_rho_terminated.append(logprobs[idx_start:idx_end,j].unsqueeze(1))
#     rewards_rho_terminated.append(rewards[idx_start:idx_end,j].unsqueeze(1))
#     dones_rho_terminated.append(dones[idx_start:idx_end,j].unsqueeze(1))
#     values_rho_terminated.append(values[idx_start:idx_end,j].unsqueeze(1))
#     times_rho_terminated.append(times[idx_start:idx_end,j].unsqueeze(1))
#     nb_ep += 1
#     idx_per_env[j] = idx_end + 1
# for j in range(args.num_envs):
#     idx_start = idx_per_env[j]
#     idx_end = args.num_steps
#     obs_rho_not_terminated.append(obs[idx_start:idx_end,j].unsqueeze(1))
#     actions_rho_not_terminated.append(actions[idx_start:idx_end,j].unsqueeze(1))
#     logprobs_rho_not_terminated.append(logprobs[idx_start:idx_end,j].unsqueeze(1))
#     rewards_rho_not_terminated.append(rewards[idx_start:idx_end,j].unsqueeze(1))
#     dones_rho_not_terminated.append(dones[idx_start:idx_end,j].unsqueeze(1))
#     values_rho_not_terminated.append(values[idx_start:idx_end,j].unsqueeze(1))
#     times_rho_not_terminated.append(times[idx_start:idx_end,j].unsqueeze(1))