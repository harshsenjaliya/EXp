"""Minimal open-system paired experiment."""

from amr_capacity import (
    OpenCorridorConfig,
    analyze_arrival_cohort,
    analyze_operating_window,
    poisson_arrival_times,
    poisson_crossing_requests,
    run_open_paired_rollout,
    summarize_open_counterfactual,
)


config = OpenCorridorConfig(
    corridor_length=30.0,
    desired_speed=1.2,
    acceleration=1.0,
    braking=1.5,
    reaction_time=0.2,
    dt=0.05,
    robot_length=0.8,
    safety_margin=0.2,
    sensor_range=8.0,
    crossing_x=15.0,
)
duration = 240.0
arrivals = poisson_arrival_times(0.65, duration, seed=11)
requests = poisson_crossing_requests(
    0.025, duration, (2.0, 4.0), seed=12, start_time=20.0
)
pair = run_open_paired_rollout(
    config, duration, arrivals, requests, record_stride=5
)
treated = analyze_operating_window(pair.treated, 120.0, 240.0)
cohort = analyze_arrival_cohort(pair.treated, 80.0, 120.0, 100.0)

print("stability evidence:", treated.stability)
print("completion rate:", treated.completion_rate)
print("queue drift (jobs/s):", treated.drift.slope)
print("cohort restricted mean sojourn (s):", cohort.restricted_mean_sojourn_time)
print("paired effects:", summarize_open_counterfactual(pair))
print("job mass residual:", pair.treated.mass_balance_residual)
print("collisions:", pair.treated.collision_count)
