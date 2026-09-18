# Corridor Simulator Specification

This file specifies the closed periodic simulator. The open-boundary simulator
is specified separately in [`OPEN_SYSTEM_SPEC.md`](OPEN_SYSTEM_SPEC.md).

## Implemented abstractions

The baseline simulator models AGVs constrained to a one-way rectangular route.
Fleet dynamics evolve in longitudinal route coordinate \(s\), and
`RectangularLoop` maps \(s\) to \((x,y,\psi)\) for 2D output. Milestone 7 adds
`network_simulation.py`: several named curvature-aware routes can now carry
interacting finite fleets through merges and intersections with shared
reservation zones. Both remain fixed-path abstractions; neither is presented as
free-space navigation or a grid planner.

At every fixed step, fleet operations are vectorized over robots. The time loop
remains explicit so interventions have ordered causal semantics. For a fixed
route order, finding each leader is \(O(N)\), not \(O(N^2)\).

## Dynamics and shield

For each robot, the simulator computes safe-speed caps against:

1. its immediate leader, treated conservatively as stationary at its current
   position; and
2. an occupied crossing, visible only to the nearest approaching robot.

The second assumption represents line-of-sight occlusion and makes downstream
effects propagate through robots rather than allowing every robot to respond
directly to the same pedestrian. It must become an experimental factor in a
later visibility ablation.

Acceleration is clipped to \([-b,a]\), and position is integrated with constant
acceleration over the step. The discrete shield solves

\[
\tfrac12(v_k+v_{k+1})\Delta t+d^{\mathrm{stop}}(v_{k+1})
\le d_{\mathrm{free},k}
\]

for the largest admissible next speed, verifies that it is reachable under
braking \(b\), and asserts the stopping invariant again after integration. The
configuration requires the modeled reaction time to be at least one
integration step and the sensor range to cover the desired-speed stopping
distance.

An exogenous crossing may become occupied only when the nearest approaching
robot is outside its stopping distance. An unsafe schedule raises
`UnsafeCrossingScheduleError`; it is not silently repaired. Robot overlap or
entry into an occupied crossing raises an assertion immediately.

## Intervention event

An intervention begins when either the safe-speed cap or realized speed is
below desired speed by more than the configured tolerance. It remains active
through recovery until desired speed is restored. Its severity-weighted loss is

\[
D_e=\int_e \left(1-\frac{v_i(t)}{v_{\mathrm{des}}}\right)dt.
\]

This treats a complete stop as severity one and a partial slowdown
proportionally.

## Direct causal attribution

If robot \(j\)'s event starts while its immediate leader \(i\) has an active
event, the event on \(i\) is the direct parent. If the occupied crossing is the
active limiting constraint, the new event is a primary. Every causal event
stores its parent, primary root, generation, and start position. A primary also
stores the crossing index. Start position supports spatial type models without
reconstructing state from a strided trajectory log.

This definition counts one generation only. `estimate_horizon_branching` is an
explicitly correlational ablation: it counts all later interventions inside a
window and therefore mixes generations and double-counts events.

## Paired rollout

`run_paired_rollout` starts treated and no-crossing simulations from identical
states. The reported counterfactual severity and travel losses are
nonnegative treated-minus-control differences by robot. The present dynamics
are deterministic; the pairing contract is already in place for future common
random numbers in task and pedestrian processes.

## Validation already automated

- stopping-distance and safe-speed inversion;
- no-event equilibrium motion;
- rejection of unsafe crossing activation;
- exact causal chain and parent IDs;
- bitwise deterministic replay;
- zero collision and crossing violations over a speed-density sweep;
- direct versus horizon attribution behavior;
- pooled training and held-out progeny evaluation;
- finite-depth versus infinite-resolvent prediction with temporal and fleet-
  boundary censoring separated;
- convergence of event count, severity, traversal loss, and event timing under
  decreasing integration steps.

## Known limitations and next additions

- Route-coordinate stopping and a global center-distance check are the
  collision abstractions; exact oriented-rectangle contact is not modeled.
- The baseline corridor remains one lane, while the network layer supports
  multiple named routes and shared conflict zones but not arbitrary rerouting.
- Merge/intersection reservations are implemented; an optimizing scheduler and
  zone-level reconfiguration remain future work.
- Crossing visibility is nearest-robot-only and has not been ablated.
- Cluster-bootstrap intervals and repeated-dataset calibration are implemented,
  but the registered paper-profile sample sizes have not yet been executed.
- A robot has one immediate leader at an instant, but an event can rarely
  parent more than one later event. The current serial layout therefore permits
  temporal fan-out, although the generated quick sweep does not show enough of
  it for a robust supercritical estimate. A merge or reservation-network layout
  is now implemented, while confidence-certified transition evidence remains a
  paper-profile requirement.

## Loaded route-network semantics

Each network robot stores a route index, arc progress, speed, job identity,
physical robot identity, severity accumulator, and active intervention ID.
Nominal speed is interpolated from `plan_kinematic_speed_profile`, so curvature,
yaw rate, lateral acceleration, wheel speed, acceleration, and braking affect
travel time before any safety intervention is introduced.

A conflict zone has one owner. The closest upstream robot receives ownership;
the closest robot on every non-owner stream stops at the zone entry, and its
followers react through ordinary leader shields. On an aligned shared merge
segment, robots on different route namespaces can become proximate leaders.
The simulator additionally checks all robot centers globally after every step.

Guarded route disturbances use the same two-phase principle as open-corridor
crossings. Robots already too close to stop clear the location; the nearest
robot that can stop becomes the gate; occupation begins only when the protected
interval is empty and the nearest approach is safe. Entry into an active
interval fails immediately.

The network event type defaults to route index. Experiments may retype the same
events by severity before inference. Planned curvature slowdowns never start an
intervention: an event begins only when a shield/reservation cap falls below the
nominal kinematic profile, and it remains active through recovery to that
profile.

Every retained network run asserts:

- exact job mass balance;
- exact finite-fleet balance, including returning robots;
- same-route next-step stopping invariance;
- zero global physical overlap;
- at most one conflict-zone occupant and correct ownership;
- zero active-disturbance intrusion;
- yaw-rate, lateral-acceleration, and wheel-speed limits.

For N scaling, `extend_network_map` adds straight approach and exit storage
without uniformly scaling away the central turn/conflict geometry. Experiments
still report actual WIP, robots per metre of lane, and mean lane headway;
configured fleet size is never used as a proxy for density.

## Finite-chain cohort rule

The closed experiment follows a primary cascade until it resolves or reaches
all \(N\) unique robots. A resolved chain of size \(L<N\) contributes \(L-1\)
propagation successes and one observed failure. A chain reaching \(N\)
contributes \(N-1\) successes and no artificial failure at the physical
boundary. A temporally censored chain below \(N\) is indeterminate. This rule
prevents a short simulation horizon from masquerading as finite-fleet
regularization. If every training chain reaches \(N\), however,
\(\widehat\phi=1\) and \(M_N(1)=N\) identically; that case is labeled
`descriptive_not_predictive` rather than reported as zero-error validation.
