# Add a harder task

1. Open harness/tasks/specs.py.
2. Write a generator returning a TaskSpec, decorated with @register_task('name'):

       @register_task("my_task")
       def make_my_task(seed=0, difficulty=0.5, **kw):
           ...build objects/goals/obstacles...
           return TaskSpec(kind="my_task", description="...", objects=..., goals=..., params=...)

3. The env must know how to score the kind: add a branch in the success check
   (TabletopEnv._check_success) and reward (_primary_cost). For a new 3D kind
   implement it in the Franka backend instead.

## 3D (Franka) tasks

The TaskSpec schema is the same; use 3D positions in objects/goals/ee_target.
Provide the same object-aware query API on the Franka env so tools work.

## Difficulty

Scale distances, obstacle sizes, or object counts with the difficulty arg, and
seeded RNG (np.random.default_rng(seed)) so instances are reproducible.
