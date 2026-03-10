# Tron1 RL IsaacLab

## Dependencies

### IsaacLab

Use official IsaacSim 5.1.0 and IsaacLab 2.3.0

[IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

### RSL-RL

Use custom RSL-RL in `./rsl_rl`

```sh
pip install -e ./rsl_rl
```

Remove official RSL-RL if it is installed

```sh
pip uninstall rsl-rl-lib
```

### This Project

```sh
pip install -e ./exts/bipedal_locomotion
```

Ignore isolation if an error like `ModuleNotFoundError: No module named 'toml'` appears

```sh
pip install -e ./exts/bipedal_locomotion --no-build-isolation
```

## Runtime

### Training

```sh
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-Blind-Flat-v0 --num_envs 2048 --headless
```

### Playing

```sh
python scripts/rsl_rl/play.py --task Isaac-Limx-WF-Blind-Flat-v0
```
