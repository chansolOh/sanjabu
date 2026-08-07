# Scene / PreGrasp / Grasp server control

The same TUI controls three isolated worker commands:

- `SceneGen`: Isaac Sim scene generation environment
- `PreGrasp`: `/home/<user>/ochansol/isaaclab_232/.venv`
- `Grasp`: `/home/<user>/ochansol/isaaclab_232/.venv`

`Generator_client.py` selects the Python environment and script from the
command sent by the server. PreGrasp and Grasp are run with the Isaac Lab
project as their working directory.

## 1. Start the controller server

Run this on `192.168.0.137`:

```bash
cd /home/uon/ochansol/isaac_code/python/sanjabu/2026/scene_generation
/home/uon/ochansol/isaac_code/.venv/bin/python SceneGen_Agent_TUI.py
```

The dataset root used by the TUI is the `self.output_root_path` variable in
`SceneGen_Agent_TUI.py`.

## 2. Start a worker client

Run this on each generation PC after mounting the NAS and synchronizing the
same source paths:

```bash
cd /home/uon/ochansol/isaac_code/python/sanjabu/2026/scene_generation
/home/uon/ochansol/isaac_code/.venv/bin/python -u Generator_client.py
```

The worker registers with the server using its local IP address.

## 3. Start work from the TUI

For one connected PC, edit these columns and press **Play**:

1. `Command`: `SceneGen`, `PreGrasp`, or `Grasp`
2. `Env`
3. `Section`
4. `Platform`
5. `Start`
6. `End`

Recommended pipeline order:

1. Generate scenes with `SceneGen`.
2. Generate schema-compatible hand height maps once with
   `Generate_Hand_Heightmap.py`.
3. Generate missing pre-grasp scenes with `PreGrasp`.
4. Collect missing gripper groups with `Grasp`.

## Resume behavior

- `PreGrasp` scans `pre_grasp/<scene>.json` and starts at the first missing
  scene in the selected range for which `conf/<scene>.json` exists.
- `Grasp` scans `output_grasp/<scene>.json`, compares its gripper models with
  the groups in `pre_grasp/<scene>.json`, and resumes from the first missing
  gripper group.
- A non-zero child-process exit changes the TUI status to `Error` and stops
  automatic retries. The worker terminal contains the full traceback.
- A timeout is treated as a restart request and is retried automatically.

## Direct script execution remains available

PreGrasp can still use the configuration variables at the top of its file, or
it can be called with server-compatible arguments:

```bash
cd /home/uon/ochansol/isaaclab_232
.venv/bin/python -u 2026_Codex/pregrasp/Collect_Hand_PreGrasp_dataset.py \
  --output_root_path /nas/Dataset/Dataset_2026/dataset_v2 \
  --env_name Logistic_site \
  --section_name General_LogisticSite \
  --platform_name conveyor_track_01 \
  --scene_start 0
```

Grasp direct execution:

```bash
cd /home/uon/ochansol/isaaclab_232
.venv/bin/python -u 2026_Codex/Grasp_arg_new_filter.py \
  --root_path /nas/Dataset/Dataset_2026/dataset_v2 \
  --env_name Logistic_site \
  --section_name General_LogisticSite \
  --platform_name conveyor_track_01 \
  --scene_start 0 \
  --pre_grasp_index 0
```

