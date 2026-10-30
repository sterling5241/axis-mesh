# Axis Twist Mesh for Kalico

Per-point probe correction for [Kalico](https://github.com/KalicoCrew/kalico).

Scanning and offset probes (Cartographer, Beacon, eddy, inductive) read the bed through a coil that sits beside the nozzle. Gantry twist, rail tilt and bed effects make them read slightly differently from where the nozzle really is, and that error ends up in every bed mesh.

Axis Twist Mesh measures that error at every point of a grid against a reference that touches the bed at the nozzle. It saves the result as a map and corrects the probe's readings with it from then on. Your bed mesh then comes out as if the reference had measured it.

It does the same job as Kalico's mesh-based `[axis_twist_compensation]` and stores its maps in the same format. It adds:

- **Calibration both probes at each point.** Both probes measure a point before moving on, so temperature drift during a run doesn't turn into a fake slope across the bed.
- **3 agreeing readings per point.** A single bad tap can't get into the map.
- **Warnings.** It flags edge rows and columns where an eddy probe misreads near the plate edge, and single outlier points.
- **A report.** It shows X twist, Y tilt, and the pattern left over after both.
- **Cartographer and Beacon support**, including scan-only correction and single-probe calibration.
- **A self-check** that warns at startup if corrections stop reaching the probe.

## Supported setups

| Reference ("true" reading) | Corrected probe |
|---|---|
| Load cell (`load_cell_probe`) | Cartographer, Beacon, or any Kalico probe |
| Any other registered probe | Cartographer, Beacon, or any Kalico probe |
| Cartographer touch (single probe) | Cartographer scan |
| Beacon contact (single probe) | Beacon proximity |
| Paper test (`TESTZ` / `ACCEPT`) | Any of the above |

More than one probe can be corrected at once; each gets its own map. Rectangular and round beds are supported.

Not supported: eddy-ng (its tap re-homes Z on every tap).

## Install

Copy `axis_twist_mesh.py` into Kalico's extras folder, then restart Klipper:

```
cp axis_twist_mesh.py ~/klipper/klippy/extras/
```

The folder is `~/kalico/klippy/extras/` if Kalico is installed under that name.

## Configuration

Minimal:

```ini
[axis_twist_mesh]
mesh_min: 20, 45
mesh_max: 195, 188
```

Load cell plus Cartographer, 11×11:

```ini
[axis_twist_mesh]
reference: load_cell
correction_probe: cartographer
mesh_min: 20, 45
mesh_max: 195, 188
probe_count: 11, 11

# Cartographer only applies corrections when this section exists
[axis_twist_compensation]
calibrate_start_x: 20
calibrate_end_x: 195
calibrate_y: 116
```

The `calibrate_*` lines are only needed on current Kalico; Kalico builds with mesh-based `[axis_twist_compensation]` don't use them.

| Option | Default | Notes |
|---|---|---|
| `reference` | `manual` | `manual` (paper), `load_cell`, a probe name from `LIST_PROBES`, or `touch` (Cartographer touch / Beacon contact). `single_probe: True` is the older name for `touch`. |
| `correction_probe` | Kalico's default probe | One or more probe names, comma separated, e.g. `cartographer, eddy`. |
| `mesh_min`, `mesh_max` | required | Bed area to calibrate, in bed coordinates. Every point must be reachable by both the nozzle and the probe. |
| `probe_count` | `3, 3` | Points per axis. |
| `mesh_radius` | | Round bed instead of `mesh_min` / `mesh_max` / `probe_count`. Use with `mesh_origin` (default `0, 0`) and `round_probe_count` (default `5`, must be odd). |
| `algorithm` | `auto` | Interpolation, as in bed_mesh: `auto`, `bicubic`, `lagrange` or `direct`. |
| `mesh_pps` | `2, 2` | Interpolated points between grid points. |
| `bicubic_tension` | `0.2` | |
| `tap_tolerance` | `0.010` | Spread allowed between the 3 agreeing readings (mm). |
| `max_taps` | `7` | Readings per point before giving up. |
| `tap_retract` | `1.0` | Lift between readings (mm). |
| `horizontal_move_z` | `5` | Travel height (mm). |
| `speed`, `lift_speed` | `50`, `10` | mm/s |
| `max_compensation` | `0.5` | A map with a larger correction is refused (mm). |
| `apply_offset` | `False` | Also correct the constant offset between the two probes. Normally left off; Z homing and your Z offset handle it. |

## Usage

```
G28
AXIS_TWIST_MESH_CALIBRATE
SAVE_CONFIG
```

Heat-soak first, at the temperature you print at. With `reference: touch`, the nozzle must be below the probe's touch temperature limit.

`CALIBRATE` probes every point, prints the report, saves the map and applies it. `SAVE_CONFIG` keeps it after a restart. From then on every reading from the corrected probe gets the correction: `PROBE`, bed meshes, and Cartographer's and Beacon's own scan meshes. Bed meshes made before calibrating still contain the error, so mesh again afterwards.

Read the report before relying on the map. Edge warnings usually mean the grid reaches too close to the plate edge; move `mesh_min` / `mesh_max` inward. A suspect point usually means a bad reading; re-run.

### Commands

| Command | |
|---|---|
| `AXIS_TWIST_MESH_CALIBRATE [PROBE=] [REFERENCE=] [PROBES=both\|main\|correction]` | Calibrate a probe (default: the first `correction_probe`). `REFERENCE=` overrides the reference for one run. `PROBES=main` / `correction` re-probes one side of the last run. Other parameters (e.g. `PROBE_SPEED=`) are passed to the probes. |
| `AXIS_TWIST_MESH_CLEAR [PROBE=]` | Turn compensation off, for all probes or one. |
| `AXIS_TWIST_MESH_PROFILE LOAD=<probe>` | Turn a probe's saved map back on. |
| `AXIS_TWIST_MESH_PROFILE REMOVE=<probe>` | Delete a probe's saved map (then `SAVE_CONFIG`). |
| `AXIS_TWIST_MESH_TABLE [PROBE=] [FORMAT=grid\|twist]` | Print a map. `twist` prints the X-only equivalent as old-style `z_compensations`. |
| `AXIS_TWIST_MESH_ANALYZE` | Repeat the last calibration report. |
| `AXIS_TWIST_MESH_IMPORT [PROBE=]` | Turn old `z_compensations` / `zy_compensations` lines into a map. |
| `AXIS_TWIST_MESH_STATUS` | Configuration, probes found, maps saved and active. |
| `AXIS_TWIST_MESH_CHECK` | Verify the maps actually reach each probe. |

A paper-test calibration continues point by point: the probe measures, then lower the nozzle onto the paper with `TESTZ Z=...` and send `ACCEPT`. `ABORT` cancels and leaves the current map in place.

## How it works

**Calibration.** At each grid point the reference measures the bed, then the corrected probe measures the same spot. The difference between them, minus its average, is the map; the average is the constant offset between the probes. The report splits the difference into an X curve, a Y curve and what's left over.

**Storage.** Each probe's map is saved as `[axis_twist_compensation <probe>]` in the format Kalico's mesh-based axis twist compensation uses, so that Kalico reads the same data. Probes Kalico can't hold a map for, because they aren't in its probe list (Beacon), use the same format under `[axis_twist_mesh <probe>]`. Profiles from earlier versions of this add-on are converted at startup; run `SAVE_CONFIG` once to keep the conversion.

**Applying.** Kalico's probe code, Cartographer and Beacon all call `axis_twist_compensation.get_z_compensation_value()` for every reading.
- **Kalico builds with per-probe maps:** the add-on gives Kalico its maps, and Kalico corrects those probes itself. The add-on only wraps that function for readings Kalico can't attribute: Beacon, and Cartographer when it isn't the default probe.
- **Current Kalico:** the add-on wraps the function. It works out which probe is reading and returns that probe's map.
- **Either way:** the reference probe's readings are never changed, and nothing is applied twice.

**Self-check.** An add-on can stop working after a Kalico or plugin update without any error. At every startup the add-on tests each probe's real correction path. If the correction no longer arrives, it prints a `!!` warning on the console.

## Tests

`test_axis_twist_mesh.py` runs the add-on against Kalico's real config parser, probe list, G-code parser, bed_mesh interpolation and axis twist compensation code, with simulated probes, toolhead, Cartographer and Beacon:

```
KALICO=~/kalico python -m pytest -q test_axis_twist_mesh.py
```

Tests that need mesh-based axis twist compensation are skipped on Kalico versions without it. To run them, point `KALICO` at a checkout that has it.

## Limitations

- **Hardware testing:** the Cartographer and Beacon integrations are tested against simulations built from their source, not on hardware. After the first calibration, run `AXIS_TWIST_MESH_CHECK`.
- **Cartographer on mesh-based Kalico:** Cartographer's current release fails to start on Kalico builds with mesh-based axis twist compensation whenever an `[axis_twist_compensation]` section exists. That's a Cartographer issue; it needs an update.
- **Paper test:** sending `ACCEPT` without having moved Z with `TESTZ` cancels the whole run, not just that point. That's how Kalico's paper-test helper works.
