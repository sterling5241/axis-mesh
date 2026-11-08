# Map-based probe location bias compensation
#
# Copyright (C) 2022 Jeremy Tan <jeremytkw98@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import math

from . import bed_mesh, manual_probe
from . import probe as probe_module

MESH_VERSION = 1


def verify_mesh_algorithm(params, error, algorithms=("lagrange", "bicubic")):
    x_pps = params["mesh_x_pps"]
    y_pps = params["mesh_y_pps"]
    if params["algo"] not in algorithms:
        raise error("bed_mesh: Unknown algorithm <%s>" % (params["algo"]))
    max_probe_cnt = max(params["x_count"], params["y_count"])
    min_probe_cnt = min(params["x_count"], params["y_count"])
    if max(x_pps, y_pps) == 0:
        params["algo"] = "direct"
    elif params["algo"] == "lagrange" and max_probe_cnt > 6:
        raise error(
            "bed_mesh: cannot exceed a probe_count of 6 when using "
            "lagrange interpolation. Configured Probe Count: %d, %d"
            % (params["x_count"], params["y_count"])
        )
    elif params["algo"] == "bicubic" and min_probe_cnt < 4:
        if max_probe_cnt > 6:
            raise error(
                "bed_mesh: invalid probe_count option when using bicubic "
                "interpolation.  Combination of 3 points on one axis with "
                "more than 6 on another is not permitted. "
                "Configured Probe Count: %d, %d"
                % (params["x_count"], params["y_count"])
            )
        params["algo"] = "lagrange"


def verify_mesh_params(params, error):
    if not all(
        math.isfinite(v) for v in params.values() if isinstance(v, (int, float))
    ):
        raise error("Mesh parameters must be finite")
    minimum_count = 2 if params["algo"] == "direct" else 3
    if (
        params["max_x"] <= params["min_x"]
        or params["max_y"] <= params["min_y"]
        or min(params["x_count"], params["y_count"]) < minimum_count
        or min(params["mesh_x_pps"], params["mesh_y_pps"]) < 0
        or not 0.0 <= params["tension"] <= 2.0
    ):
        raise error("Invalid mesh parameters")
    if (
        params["algo"] == "direct"
        and max(params["mesh_x_pps"], params["mesh_y_pps"]) != 0
    ):
        raise error("Direct meshes require mesh_pps: 0")
    verify_mesh_algorithm(params, error, ("lagrange", "bicubic", "direct"))


def pad_round_mesh(probed_matrix, x_count, error):
    for row in probed_matrix:
        row_size = len(row)
        if not row_size & 1:
            msg = "bed_mesh: incorrect number of points sampled on X\n"
            msg += "Probed Table:\n"
            msg += str(probed_matrix)
            raise error(msg)
        buf_cnt = (x_count - row_size) // 2
        if buf_cnt == 0:
            continue
        left_buffer = [row[0]] * buf_cnt
        right_buffer = [row[row_size - 1]] * buf_cnt
        row[0:0] = left_buffer
        row.extend(right_buffer)


class CompensationMesh:
    def __init__(self, config):
        self.radius = config.getfloat("mesh_radius", None, above=0.0)
        if self.radius is None:
            minimum = config.getfloatlist("mesh_min", count=2)
            maximum = config.getfloatlist("mesh_max", count=2)
            x_count, y_count = bed_mesh.parse_config_pair(
                config, "probe_count", 3, minval=3
            )
            origin = (0.0, 0.0)
        else:
            self.radius = math.floor(self.radius * 10) / 10
            origin = config.getfloatlist("mesh_origin", (0.0, 0.0), count=2)
            x_count = y_count = config.getint("round_probe_count", 5, minval=3)
            if not x_count & 1:
                raise config.error("round_probe_count must be odd")
            minimum = (-self.radius, -self.radius)
            maximum = (self.radius, self.radius)
        if any(b <= a for a, b in zip(minimum, maximum)):
            raise config.error("Invalid compensation mesh bounds")
        dx = math.floor((maximum[0] - minimum[0]) / (x_count - 1) * 100) / 100
        dy = math.floor((maximum[1] - minimum[1]) / (y_count - 1) * 100) / 100
        if min(dx, dy) < 1.0:
            raise config.error(
                "Compensation mesh points must be at least 1mm apart"
            )
        if self.radius is not None:
            dy = dx
            radius = (x_count // 2) * dx
            minimum = (-radius, -radius)
        min_x, min_y = (a + b for a, b in zip(minimum, origin))
        pps = bed_mesh.parse_config_pair(config, "mesh_pps", 2, minval=0)
        self.params = {
            "min_x": min_x,
            "max_x": min_x + (x_count - 1) * dx,
            "min_y": min_y,
            "max_y": min_y + (y_count - 1) * dy,
            "x_count": x_count,
            "y_count": y_count,
            "mesh_x_pps": pps[0],
            "mesh_y_pps": pps[1],
            "algo": config.get("algorithm", "lagrange").strip().lower(),
            "tension": config.getfloat(
                "bicubic_tension", 0.2, minval=0.0, maxval=2.0
            ),
        }
        verify_mesh_params(self.params, config.error)
        self.settings = {
            "mesh_pps": pps,
            "algorithm": config.get("algorithm", "lagrange").strip().lower(),
            "bicubic_tension": self.params["tension"],
        }
        if self.radius is None:
            self.settings.update(
                mesh_min=minimum,
                mesh_max=maximum,
                probe_count=(x_count, y_count),
            )
        else:
            self.settings.update(
                mesh_radius=self.radius,
                mesh_origin=origin,
                round_probe_count=x_count,
            )
        self.points = []
        self.indices = []
        for y in range(y_count):
            columns = range(x_count) if not y % 2 else reversed(range(x_count))
            for x in columns:
                px, py = min_x + x * dx, min_y + y * dy
                if (
                    self.radius is not None
                    and math.hypot(minimum[0] + x * dx, minimum[1] + y * dy)
                    > self.radius + 1.0e-9
                ):
                    continue
                self.points.append((px, py))
                self.indices.append((y, x))

    @staticmethod
    def build(params, points, name, error):
        if (
            len(points) != params["y_count"]
            or any(len(row) != params["x_count"] for row in points)
            or any(not math.isfinite(z) for row in points for z in row)
        ):
            raise error("Invalid compensation mesh points")
        mesh = bed_mesh.ZMesh(dict(params), name)
        try:
            mesh.build_mesh(points)
        except bed_mesh.BedMeshError as e:
            raise error(str(e))
        return mesh

    def build_difference(self, target, reference, name, error):
        if len(target) != len(self.points) or len(reference) != len(
            self.points
        ):
            raise error("Incomplete compensation calibration")
        differences = [r - t for t, r in zip(target, reference)]
        mean = sum(differences) / len(differences)
        rows = [
            [None] * self.params["x_count"]
            for _ in range(self.params["y_count"])
        ]
        for (y, x), value in zip(self.indices, differences):
            rows[y][x] = value - mean
        if self.radius is not None:
            rows = [
                [value for value in row if value is not None] for row in rows
            ]
            pad_round_mesh(rows, self.params["x_count"], error)
        return self.build(self.params, rows, name, error)


class _CalibrationConfig:
    def __init__(self, config, base):
        self.config = config
        self.base = base

    def __getattr__(self, name):
        option_getters = (
            "get",
            "getint",
            "getfloat",
            "getboolean",
            "getchoice",
            "getlist",
            "getlists",
            "getintlist",
            "getfloatlist",
        )
        if name not in option_getters:
            return getattr(self.config, name)

        def get(option, *args, **kwargs):
            source = self.config
            rectangular = option == "mesh_radius" and any(
                self.config.get(key, None) is not None
                for key in ("mesh_min", "mesh_max")
            )
            if not rectangular and self.config.get(option, None) is None:
                source = self.base
            return getattr(source, name)(option, *args, **kwargs)

        return get


_MOVEMENT_OPTIONS = (
    "speed",
    "horizontal_move_z",
    "horizontal_z_clearance",
    "adaptive_horizontal_move_z",
    "min_horizontal_move_z",
    "alternate_probe_direction",
    "start_reverse",
)
_SHARED_OPTIONS = (
    "reference_probe",
    "mesh_min",
    "mesh_max",
    "probe_count",
    "mesh_radius",
    "mesh_origin",
    "round_probe_count",
    "mesh_pps",
    "algorithm",
    "bicubic_tension",
    "use_probe_xy_offsets",
    *_MOVEMENT_OPTIONS,
)
_LEGACY_OPTIONS = (
    "z_compensations",
    "compensation_start_x",
    "compensation_end_x",
    "zy_compensations",
    "compensation_start_y",
    "compensation_end_y",
    "calibrate_start_x",
    "calibrate_end_x",
    "calibrate_y",
    "calibrate_start_y",
    "calibrate_end_y",
    "calibrate_x",
)


def _import_legacy_mesh(config, selected_probe):
    curves = []
    for axis, option in (("x", "z_compensations"), ("y", "zy_compensations")):
        values = config.getlists(option, [], parser=float)
        if not values:
            curves.append(None)
            continue
        start = config.getfloat("compensation_start_" + axis, None)
        end = config.getfloat("compensation_end_" + axis, None)
        if start is None or end is None:
            raise config.error(
                option
                + " requires compensation_start_"
                + axis
                + " and compensation_end_"
                + axis
            )
        if (
            len(values) < 2
            or start == end
            or not all(math.isfinite(v) for v in [start, end, *values])
        ):
            raise config.error("Invalid " + option)
        if end < start:
            start, end, values = end, start, values[::-1]
        curves.append((start, (end - start) / (len(values) - 1), values))
    if not any(curves):
        return None
    printer = config.get_printer()
    toolhead = printer.lookup_object("toolhead")
    status = toolhead.get_kinematics().get_status(
        printer.get_reactor().monotonic()
    )
    minimum, maximum = status["axis_minimum"], status["axis_maximum"]
    coordinates, corrections = [], []
    for axis, curve in enumerate(curves):
        low, high = minimum[axis], maximum[axis]
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise config.error(
                "Legacy compensation requires finite XY work-area bounds"
            )
        if curve is None:
            coordinates.append([low, high])
            corrections.append([0.0, 0.0])
            continue
        start, spacing, values = curve
        first = math.floor((low - start) / spacing)
        last = math.ceil((high - start) / spacing)
        coordinates.append(
            [start + index * spacing for index in range(first, last + 1)]
        )
        axis_values = []
        for index in range(first, last + 1):
            edge = max(0, min(index, len(values) - 2))
            axis_values.append(
                values[edge]
                + (index - edge) * (values[edge + 1] - values[edge])
            )
        corrections.append(axis_values)
    offsets = selected_probe.get_offsets()
    params = {
        "min_x": coordinates[0][0] + offsets[0],
        "max_x": coordinates[0][-1] + offsets[0],
        "min_y": coordinates[1][0] + offsets[1],
        "max_y": coordinates[1][-1] + offsets[1],
        "x_count": len(coordinates[0]),
        "y_count": len(coordinates[1]),
        "mesh_x_pps": 0,
        "mesh_y_pps": 0,
        "algo": "direct",
        "tension": 0.2,
    }
    verify_mesh_params(params, config.error)
    return CompensationMesh.build(
        params,
        [[x + y for x in corrections[0]] for y in corrections[1]],
        "axis_twist_compensation " + selected_probe.probe_name,
        config.error,
    )


class Calibration:
    def __init__(self, config, compensation):
        self.compensation = compensation
        self.printer = compensation.printer
        self.layout = CompensationMesh(config)
        if config.get("calibration_points", None) is not None:
            raise config.error(
                "Calibration points are generated from mesh settings"
            )
        self.config = config
        self.gcmd = None
        self.active = False
        self.previous_mesh = None
        for event in (
            "gcode:command_error",
            "klippy:shutdown",
            "klippy:disconnect",
        ):
            self.printer.register_event_handler(event, self.cancel)

    def is_calibrating(self):
        return self.active

    def create_helper(self, callback):
        helper = probe_module.ProbePointsHelper(
            self.config,
            callback,
            default_points=self.layout.points,
            option_name="calibration_points",
            use_offsets=True,
            enable_horizontal_z_clearance=True,
        )
        helper.use_xy_offsets(True)
        return helper

    def create_command(self, gcmd, method, probe_name):
        params = dict(gcmd.get_command_parameters())
        params["METHOD"] = method
        if probe_name is None:
            params.pop("PROBE", None)
        else:
            params["PROBE"] = probe_name
        gcode = self.printer.lookup_object("gcode")
        return gcode.create_gcode_command(
            gcmd.get_command(), gcmd.get_commandline(), params
        )

    def start(self, gcmd, reference):
        if self.active:
            raise gcmd.error("An axis twist calibration is already in progress")
        manual_probe.verify_no_manual_probe(self.printer)
        self.gcmd = gcmd
        self.active = True
        self.previous_mesh = self.compensation.mesh
        self.compensation.mesh = None
        try:
            gcmd.respond_info(
                "Axis twist compensation: measuring %s mesh"
                % self.compensation.probe_name
            )
            target = self.get_values(
                gcmd,
                self.measure_mesh(gcmd, self.compensation.probe_name),
            )
            if reference is None:
                gcmd.respond_info(
                    "Axis twist compensation: measuring manual mesh. "
                    "Use TESTZ and ACCEPT at each point, or ABORT."
                )
                self.measure_manual_mesh(
                    gcmd,
                    lambda result: self.finish(gcmd, target, result, "manual"),
                )
            else:
                gcmd.respond_info(
                    "Axis twist compensation: measuring %s mesh"
                    % reference.probe_name
                )
                result = self.measure_mesh(gcmd, reference.probe_name)
                self.finish(gcmd, target, result, reference.probe_name)
        except Exception:
            self.cancel()
            raise

    def measure_mesh(self, gcmd, probe_name):
        results = []

        def finalize(offsets, positions):
            results.append((offsets, list(positions)))

        helper = self.create_helper(finalize)
        helper.start_probe(self.create_command(gcmd, "automatic", probe_name))
        if not results:
            raise gcmd.error("Probe calibration aborted")
        return results[0]

    def measure_manual_mesh(self, gcmd, callback):
        results = []

        def finalize(offsets, positions):
            results.append((offsets, list(positions)))

        helper = self.create_helper(finalize)
        original = helper._manual_probe_finalize

        def manual_finalize(kin_pos):
            if kin_pos is None:
                self.cancel()
                return
            try:
                original(kin_pos)
                if results and self.active:
                    callback(results[0])
            except Exception:
                self.cancel()
                raise

        helper._manual_probe_finalize = manual_finalize
        helper.start_probe(self.create_command(gcmd, "manual", None))

    def get_values(self, gcmd, result):
        offsets, positions = result
        layout = self.layout
        if len(positions) != len(layout.points):
            raise gcmd.error("Incomplete compensation calibration")
        values = []
        for expected, measured in zip(layout.points, positions):
            if any(
                abs(expected[i] - measured[i] - offsets[i]) > 0.1
                for i in (0, 1)
            ):
                raise gcmd.error("Compensation calibration point mismatch")
            value = measured[2] - offsets[2]
            if not math.isfinite(value):
                raise gcmd.error("Non-finite compensation calibration result")
            values.append(value)
        return values

    def finish(self, gcmd, target, result, reference_name):
        mesh = self.layout.build_difference(
            target,
            self.get_values(gcmd, result),
            self.compensation.name,
            gcmd.error,
        )
        self.compensation.save(mesh, reference_name, self.layout)
        self.compensation.mesh_reference_name = reference_name
        self.compensation.mesh = mesh
        self.previous_mesh = None
        self.active = False
        gcmd.respond_info(
            "Axis twist compensation calibration complete. "
            "The map is active; run SAVE_CONFIG to persist it."
        )

    def cancel(self):
        if not self.active:
            return
        self.active = False
        self.compensation.mesh = self.previous_mesh
        self.previous_mesh = None
        self.gcmd.respond_info("Axis twist compensation calibration aborted")


class AxisTwistCompensation:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.config = config
        self.probe_list = probe_module.ProbeList.get_list(self.printer)
        self.maps = {}
        self.reading_probe = None
        self.scan_probe = None
        for option in (*_SHARED_OPTIONS, *_LEGACY_OPTIONS):
            config.get(option, None)
        self.printer.lookup_object("gcode").register_command(
            "AXIS_TWIST_COMPENSATION_CALIBRATE",
            self.cmd_CALIBRATE,
            desc="Calibrate a probe compensation mesh against a reference mesh",
        )
        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect
        )

    def _track_probe(self, probe):
        if getattr(probe, "_twist_tracked", False):
            return
        probe._twist_tracked = True
        original_probe = probe._probe
        original_begin = probe.multi_probe_begin
        original_end = probe.multi_probe_end

        def tracked_probe(*args, **kwargs):
            previous = self.reading_probe
            self.reading_probe = probe
            try:
                return original_probe(*args, **kwargs)
            finally:
                self.reading_probe = previous

        def tracked_begin(*args, **kwargs):
            self.scan_probe = probe
            return original_begin(*args, **kwargs)

        def tracked_end(*args, **kwargs):
            self.scan_probe = None
            return original_end(*args, **kwargs)

        probe._probe = tracked_probe
        probe.multi_probe_begin = tracked_begin
        probe.multi_probe_end = tracked_end

    def _handle_connect(self):
        for probe in self.probe_list.get_all().values():
            self._track_probe(probe)
        for compensation in self.maps.values():
            compensation.handle_connect()
        selected = self.probe_list.get_default_probe()
        if selected is None:
            return
        compensation = self.maps.get(selected.probe_name)
        if compensation is not None and compensation.mesh is not None:
            return
        mesh = _import_legacy_mesh(self.config, selected)
        if mesh is not None:
            if compensation is None:
                compensation = self._create_compensation(
                    selected, self.printer.config_error
                )
            compensation.mesh = mesh

    def _create_compensation(self, selected, error):
        config = self.config.getsection(
            "axis_twist_compensation " + selected.probe_name
        )
        try:
            compensation = ProbeCompensation(config, self)
        except self.printer.config_error as e:
            raise error(str(e))
        compensation.handle_connect()
        self.printer.add_object(compensation.name, compensation)
        return compensation

    def get_compensation(self, selected_probe):
        compensation = self.maps.get(selected_probe.probe_name)
        if compensation is not None and compensation.probe is selected_probe:
            return compensation
        return None

    def get_z_compensation_value(self, pos, probe=None):
        if probe is None:
            probe = self.reading_probe or self.scan_probe
        if probe is None:
            probe = self.probe_list.get_default_probe()
        if probe is None:
            return 0.0
        compensation = self.get_compensation(probe)
        if compensation is None or compensation.mesh is None:
            return 0.0
        x_offset, y_offset, _ = probe.get_offsets()
        return compensation.mesh.calc_z(pos[0] + x_offset, pos[1] + y_offset)

    def cmd_CALIBRATE(self, gcmd):
        if any(
            key in gcmd.get_command_parameters()
            for key in ("AXIS", "SAMPLE_COUNT")
        ):
            raise gcmd.error(
                "AXIS and SAMPLE_COUNT are no longer supported; configure a full mesh"
            )
        selected = self.probe_list.get_command_probe(gcmd)
        compensation = self.maps.get(selected.probe_name)
        if compensation is None:
            compensation = self._create_compensation(selected, gcmd.error)
        calibration = compensation.calibration
        reference_name = gcmd.get(
            "REFERENCE_PROBE", compensation.config.get("reference_probe", "manual")
        )
        reference = None
        if reference_name.lower() != "manual":
            reference = self.probe_list.get_probe(reference_name)
            if reference is None:
                raise gcmd.error("Unknown reference probe %s" % reference_name)
            if reference is selected:
                raise gcmd.error("Compensation probe cannot reference itself")
        calibration.start(gcmd, reference)


class ProbeCompensation:
    def __init__(self, config, manager):
        self.printer = config.get_printer()
        self.manager = manager
        self.name = config.get_name()
        self.probe_name = self.name.split(" ", 1)[1]
        self.config = _CalibrationConfig(config, manager.config)
        self.probe = None
        self.mesh = None
        self.mesh_reference_name = self.config.get("reference_probe", "manual")
        self._calibration = None
        points = config.getlists("points", [], seps=(",", "\n"), parser=float)
        version = config.getint("version", None)
        if points:
            if version != MESH_VERSION:
                raise config.error(
                    "Unsupported axis twist compensation mesh version"
                )
            params = {}
            for key, parser in bed_mesh.PROFILE_OPTIONS.items():
                getter = {
                    int: config.getint,
                    float: config.getfloat,
                    str: config.get,
                }[parser]
                params[key] = getter(key)
            verify_mesh_params(params, config.error)
            self.mesh = CompensationMesh.build(
                params, points, self.name, config.error
            )
        elif version is not None:
            raise config.error("Saved compensation mesh has no points")
        if (
            self.config.get("mesh_min", None) is not None
            or self.config.get("mesh_radius", None) is not None
        ):
            self._calibration = Calibration(self.config, self)
        manager.maps[self.probe_name] = self

    @property
    def calibration(self):
        if self._calibration is None:
            try:
                self._calibration = Calibration(self.config, self)
            except self.printer.config_error as e:
                raise self.printer.command_error(str(e))
        return self._calibration

    def handle_connect(self):
        self.probe = self.manager.probe_list.get_probe(self.probe_name)
        if self.probe is None:
            raise self.printer.config_error(
                "Unknown compensation probe: %s" % self.probe_name
            )

    def save(self, mesh, reference_name, layout):
        configfile = self.printer.lookup_object("configfile")
        configfile.set(self.name, "reference_probe", reference_name)
        configfile.set(self.name, "version", MESH_VERSION)
        configfile.set(
            self.name,
            "points",
            "\n"
            + "\n".join(
                "  " + ", ".join("%.6f" % z for z in row)
                for row in mesh.get_probed_matrix()
            ),
        )
        settings = {**mesh.get_mesh_params(), **layout.settings}
        for option in _MOVEMENT_OPTIONS:
            value = self.config.get(option, None)
            if value is not None:
                settings[option] = value
        for key, value in settings.items():
            if isinstance(value, (tuple, list)):
                value = ", ".join(str(v) for v in value)
            configfile.set(self.name, key, value)

    def get_status(self, eventtime):
        return {
            "probe": self.probe_name,
            "reference_probe": self.mesh_reference_name,
            "calibrating": self._calibration is not None
            and self._calibration.is_calibrating(),
            "mesh_matrix": self.mesh.get_mesh_matrix() if self.mesh else [],
            "mesh_params": dict(self.mesh.get_mesh_params())
            if self.mesh
            else {},
        }


def load_config(config):
    return AxisTwistCompensation(config)


def load_config_prefix(config):
    manager = config.get_printer().load_object(
        config, "axis_twist_compensation"
    )
    return ProbeCompensation(config, manager)
