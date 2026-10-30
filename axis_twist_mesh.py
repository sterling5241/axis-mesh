# Axis Twist Mesh plugin for Kalico
#
# Measures how an offset probe (Cartographer scan, Beacon proximity, eddy,
# inductive...) reads the bed differently from a reference, at every point of
# a grid, and corrects that probe's readings per point. Works like Kalico's
# mesh-based [axis_twist_compensation], with interleaved, outlier-checked
# calibration and support for the Cartographer and Beacon plugins.
#
#   residual(x, y) = correction_probe_height - reference_height
#                  = offset + x_effect(x) + y_effect(y) + interaction(x, y)
#
# Reference ("true") readings, set with `reference:`
#   manual (default)     paper test at each point (TESTZ / ACCEPT); works on
#                        any printer.
#   load_cell            [load_cell_probe] taps at the nozzle.
#   <probe name>         any other registered probe (see LIST_PROBES).
#   touch                the correction probe's own nozzle-contact mode:
#                        Cartographer touch or Beacon contact. Its scan /
#                        proximity mode gets corrected. (single_probe: True
#                        is the older name for this.)
#
# Correction probes (`correction_probe:`, one or more, comma separated; Kalico's
# default probe if unset): any
# Kalico probe (e.g. [probe_eddy_current]), the Cartographer plugin
# (cartographer) and the Beacon plugin (beacon). Each has its own map, chosen
# automatically by which probe is taking the reading.
#
# Calibration is interleaved: at each grid point the reference measures, then
# the correction probe measures the same bed point, then it moves on. Every
# probe measurement needs 3 readings within tap_tolerance (Cartographer touch
# and Beacon contact do their own 3-of-N check). Compensation is suspended
# while calibrating; an aborted or failed run leaves the old map active.
# Rectangular grids (mesh_min / mesh_max / probe_count) and round beds
# (mesh_radius / mesh_origin / round_probe_count) are supported.
#
# Storage is Kalico's own: one map per probe in [axis_twist_compensation
# <probe>] (version, points, the bed_mesh mesh parameters, reference_probe and
# the calibration layout), written by SAVE_CONFIG and read by mesh-based
# Kalico too. Probes Kalico can't hold a map for (not in its probe list, e.g.
# Beacon) use the same format in [axis_twist_mesh <probe>]. Saved maps load at
# startup and are added (interpolated with Kalico's bed_mesh code) to that
# probe's readings: PROBE, bed mesh probing and scanning, including the
# Cartographer and Beacon plugins' own scan meshes. Reference readings are
# never changed. It hooks the get_z_compensation_value() call Kalico,
# Cartographer and Beacon already make; for a correction probe's reading the
# map replaces Kalico's value (on mesh-based Kalico it is the same data), so
# nothing is applied twice. A self-test at every start warns on the console
# if the map stops reaching the probe (e.g. after a Kalico update).
#
# Commands:
#   AXIS_TWIST_MESH_CALIBRATE [PROBE=name] [REFERENCE=...]
#                             [PROBES=both|main|correction]
#       probe, analyze, save as that probe's map and apply it (SAVE_CONFIG)
#   AXIS_TWIST_MESH_PROFILE LOAD=probe | REMOVE=probe
#   AXIS_TWIST_MESH_CLEAR [PROBE=name]        turn compensation off
#   AXIS_TWIST_MESH_TABLE [PROBE=name] [FORMAT=grid|twist]
#   AXIS_TWIST_MESH_ANALYZE                   repeat the last report
#   AXIS_TWIST_MESH_IMPORT [PROBE=name]       old z_compensations lines -> map
#   AXIS_TWIST_MESH_STATUS
#   AXIS_TWIST_MESH_CHECK                     verify maps reach the probes
#
# Config (defaults shown; the grid has no default):
#   [axis_twist_mesh]
#   correction_probe:                  (Kalico's default probe; or a comma
#                                       list: cartographer, beacon, ...)
#   reference: manual                  (load_cell, a probe name, or touch)
#   mesh_min: X, Y      (required)     or, for a round bed:
#   mesh_max: X, Y      (required)       mesh_radius: R      (required)
#   probe_count: 3, 3                    mesh_origin: 0, 0
#                                        round_probe_count: 5   (odd)
#   algorithm: auto                    (bicubic, lagrange, direct)
#   mesh_pps: 2, 2            bicubic_tension: 0.2
#   horizontal_move_z: 5      speed: 50      lift_speed: 10
#   tap_tolerance: 0.010      max_taps: 7    tap_retract: 1.0
#   max_compensation: 0.5     apply_offset: False

import logging
import math
import sys

try:
    from . import bed_mesh, manual_probe
except ImportError:  # loaded outside klippy.extras (tests)
    from klippy.extras import bed_mesh, manual_probe

KALICO_MESH_VERSION = 1          # [axis_twist_compensation <probe>] format
MAIN_PROBE = "load_cell_probe"   # default reference probe
MIN_AGREEING_TAPS = 3            # hard-coded: three readings must agree
DEPRECATED_OPTIONS = ("correction_method", "scan_height",
                      "compensation_file", "pair_tolerance",
                      "remove_twist_compensation")
ALGORITHMS = ("auto", "bicubic", "lagrange", "direct")


def _resolve_algorithm(algo, x_count, y_count, pps, error):
    """Same rules as bed_mesh: returns the algorithm ZMesh should use."""
    lo, hi = min(x_count, y_count), max(x_count, y_count)
    if max(pps) == 0:
        return "direct"
    if algo == "auto":
        if lo < 3:
            return "direct"
        if lo >= 4:
            return "bicubic"
        return "lagrange" if hi <= 6 else "direct"
    if algo == "lagrange" and hi > 6:
        raise error("axis_twist_mesh: lagrange can't use more than 6 points "
                    "per axis (probe_count %d, %d); use algorithm: bicubic "
                    "or auto" % (x_count, y_count))
    if algo in ("lagrange", "bicubic") and lo < 3:
        raise error("axis_twist_mesh: %s needs at least 3 points per axis; "
                    "use algorithm: direct or auto" % algo)
    if algo == "bicubic" and lo < 4:
        if hi > 6:
            raise error("axis_twist_mesh: bicubic can't combine 3 points on "
                        "one axis with more than 6 on the other; use "
                        "algorithm: auto")
        return "lagrange"
    return algo


def _axis(lo, hi, count):
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def _median(vals):
    vals = sorted(vals)
    m = len(vals)
    return vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2


def _in_round(x, y, origin, radius):
    return math.hypot(x - origin[0], y - origin[1]) <= radius + 1e-6


def _pad_rows(grid):
    """Fill missing (None) entries of each row with the nearest present
    value in that row, like bed_mesh does for round meshes."""
    out = []
    for row in grid:
        present = [i for i, v in enumerate(row) if v is not None]
        if not present:
            raise ValueError("round grid row has no points")
        out.append([row[min(present, key=lambda p: abs(p - i))]
                    if v is None else v for i, v in enumerate(row)])
    return out


def _check_points(prof, name, error):
    rows = prof["points"]
    if len(rows) != prof["y_count"] or \
            any(len(r) != prof["x_count"] for r in rows):
        raise error("axis_twist_mesh: [%s] points don't match x_count / "
                    "y_count" % name)
    if not all(math.isfinite(v) for r in rows for v in r) or \
            prof["max_x"] <= prof["min_x"] or prof["max_y"] <= prof["min_y"]:
        raise error("axis_twist_mesh: [%s] has invalid bounds or points"
                    % name)
    if prof["algo"] is not None and prof["algo"] not in ALGORITHMS:
        raise error("axis_twist_mesh: [%s] has unknown algo '%s'"
                    % (name, prof["algo"]))


def _reference_value(reference):
    """reference_probe as Kalico stores it: a probe name or 'manual'."""
    kind, probe = reference
    return probe if kind == "probe" else kind


def _parse_reference(value):
    """('probe', name) | ('touch', None) | ('manual', None)."""
    v = value.strip()
    low = v.lower()
    if low in ("touch", "manual"):
        return low, None
    if low in ("load_cell", ""):
        return "probe", MAIN_PROBE
    return "probe", v


def _mesh_based_kalico():
    """Newer Kalico with per-probe [axis_twist_compensation <probe>] maps."""
    try:
        from . import axis_twist_compensation as atc_module
    except ImportError:
        try:
            from klippy.extras import axis_twist_compensation as atc_module
        except ImportError:
            return False
    return hasattr(atc_module, "ProbeCompensation")


def _frames(skip, depth=8):
    """(module, function) of the callers above `skip` frames."""
    out = []
    frame = sys._getframe(skip + 1)
    while frame is not None and len(out) < depth:
        out.append((frame.f_globals.get("__name__", ""),
                    frame.f_code.co_name))
        frame = frame.f_back
    return out


class _CartographerMode:
    """Probe-like wrapper around one mode of the Cartographer plugin
    (PrinterCartographer.touch_mode / .scan_mode). perform_probe() already
    returns the bed height in toolhead Z, so the Z offset reported is 0."""

    def __init__(self, printer, mode, label, self_consensus=False):
        self.printer = printer
        self.mode = mode
        self.probe_name = label
        # Touch mode already returns the median of >= 3 touches that agree
        # within the touch model's sample_range, so one call per point.
        self.self_consensus = self_consensus

    def touch_area(self):
        """(min_x, max_x, min_y, max_y) nozzle area where Cartographer
        allows touches (its bed_mesh limits shrunk by the coil offset)."""
        b = getattr(self.mode, "boundaries", None)
        if b is None:
            return None
        return (float(b.min_x), float(b.max_x),
                float(b.min_y), float(b.max_y))

    def consensus_info(self):
        """(samples, sample_range) Cartographer enforces per touch."""
        try:
            model = self.mode.get_model()
            return int(model.samples), float(model.sample_range)
        except Exception:
            return None

    def get_offsets(self):
        off = self.mode.offset
        return (float(off.x), float(off.y), 0.0)

    def run_probe(self, gcmd, *args, **kwargs):
        toolhead = self.printer.lookup_object("toolhead")
        pos = toolhead.get_position()
        try:
            z = self.mode.perform_probe()
        except Exception as exc:
            raise self.printer.command_error(
                "%s: %s" % (self.probe_name, exc))
        return [pos[0], pos[1], float(z)]


class _BeaconMode:
    """Probe-like wrapper around one method of the Beacon plugin's
    BeaconProbe ("contact" or "proximity"). Results follow Beacon's own
    run_probe(): contact returns the nozzle-contact Z, proximity returns
    Z + trigger_distance, matching the offsets below."""

    def __init__(self, printer, beacon, method, samples=3, tolerance=0.010,
                 retries=3):
        self.printer = printer
        self.beacon = beacon
        self.method = method
        self.probe_name = "beacon " + method
        self.self_consensus = method == "contact"
        self._samples, self._tolerance = samples, tolerance
        self._retries = retries

    def touch_area(self):
        return None

    def consensus_info(self):
        return self._samples, self._tolerance

    def get_offsets(self):
        if self.method == "contact":
            return (0.0, 0.0, 0.0)
        return (float(self.beacon.x_offset), float(self.beacon.y_offset),
                float(self.beacon.trigger_distance))

    def precheck(self, error):
        if self.method != "contact":
            return
        limit = getattr(getattr(self.beacon, "mcu_contact_probe", None),
                        "max_hotend_temp", None)
        if limit is None:
            return
        try:
            toolhead = self.printer.lookup_object("toolhead")
            heater = toolhead.get_extruder().get_heater()
            eventtime = self.printer.get_reactor().monotonic()
            temp = heater.get_status(eventtime)["temperature"]
        except Exception:
            return
        if temp >= limit:
            raise error("Beacon contact needs the hotend below %.0fC "
                        "(contact_max_hotend_temperature); it is %.1fC"
                        % (limit, temp))

    def run_probe(self, gcmd, *args, **kwargs):
        params = {"PROBE_METHOD": self.method}
        if self.method == "contact":
            params.update({
                "SAMPLES": str(self._samples),
                "SAMPLES_TOLERANCE": repr(self._tolerance),
                "SAMPLES_TOLERANCE_RETRIES": str(self._retries),
                "SAMPLES_RESULT": "mean"})
        sub = self.printer.lookup_object("gcode").create_gcode_command(
            "", "", params)
        pos = list(self.beacon.run_probe(sub))
        return [float(pos[0]), float(pos[1]), float(pos[2])]


class _Nozzle:
    probe_name = "nozzle"

    @staticmethod
    def get_offsets():
        return (0.0, 0.0, 0.0)


class AxisTwistMesh:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object("gcode")
        self.name = config.get_name()

        self._read_grid(config)
        self.speed = config.getfloat("speed", 50.0, above=0.0)
        self.lift_speed = config.getfloat("lift_speed", 10.0, above=0.0)
        self.horizontal_move_z = config.getfloat(
            "horizontal_move_z", 5.0, above=0.0)

        # correction probes; unset means Kalico's default probe (resolved
        # at connect, like the target probe of axis_twist_compensation)
        names = config.getlist("correction_probe", None)
        self._auto_correction = names is None
        self.correction_probes = [n.strip() for n in names or [] if n.strip()]
        if not self._auto_correction and (
                not self.correction_probes or
                len(set(self.correction_probes))
                != len(self.correction_probes)):
            raise config.error("axis_twist_mesh: correction_probe must list "
                               "one or more different probes")

        # reference
        single = config.getboolean("single_probe", None)
        main_opt = config.get("main_probe", None)
        default_ref = "touch" if single else (main_opt or "manual")
        self.reference, self.reference_probe = _parse_reference(
            config.get("reference", default_ref))
        if single is not None and single != (self.reference == "touch"):
            raise config.error("axis_twist_mesh: single_probe conflicts with "
                               "reference; keep only reference")
        if self.reference_probe in self.correction_probes:
            raise config.error("axis_twist_mesh: the reference probe can't "
                               "also be a correction probe")

        self.algorithm = config.getchoice(
            "algorithm", {a: a for a in ALGORITHMS}, "auto")
        self.mesh_pps = self._get_pair_int(config, "mesh_pps", (2, 2))
        if min(self.mesh_pps) < 0:
            raise config.error("axis_twist_mesh: mesh_pps can't be negative")
        self.tension = config.getfloat("bicubic_tension", 0.2, minval=0.0,
                                       maxval=2.0)
        self.apply_offset = config.getboolean("apply_offset", False)
        self.tap_tolerance = config.getfloat(
            "tap_tolerance", 0.010, above=0.0)
        self.max_taps = config.getint(
            "max_taps", 7, minval=MIN_AGREEING_TAPS)
        self.tap_retract = config.getfloat("tap_retract", 1.0, above=0.0)
        self.max_compensation = config.getfloat(
            "max_compensation", 0.5, above=0.0)
        for opt in DEPRECATED_OPTIONS:
            if config.get(opt, None) is not None:
                logging.warning("axis_twist_mesh: option '%s' is no longer "
                                "used and can be removed", opt)
        _resolve_algorithm(self.algorithm, self.probe_count[0],
                           self.probe_count[1], self.mesh_pps, config.error)
        self.printer.load_object(config, "manual_probe")

        # Beacon looks the object up once, in its own klippy:connect handler.
        # Without an [axis_twist_compensation] section, register ours now so
        # it exists whichever handler runs first.
        self._hook_installed = False
        self._orig_twist_fn = None
        # Any [axis_twist_compensation ...] section (newer Kalico also loads
        # per-probe ones on their own) means Kalico creates the object.
        if not config.has_section("axis_twist_compensation") and not (
                _mesh_based_kalico() and
                config.get_prefix_sections("axis_twist_compensation ")):
            self.printer.add_object("axis_twist_compensation", self)
            self._hook_installed = True

        self.maps, self.old_profiles = self._read_maps(config)
        self._converted = []
        self.active = {}            # correction probe name -> active map
        self.last = None            # last calibration (in memory)
        self.report = None          # analysis of self.last
        self.calibrating = False
        self._manual = None         # state of a running manual calibration
        self._kalico_handles = set()   # probes Kalico corrects by itself
        self._suspended = None

        self.printer.register_event_handler(
            "klippy:connect", self._handle_connect)
        self.printer.register_event_handler(
            "klippy:ready", self._handle_ready)
        for event in ("klippy:shutdown", "klippy:disconnect"):
            self.printer.register_event_handler(event, self._abort_manual)
        commands = [
            ("CALIBRATE", self.cmd_CALIBRATE,
             "Probe the grid with both probes, save and apply as a profile"),
            ("PROFILE", self.cmd_PROFILE,
             "Turn a probe's saved map on (LOAD=) or delete it (REMOVE=)"),
            ("CLEAR", self.cmd_CLEAR,
             "Turn axis twist mesh compensation off"),
            ("TABLE", self.cmd_TABLE,
             "Print a probe's per-point compensation table"),
            ("ANALYZE", self.cmd_ANALYZE,
             "Repeat the last calibration report"),
            ("IMPORT", self.cmd_IMPORT,
             "Turn z_compensations / zy_compensations lines into a map"),
            ("STATUS", self.cmd_STATUS,
             "Report axis twist mesh configuration and state"),
            ("CHECK", self.cmd_CHECK,
             "Verify the active maps reach the correction probes"),
        ]
        for suffix, func, desc in commands:
            self.gcode.register_command(
                "AXIS_TWIST_MESH_" + suffix, func, desc=desc)

    # ------------------------------------------------------------------
    # Config helpers / grid

    @staticmethod
    def _get_pair(config, option, default):
        value = config.getfloatlist(
            option, default=None if default is None else list(default),
            count=2)
        return (float(value[0]), float(value[1]))

    @staticmethod
    def _get_pair_int(config, option, default):
        value = config.getintlist(option, default=list(default), count=2)
        return (int(value[0]), int(value[1]))

    def _read_grid(self, config):
        self.radius = config.getfloat("mesh_radius", None, above=0.0)
        if self.radius is None:
            self.origin = None
            if config.get("mesh_min", None) is None or \
                    config.get("mesh_max", None) is None:
                raise config.error("axis_twist_mesh: set mesh_min and "
                                   "mesh_max (or mesh_radius for a round "
                                   "bed)")
            self.mesh_min = self._get_pair(config, "mesh_min", None)
            self.mesh_max = self._get_pair(config, "mesh_max", None)
            self.probe_count = self._get_pair_int(config, "probe_count",
                                                  (3, 3))
            if self.mesh_min[0] >= self.mesh_max[0] or \
                    self.mesh_min[1] >= self.mesh_max[1]:
                raise config.error(
                    "axis_twist_mesh: mesh_min must be below mesh_max")
            if min(self.probe_count) < 2:
                raise config.error(
                    "axis_twist_mesh: probe_count values must be >= 2")
        else:
            for opt in ("mesh_min", "mesh_max", "probe_count"):
                if config.get(opt, None) is not None:
                    raise config.error("axis_twist_mesh: use either "
                                       "mesh_radius or %s, not both" % opt)
            self.origin = self._get_pair(config, "mesh_origin", (0.0, 0.0))
            count = config.getint("round_probe_count", 5, minval=3)
            if not count & 1:
                raise config.error("axis_twist_mesh: round_probe_count must "
                                   "be odd")
            r = self.radius
            self.mesh_min = (self.origin[0] - r, self.origin[1] - r)
            self.mesh_max = (self.origin[0] + r, self.origin[1] + r)
            self.probe_count = (count, count)
        steps = [(self.mesh_max[i] - self.mesh_min[i])
                 / (self.probe_count[i] - 1) for i in (0, 1)]
        if min(steps) < 1.0:
            raise config.error("axis_twist_mesh: grid points must be at "
                               "least 1 mm apart")

    def _grid_axis(self, axis):
        return _axis(self.mesh_min[axis], self.mesh_max[axis],
                     self.probe_count[axis])

    def _grid_nodes(self):
        """Serpentine order, rows along X, as (ix, iy, x, y). Round beds
        skip points outside the radius."""
        xs, ys = self._grid_axis(0), self._grid_axis(1)
        nodes = []
        for iy, y in enumerate(ys):
            order = range(len(xs)) if iy % 2 == 0 else \
                range(len(xs) - 1, -1, -1)
            for ix in order:
                if self.radius is not None and \
                        not _in_round(xs[ix], y, self.origin, self.radius):
                    continue
                nodes.append((ix, iy, xs[ix], y))
        return nodes

    def _grid_key(self):
        return (tuple(self._grid_axis(0)), tuple(self._grid_axis(1)),
                self.radius, self.origin)

    # ------------------------------------------------------------------
    # Profiles (stored like bed_mesh: [axis_twist_mesh <name>])

    def _first_probe(self):
        return self.correction_probes[0] if self.correction_probes else None

    def _kalico_default_probe(self):
        """Name of Kalico's default probe (what a bare PROBE uses)."""
        plist = self.printer.lookup_object("probe_list", None)
        probe = plist.get_default_probe() if plist is not None else None
        if probe is None:
            probe = self.printer.lookup_object("probe", None)
        if probe is None:
            return None
        beacon = getattr(probe, "beacon", None)   # Beacon's probe wrapper
        if beacon is not None:
            name = getattr(beacon, "name", "beacon")
            return name if self.printer.lookup_object(name, None) is beacon \
                else None
        return getattr(probe, "probe_name", None)

    @staticmethod
    def _read_map(section, error):
        """A saved map in Kalico's [axis_twist_compensation <probe>] format
        (also used for [axis_twist_mesh <probe>])."""
        for opt in section.get_prefix_options(""):
            section.get(opt, None)          # all options are ours / valid
        if section.getint("version", None) != KALICO_MESH_VERSION:
            raise error("axis_twist_mesh: [%s] has an unsupported version"
                        % section.get_name())
        prof = {
            "x_count": section.getint("x_count", minval=2),
            "y_count": section.getint("y_count", minval=2),
            "min_x": section.getfloat("min_x"),
            "max_x": section.getfloat("max_x"),
            "min_y": section.getfloat("min_y"),
            "max_y": section.getfloat("max_y"),
            "mesh_x_pps": section.getint("mesh_x_pps", minval=0),
            "mesh_y_pps": section.getint("mesh_y_pps", minval=0),
            "algo": section.get("algo").strip().lower(),
            "tension": section.getfloat("tension", 0.2, minval=0.0,
                                        maxval=2.0),
            "reference": section.get("reference_probe", "manual"),
            "points": section.getlists("points", seps=(",", "\n"),
                                       parser=float),
            "mesh_radius": section.getfloat("mesh_radius", None, above=0.0),
            "mesh_origin": None,
        }
        if prof["mesh_radius"] is not None:
            prof["mesh_origin"] = tuple(section.getfloatlist(
                "mesh_origin", (0.0, 0.0), count=2))
        _check_points(prof, section.get_name(), error)
        return prof

    @staticmethod
    def _read_old_profile(section, error):
        """[axis_twist_mesh <name>] from earlier versions of this add-on."""
        for opt in section.get_prefix_options(""):
            section.get(opt, None)
        if section.getint("version", 0) != 1:
            return None
        origin = section.getfloatlist("mesh_origin", None, count=2)
        prof = {
            "probe": section.get("probe", None),
            "x_count": section.getint("x_count", minval=2),
            "y_count": section.getint("y_count", minval=2),
            "min_x": section.getfloat("min_x"),
            "max_x": section.getfloat("max_x"),
            "min_y": section.getfloat("min_y"),
            "max_y": section.getfloat("max_y"),
            "mesh_radius": section.getfloat("mesh_radius", None, above=0.0),
            "mesh_origin": tuple(origin) if origin else None,
            "offset": section.getfloat("offset", 0.0),
            "reference": section.get("reference", "") or "manual",
            "points": section.getlists("points", seps=(",", "\n"),
                                       parser=float),
            "algo": section.get("algo", None),
            "mesh_x_pps": section.getint("mesh_x_pps", None, minval=0),
            "mesh_y_pps": section.getint("mesh_y_pps", None, minval=0),
            "tension": section.getfloat("tension", None, minval=0.0,
                                        maxval=2.0),
        }
        _check_points(prof, section.get_name(), error)
        return prof

    def _read_maps(self, config):
        """Saved maps, one per probe, keyed by probe name; plus profiles
        left over from earlier versions of this add-on."""
        maps, old = {}, {}
        for prefix in ("axis_twist_compensation ", self.name + " "):
            for section in config.get_prefix_sections(prefix):
                probe = section.get_name().split(" ", 1)[1].strip()
                if prefix != "axis_twist_compensation " and \
                        section.get("reference_probe", None) is None:
                    prof = self._read_old_profile(section, config.error)
                    if prof is not None:
                        old[probe] = prof
                    continue
                if section.get("points", None) is None:
                    continue        # Kalico section with no map yet
                prof = self._read_map(section, config.error)
                prof["section"] = section.get_name()
                if probe in maps:
                    raise config.error(
                        "axis_twist_mesh: '%s' has maps in both [%s] and "
                        "[%s]; remove one" % (probe, maps[probe]["section"],
                                              prof["section"]))
                maps[probe] = prof
        return maps, old

    def _section_for(self, probe_name):
        """Kalico's own section where Kalico can hold the map (probes in its
        probe list); this add-on's section otherwise (e.g. Beacon)."""
        plist = self.printer.lookup_object("probe_list", None)
        if plist is not None and plist.get_probe(probe_name) is not None:
            return "axis_twist_compensation " + probe_name
        return "%s %s" % (self.name, probe_name)

    def _store_map(self, probe_name, prof, layout=True):
        """Write a probe's map in Kalico's format (SAVE_CONFIG keeps it)."""
        configfile = self.printer.lookup_object("configfile")
        section = self._section_for(probe_name)
        old = self.maps.get(probe_name)
        if old is not None and old.get("section") not in (None, section):
            configfile.remove_section(old["section"])
        params = self._mesh_params(prof, ValueError)
        configfile.remove_section(section)
        configfile.set(section, "reference_probe", prof["reference"])
        configfile.set(section, "version", KALICO_MESH_VERSION)
        configfile.set(section, "points", "\n" + "\n".join(
            "  " + ", ".join("%.6f" % z for z in row)
            for row in prof["points"]))
        for key in ("min_x", "max_x", "min_y", "max_y"):
            configfile.set(section, key, "%.3f" % params[key])
        for key in ("x_count", "y_count", "mesh_x_pps", "mesh_y_pps"):
            configfile.set(section, key, "%d" % params[key])
        configfile.set(section, "algo", params["algo"])
        configfile.set(section, "tension", "%.3f" % params["tension"])
        if prof.get("mesh_radius") is not None:
            configfile.set(section, "mesh_radius", "%.3f"
                           % prof["mesh_radius"])
            configfile.set(section, "mesh_origin", "%.3f, %.3f"
                           % tuple(prof["mesh_origin"]))
        if layout and min(self.probe_count) >= 3:
            # the calibration layout, as Kalico writes it
            configfile.set(section, "mesh_pps", "%d, %d" % (
                params["mesh_x_pps"], params["mesh_y_pps"]))
            configfile.set(section, "algorithm", params["algo"])
            configfile.set(section, "bicubic_tension", "%.3f"
                           % params["tension"])
            if self.radius is None:
                configfile.set(section, "mesh_min", "%.3f, %.3f"
                               % self.mesh_min)
                configfile.set(section, "mesh_max", "%.3f, %.3f"
                               % self.mesh_max)
                configfile.set(section, "probe_count", "%d, %d"
                               % self.probe_count)
            else:
                configfile.set(section, "mesh_radius", "%.3f" % self.radius)
                configfile.set(section, "mesh_origin", "%.3f, %.3f"
                               % self.origin)
                configfile.set(section, "round_probe_count", "%d"
                               % self.probe_count[0])
        prof = dict(prof, **{k: params[k] for k in (
            "algo", "mesh_x_pps", "mesh_y_pps", "tension")})
        prof["section"] = section
        maps = dict(self.maps)
        maps[probe_name] = prof
        self.maps = maps
        return section

    def _mesh_params(self, prof, error):
        """bed_mesh ZMesh parameters for a map."""
        pps = (prof.get("mesh_x_pps"), prof.get("mesh_y_pps"))
        if None in pps:
            pps = self.mesh_pps
        algo = _resolve_algorithm(prof.get("algo") or self.algorithm,
                                  prof["x_count"], prof["y_count"], pps,
                                  error)
        if algo == "direct":
            pps = (0, 0)        # bed_mesh: direct means no sub-points
        tension = prof.get("tension")
        return {
            "min_x": prof["min_x"], "max_x": prof["max_x"],
            "min_y": prof["min_y"], "max_y": prof["max_y"],
            "x_count": prof["x_count"], "y_count": prof["y_count"],
            "mesh_x_pps": pps[0], "mesh_y_pps": pps[1],
            "algo": algo,
            "tension": self.tension if tension is None else tension,
        }

    def _build_mesh(self, probe_name, prof, error):
        mesh = bed_mesh.ZMesh(self._mesh_params(prof, error),
                              "%s %s" % (self.name, probe_name))
        try:
            mesh.build_mesh([list(r) for r in prof["points"]])
        except Exception as exc:
            raise error("axis_twist_mesh: map for '%s': %s"
                        % (probe_name, exc))
        return mesh

    def _activate(self, probe_name, error):
        prof = self.maps.get(probe_name)
        if prof is None:
            raise error("axis_twist_mesh: no saved map for '%s'"
                        % probe_name)
        if probe_name not in self.correction_probes:
            raise error("axis_twist_mesh: '%s' is not in correction_probe"
                        % probe_name)
        worst = max(abs(v) for row in prof["points"] for v in row)
        if worst > self.max_compensation:
            raise error("axis_twist_mesh: the map for '%s' has a %.3f mm "
                        "correction, above max_compensation (%.3f mm); not "
                        "loaded" % (probe_name, worst, self.max_compensation))
        problem = self._support_problem(probe_name)
        if problem:
            raise error(problem)
        mesh = self._build_mesh(probe_name, prof, error)
        active = dict(self.active)
        active[probe_name] = {
            "name": prof.get("section", probe_name),
            "probe": probe_name,
            "xs": _axis(prof["min_x"], prof["max_x"], prof["x_count"]),
            "ys": _axis(prof["min_y"], prof["max_y"], prof["y_count"]),
            "grid": prof["points"],
            "mesh": mesh,
        }
        self.active = active
        self._update_integration()

    # ------------------------------------------------------------------
    # Probe access

    def _resolve_probe(self, name):
        plist = self.printer.lookup_object("probe_list", None)
        if plist is not None:
            probe = plist.get_probe(name)
            if probe is not None:
                return probe
        obj = self.printer.lookup_object(name, None)
        if obj is not None and callable(getattr(obj, "run_probe", None)) \
                and callable(getattr(obj, "get_offsets", None)):
            return obj
        return None

    @staticmethod
    def _is_beacon(obj):
        return obj is not None and type(obj).__name__ == "BeaconProbe"

    @staticmethod
    def _is_cartographer_plugin(probe):
        """True for the third-party Cartographer plugin's probe object."""
        return probe is not None and \
            type(probe).__module__.split(".")[0] == "cartographer"

    def _single_modes(self, probe_name=None):
        """(reference, corrected) wrappers for reference: touch, or None:
        Cartographer touch/scan or Beacon contact/proximity."""
        name = probe_name or self._first_probe()
        obj = self.printer.lookup_object(name, None)
        if self._is_beacon(obj):
            return (_BeaconMode(self.printer, obj, "contact",
                                MIN_AGREEING_TAPS, self.tap_tolerance,
                                max(0, self.max_taps - MIN_AGREEING_TAPS)),
                    _BeaconMode(self.printer, obj, "proximity"))
        touch = getattr(obj, "touch_mode", None)
        scan = getattr(obj, "scan_mode", None)
        if touch is None or scan is None:
            return None
        return (_CartographerMode(self.printer, touch, "cartographer touch",
                                  self_consensus=True),
                _CartographerMode(self.printer, scan, "cartographer scan"))

    def _correction_reader(self, probe):
        """What to call for correction-probe readings during calibration.
        Beacon is forced to proximity whatever its default_probe_method."""
        if self._is_beacon(probe):
            return _BeaconMode(self.printer, probe, "proximity")
        return probe

    @property
    def _touch_ref(self):
        return self.reference == "touch"

    def _main_label(self, reference=None, probe_name=None):
        if reference is None:
            reference = (self.reference, self.reference_probe)
        kind, ref_probe = reference
        if kind == "manual":
            return "manual (paper)"
        if kind == "probe":
            return ref_probe
        modes = self._single_modes(probe_name)
        if modes is not None:
            return modes[0].probe_name
        return "%s (contact/touch)" % (probe_name or
                                       self._first_probe())

    def _main_found(self):
        if self.reference == "manual":
            return True
        if self._touch_ref:
            return self._single_modes() is not None
        return self._resolve_probe(self.reference_probe) is not None

    def _available_probes(self):
        plist = self.printer.lookup_object("probe_list", None)
        if plist is None:
            return []
        return sorted(plist.get_all().keys())

    def _handle_connect(self):
        if self._auto_correction:
            name = self._kalico_default_probe()
            if name is None or name == self.reference_probe:
                logging.warning(
                    "axis_twist_mesh: correction_probe is not set and Kalico's "
                    "default probe %s; set correction_probe",
                    "is the reference" if name else "was not found")
            else:
                self.correction_probes = [name]
        if self.reference_probe is not None and \
                self.reference_probe in self.correction_probes:
            raise self.printer.config_error(
                "axis_twist_mesh: the reference probe can't also be a "
                "correction probe")
        self._convert_old_profiles()
        for probe_name in self.correction_probes:
            if probe_name not in self.maps:
                continue
            try:
                self._activate(probe_name, ValueError)
                logging.info("axis_twist_mesh: map for '%s' loaded",
                             probe_name)
            except ValueError as exc:
                logging.warning("axis_twist_mesh: map for '%s' not loaded: "
                                "%s", probe_name, exc)
        self._update_integration()
        if self._touch_ref and self._single_modes() is None:
            logging.warning(
                "axis_twist_mesh: reference: touch needs the Cartographer or "
                "Beacon plugin object '%s'; not found.",
                self._first_probe())
        names = [("correction_probe", n) for n in self.correction_probes]
        if self.reference == "probe":
            names.insert(0, ("reference", self.reference_probe))
        for role, name in names:
            if self._resolve_probe(name) is None:
                logging.warning(
                    "axis_twist_mesh: %s '%s' not found (registered probes: "
                    "%s).", role, name,
                    ", ".join(self._available_probes()) or "none")

    def _convert_old_profiles(self):
        """Earlier versions saved [axis_twist_mesh default] (and
        default_<probe>) profiles: turn each probe's default one into its
        map in the new format. SAVE_CONFIG keeps the conversion."""
        for probe_name in self.correction_probes:
            if probe_name in self.maps:
                continue
            name = "default" if probe_name == self._first_probe() \
                else "default_" + probe_name
            old = self.old_profiles.get(name)
            if old is None or (old["probe"] or self._first_probe()) \
                    != probe_name:
                continue
            points = old["points"]
            if self.apply_offset:
                points = [[v - old["offset"] for v in r] for r in points]
            prof = dict(old, points=points)
            self._store_map(probe_name, prof, layout=False)
            self.printer.lookup_object("configfile").remove_section(
                "%s %s" % (self.name, name))
            del self.old_profiles[name]
            self._converted.append((name, probe_name))

    def _require_probe(self, gcmd, role, name):
        probe = self._resolve_probe(name)
        if probe is None:
            raise gcmd.error(
                "axis_twist_mesh: %s '%s' not found. Registered probes: %s "
                "(see LIST_PROBES)" % (
                    role, name, ", ".join(self._available_probes()) or "none"))
        return probe

    def _check_touch_area(self, gcmd, probe):
        area_fn = getattr(probe, "touch_area", None)
        area = area_fn() if area_fn is not None else None
        if area is None:
            return
        x_off, y_off = probe.get_offsets()[:2]
        min_x, max_x, min_y, max_y = area
        eps = 0.01
        bad = [(x, y) for _, _, x, y in self._grid_nodes()
               if not (min_x - eps <= x - x_off <= max_x + eps
                       and min_y - eps <= y - y_off <= max_y + eps)]
        if bad:
            raise gcmd.error(
                "axis_twist_mesh: %d grid point(s) are outside Cartographer's "
                "touch area X=[%.2f, %.2f] Y=[%.2f, %.2f] (e.g. %s). Set "
                "mesh_min: %.1f, %.1f and mesh_max: %.1f, %.1f in "
                "[axis_twist_mesh], or widen Cartographer's bed mesh area."
                % (len(bad), min_x, max_x, min_y, max_y,
                   ", ".join("(%.1f, %.1f)" % b for b in bad[:4]),
                   max(self.mesh_min[0], min_x + x_off),
                   max(self.mesh_min[1], min_y + y_off),
                   min(self.mesh_max[0], max_x + x_off),
                   min(self.mesh_max[1], max_y + y_off)))

    def _check_reachable(self, gcmd, toolhead, role, probe):
        x_off, y_off = probe.get_offsets()[:2]
        eventtime = self.printer.get_reactor().monotonic()
        status = toolhead.get_kinematics().get_status(eventtime)
        amin, amax = status.get("axis_minimum"), status.get("axis_maximum")
        if amin is None or amax is None:
            return
        for _, _, x, y in self._grid_nodes():
            nx, ny = x - x_off, y - y_off
            if not (amin[0] <= nx <= amax[0] and amin[1] <= ny <= amax[1]):
                raise gcmd.error(
                    "axis_twist_mesh: bed point (%.2f, %.2f) needs nozzle at "
                    "(%.2f, %.2f) for %s, outside axis limits. Shrink the "
                    "grid." % (x, y, nx, ny, role))

    # ------------------------------------------------------------------
    # Probing

    def _tap_gcmd(self, gcmd):
        params = dict(gcmd.get_command_parameters())
        for key in ("SAMPLES", "SAMPLES_RESULT", "SAMPLES_TOLERANCE",
                    "SAMPLES_TOLERANCE_RETRIES", "PROFILE", "PROBES",
                    "REFERENCE", "PROBE"):
            params.pop(key, None)
        params["SAMPLES"] = "1"
        return self.gcode.create_gcode_command(
            "AXIS_TWIST_MESH_CALIBRATE", "AXIS_TWIST_MESH_CALIBRATE", params)

    def _probe_grid(self, gcmd, toolhead, probes):
        """Interleaved: at each bed point, every probe in `probes` takes its
        agreeing readings before moving to the next point. Returns
        {role: ({(ix, iy): height}, offsets)}."""
        nodes = self._grid_nodes()
        tap_gcmd = self._tap_gcmd(gcmd)
        info = []
        for role, probe in probes:
            info.append((role, probe, probe.get_offsets()[:3]))
            if getattr(probe, "self_consensus", False):
                ci = probe.consensus_info()
                if ci is not None and ci[0] < MIN_AGREEING_TAPS:
                    raise gcmd.error(
                        "%s: touch model samples is %d; set samples: %d or "
                        "more" % (probe.probe_name, ci[0], MIN_AGREEING_TAPS))
                gcmd.respond_info(
                    "axis_twist_mesh: %s uses the probe's own check: %s"
                    % (probe.probe_name,
                       "%d touches within %.3f mm per point" % ci
                       if ci else "its touch settings"))
        gcmd.respond_info(
            "axis_twist_mesh: %d points, at each: %s" % (
                len(nodes), " then ".join(
                    "%s '%s' (offsets x=%.3f y=%.3f z=%.3f)" % (
                        role, getattr(p, "probe_name", "?"), o[0], o[1], o[2])
                    for role, p, o in info)))
        heights = {role: {} for role, _, _ in info}
        extra = {role: 0 for role, _, _ in info}
        for ix, iy, x, y in nodes:
            for role, probe, (x_off, y_off, z_off) in info:
                toolhead.manual_move(
                    [None, None, self.horizontal_move_z], self.lift_speed)
                toolhead.manual_move([x - x_off, y - y_off, None], self.speed)
                begin = getattr(probe, "multi_probe_begin", None)
                end = getattr(probe, "multi_probe_end", None)
                if begin is not None:
                    begin()
                try:
                    z, taps = self._agreeing_taps(
                        gcmd, tap_gcmd, toolhead, probe, x, y, role)
                finally:
                    if end is not None:
                        end()
                extra[role] += len(taps) - MIN_AGREEING_TAPS
                heights[role][(ix, iy)] = z - z_off
        toolhead.manual_move(
            [None, None, self.horizontal_move_z], self.lift_speed)
        gcmd.respond_info(
            "axis_twist_mesh: probing done; extra readings needed to get %d "
            "within %.3f mm: %s" % (
                MIN_AGREEING_TAPS, self.tap_tolerance,
                ", ".join("%s %d" % (r, extra[r]) for r, _, _ in info)))
        return {role: (heights[role], tuple(o)) for role, _, o in info}

    def _agreeing_taps(self, gcmd, tap_gcmd, toolhead, probe, x, y, role):
        """Tap until MIN_AGREEING_TAPS readings all lie within tap_tolerance
        of each other; return their mean and the list of readings."""
        k = MIN_AGREEING_TAPS
        if getattr(probe, "self_consensus", False):
            z = probe.run_probe(tap_gcmd)[2]
            return z, [z] * k
        taps = []
        while len(taps) < self.max_taps:
            if taps:
                pos = toolhead.get_position()
                toolhead.manual_move(
                    [None, None, pos[2] + self.tap_retract], self.lift_speed)
            taps.append(probe.run_probe(tap_gcmd)[2])
            if len(taps) < k:
                continue
            ordered = sorted(taps)
            best = None
            for i in range(len(ordered) - k + 1):
                group = ordered[i:i + k]
                spread = group[-1] - group[0]
                if spread <= self.tap_tolerance and \
                        (best is None or spread < best[0]):
                    best = (spread, group)
            if best is not None:
                return sum(best[1]) / k, taps
        raise gcmd.error(
            "axis_twist_mesh: %s probe: no %d of %d readings at (%.2f, %.2f) "
            "agreed within %.3f mm (readings: %s). Clean the nozzle or raise "
            "tap_tolerance."
            % (role, k, self.max_taps, x, y, self.tap_tolerance,
               ", ".join("%.4f" % t for t in taps)))

    # ------------------------------------------------------------------
    # Live compensation of the correction probes

    def _install_hook(self):
        if self._hook_installed:
            return
        atc = self.printer.lookup_object("axis_twist_compensation", None)
        if atc is None:
            self.printer.add_object("axis_twist_compensation", self)
        else:
            self._orig_twist_fn = atc.get_z_compensation_value
            atc.get_z_compensation_value = self.get_z_compensation_value
        self._hook_installed = True

    def _hook_active(self):
        atc = self.printer.lookup_object("axis_twist_compensation", None)
        return atc is self or (atc is not None and getattr(
            atc.get_z_compensation_value, "__self__", None) is self)

    def _kalico_manager(self):
        """Mesh-based Kalico's axis_twist_compensation (per-probe maps), or
        None on older Kalico / when there is none."""
        owner = self._native_owner()
        if owner is None:
            owner = self.printer.lookup_object("axis_twist_compensation",
                                               None)
        if owner is None or owner is self or \
                getattr(owner, "get_compensation", None) is None or \
                not hasattr(owner, "maps"):
            return None
        return owner

    def _kalico_comp(self, probe_name, create=False):
        """Kalico's map holder for a probe in its probe list, or None."""
        manager = self._kalico_manager()
        plist = self.printer.lookup_object("probe_list", None)
        probe = plist.get_probe(probe_name) if plist is not None else None
        if manager is None or probe is None:
            return None
        comp = manager.get_compensation(probe)
        if comp is None and create and \
                hasattr(manager, "_create_compensation"):
            try:
                comp = manager._create_compensation(probe)
            except Exception:
                logging.exception("axis_twist_mesh: can't give Kalico a map "
                                  "holder for '%s'", probe_name)
                comp = None
        return comp if hasattr(comp, "mesh") else None

    def _update_integration(self):
        """Give mesh-based Kalico the active maps so it corrects those
        probes itself, and wrap its function only for readings it can't
        attribute (Beacon; Cartographer when it isn't the default probe;
        everything on older Kalico)."""
        handled = set()
        for name in self.correction_probes:
            entry = self.active.get(name)
            comp = self._kalico_comp(name, create=entry is not None)
            if comp is None:
                continue
            comp.mesh = entry["mesh"] if entry is not None else None
            if entry is not None:
                handled.add(name)
        self._kalico_handles = handled
        if self._needs_wrap():
            self._install_hook()

    def _needs_wrap(self):
        if not self.active:
            return False
        if self._kalico_manager() is None:
            return True
        default = self._kalico_default_probe()
        for name in self.active:
            corr = self._resolve_probe(name)
            if name not in self._kalico_handles or self._is_beacon(corr):
                return True
            if self._is_cartographer_plugin(corr) and name != default:
                return True     # it doesn't say which probe is reading
        return False

    def _suspend_kalico_map(self, probe_name):
        """Raw readings while calibrating: take Kalico's map for this probe
        out until the calibration ends (restored if it fails)."""
        comp = self._kalico_comp(probe_name)
        if comp is None or comp.mesh is None:
            return None
        saved, comp.mesh = comp.mesh, None
        return comp, saved

    def _resume_kalico_map(self, suspended):
        if suspended is not None:
            comp, saved = suspended
            if comp.mesh is None:
                comp.mesh = saved

    @staticmethod
    def _probe_active(probe):
        return probe is not None and \
            bool(getattr(probe, "multi_probe_pending", False))

    def _non_correction_probe_active(self):
        """A registered probe that isn't a correction probe (the reference,
        e.g. the load cell) is in the middle of probing."""
        plist = self.printer.lookup_object("probe_list", None)
        if plist is None:
            return False
        corr = [self._resolve_probe(n) for n in self.correction_probes]
        return any(p not in corr and self._probe_active(p)
                   for p in plist.get_all().values())

    def _xy_offset(self, probe_name):
        corr = self._resolve_probe(probe_name)
        # Beacon's get_offsets() depends on the method used last; the
        # proximity offset is what scan readings need.
        if self._is_beacon(corr):
            return float(corr.x_offset), float(corr.y_offset)
        return tuple(corr.get_offsets()[:2])

    def _identify_reading(self, probe):
        """Which correction probe took this reading, or None (reference
        readings, Beacon contact, anything unidentifiable)."""
        corr = {n: self._resolve_probe(n) for n in self.correction_probes}
        if probe is not None:
            # newer Kalico says which probe it is
            for name, obj in corr.items():
                if obj is not None and (probe is obj or
                                        getattr(probe, "beacon", None) is obj):
                    return name
            return None
        # Plugins (and older Kalico) don't. A Kalico probe mid-probe wins;
        # otherwise only the direct caller counts: Cartographer's adapter
        # and Beacon's lookup call us directly, while Kalico's own probing
        # may run underneath a plugin command (e.g. its fallbacks).
        found = [n for n, o in corr.items() if self._probe_active(o)]
        callers = _frames(2, depth=2)
        caller = callers[0][0] if callers else ""
        if found:
            pass
        elif caller.split(".")[-1] == "beacon":
            if any(f == "_probe_contact" for _, f in callers):
                return None         # Beacon contact: a nozzle reading
            found = [n for n, o in corr.items() if self._is_beacon(o)]
        elif caller.split(".")[0] == "cartographer":
            found = [n for n, o in corr.items()
                     if self._is_cartographer_plugin(o)]
        else:
            if len(corr) == 1:
                # e.g. older Kalico's rapid scan with a plugin probe
                (name, obj), = corr.items()
                if self._is_cartographer_plugin(obj) or self._is_beacon(obj):
                    found = [name]
        if len(found) != 1 or self._non_correction_probe_active():
            return None
        return found[0]

    def _support_problem(self, probe_name):
        """Why live compensation can't work for this probe, or None."""
        corr = self._resolve_probe(probe_name)
        if corr is None:
            return "Correction probe '%s' not found" % probe_name
        if self._is_beacon(corr):
            if getattr(corr, "mod_axis_twist_comp", None) is None:
                return ("Beacon did not find axis_twist_compensation when it "
                        "started, so it never asks for a correction. Restart "
                        "Klipper (RESTART) with this plugin installed.")
            return None
        if self._is_cartographer_plugin(corr):
            atc = self.printer.lookup_object("axis_twist_compensation", None)
            if (atc is None or atc is self) and _mesh_based_kalico():
                return ("The Cartographer plugin only applies Z corrections "
                        "through [axis_twist_compensation], but its current "
                        "release fails to start with this Kalico's "
                        "mesh-based axis_twist_compensation (it reads "
                        "horizontal_move_z / speed, which no longer exist). "
                        "Cartographer needs an update before its scan "
                        "readings can be corrected here.")
            if atc is None or atc is self:
                return ("The Cartographer plugin only applies Z corrections "
                        "when an [axis_twist_compensation] section exists. "
                        "Add this to printer.cfg and restart:\n"
                        "[axis_twist_compensation]\n"
                        "calibrate_start_x: %.1f\ncalibrate_end_x: %.1f\n"
                        "calibrate_y: %.1f" % (
                            self.mesh_min[0], self.mesh_max[0],
                            (self.mesh_min[1] + self.mesh_max[1]) / 2.0))
            return None
        if getattr(corr, "multi_probe_pending", None) is None:
            return ("Correction probe '%s' is neither a standard Kalico probe "
                    "nor the Cartographer or Beacon plugin, so its readings "
                    "can't be told apart from the reference's." % probe_name)
        return None

    def _native_value(self, pos, probe):
        if self._orig_twist_fn is None:
            return 0.0
        if probe is None:
            return self._orig_twist_fn(pos)
        return self._orig_twist_fn(pos, probe)

    def get_z_compensation_value(self, pos, probe=None):
        """Called for every probe reading, with the nozzle position, by
        Kalico's probe code and by the Cartographer and Beacon plugins.
        Returns existing twist compensation plus ours (correction probe
        scan readings only).

        Newer Kalico (mesh-based axis_twist_compensation) passes the probe
        that took the reading; older Kalico and the Cartographer/Beacon
        plugins pass only the position."""
        if self.calibrating:
            return 0.0   # raw readings while calibrating
        name = self._identify_reading(probe) if self.active else None
        if name is None or name not in self.active:
            # not a correction probe's reading: Kalico's own value
            return self._native_value(pos, probe)
        if name in self._kalico_handles and (
                probe is not None or
                name == self._kalico_default_probe() and
                not self._is_beacon(self._resolve_probe(name))):
            # Kalico has this map and knows the probe: its own value
            return self._native_value(pos, probe)
        # A correction probe's map is stored where Kalico keeps its own
        # (same data), so it replaces Kalico's value rather than adding.
        x_off, y_off = self._xy_offset(name)
        return self.compensation_at(pos[0] + x_off, pos[1] + y_off, name)

    def compensation_at(self, x, y, probe_name=None):
        """A probe's active map at a bed point, interpolated by Kalico's
        bed_mesh ZMesh; clamped to the grid edge outside it."""
        a = self.active.get(probe_name or self._first_probe())
        if a is None:
            return 0.0
        return a["mesh"].calc_z(x, y)

    # ------------------------------------------------------------------
    # Analysis

    def _analyze(self):
        last = self.last
        xs, ys = last["xs"], last["ys"]
        nx, ny = len(xs), len(ys)
        main, corr = last["main"], last["correction"]
        present = sorted(set(main) & set(corr), key=lambda k: (k[1], k[0]))
        r = [[None] * nx for _ in range(ny)]
        for i, j in present:
            r[j][i] = corr[(i, j)] - main[(i, j)]
        mean = sum(r[j][i] for i, j in present) / len(present)
        res = [[None if v is None else v - mean for v in row] for row in r]
        # Additive fit by backfitting on the measured points. On a complete
        # grid the first pass is the exact row/column-mean least-squares fit;
        # round beds (missing corners) need a few passes.
        cols = {i: [j for j in range(ny) if res[j][i] is not None]
                for i in range(nx)}
        rows = {j: [i for i in range(nx) if res[j][i] is not None]
                for j in range(ny)}
        x_eff, y_eff = [0.0] * nx, [0.0] * ny
        for _ in range(1 if self.last.get("radius") is None else 50):
            x_eff = [sum(res[j][i] - y_eff[j] for j in cols[i]) / len(cols[i])
                     for i in range(nx)]
            # centre by point count (plain mean on a full grid), so the
            # constant stays in the offset and out of the curves
            m = sum(len(cols[i]) * x_eff[i] for i in range(nx)) / \
                len(present)
            x_eff = [v - m for v in x_eff]
            y_eff = [sum(res[j][i] - x_eff[i] for i in rows[j]) / len(rows[j])
                     for j in range(ny)]
        inter = [[None if res[j][i] is None else
                  res[j][i] - x_eff[i] - y_eff[j] for i in range(nx)]
                 for j in range(ny)]

        def rms(vals):
            vals = list(vals)
            return math.sqrt(sum(v * v for v in vals) / len(vals))

        flat = [res[j][i] for i, j in present]
        comp = [[None if v is None else -v for v in row] for row in res]
        self.report = {
            "count": len(present),
            "mean_offset": mean,
            "rms_residual": rms(flat),
            "max_abs_residual": max(abs(v) for v in flat),
            "x_effect_rms": rms(x_eff),
            "x_effect_range": max(x_eff) - min(x_eff),
            "y_effect_rms": rms(y_eff),
            "y_effect_range": max(y_eff) - min(y_eff),
            "interaction_rms": rms(inter[j][i] for i, j in present),
            "xs": xs, "ys": ys,
            "x_effect": x_eff, "y_effect": y_eff,
            "comp_table": comp,                 # None outside a round bed
            "comp_grid": _pad_rows(comp),       # padded for ZMesh
            "suspect_points": self._suspect_points(inter, xs, ys),
            "edge_warnings": [] if self.last.get("radius") is not None else
            (self._edge_jumps("X", xs, x_eff)
             + self._edge_jumps("Y", ys, y_eff)),
        }
        return self.report

    def _suspect_points(self, inter, xs, ys):
        """On the trend-free part of the residual (interaction): points far
        off the median of their neighbours compared with how far points
        typically are (robust MAD threshold) are likely one bad reading."""
        nx, ny = len(xs), len(ys)
        if nx < 3 or ny < 3:
            return []
        dev = {}
        for j in range(ny):
            for i in range(nx):
                if inter[j][i] is None:
                    continue
                nb = [inter[jj][ii]
                      for jj in range(max(0, j - 1), min(ny, j + 2))
                      for ii in range(max(0, i - 1), min(nx, i + 2))
                      if (ii, jj) != (i, j) and inter[jj][ii] is not None]
                if len(nb) >= 3:
                    dev[(i, j)] = inter[j][i] - _median(nb)
        if len(dev) < 5:
            return []
        sigma = 1.4826 * _median([abs(v) for v in dev.values()])
        limit = max(4.0 * sigma, 2.0 * self.tap_tolerance)
        return [{"x": xs[i], "y": ys[j], "comp": -d}
                for (i, j), d in sorted(dev.items(),
                                        key=lambda kv: kv[0][::-1])
                if abs(d) > limit]

    @staticmethod
    def _edge_jumps(axis, pos, eff):
        """Flag an outer row/column whose effect breaks from the trend of the
        two next to it far more than the interior curve ever does: the
        signature of an eddy probe misreading at the plate edge."""
        n = len(eff)
        if n < 5:
            return []
        second = [eff[k - 1] - 2 * eff[k] + eff[k + 1] for k in range(1, n - 1)]
        limit = max(3.0 * _median([abs(v) for v in second[1:-1]]), 0.02)
        warnings = []
        for edge, a, b, side in ((0, 1, 2, "min"), (n - 1, n - 2, n - 3,
                                                    "max")):
            jump = eff[edge] - (2 * eff[a] - eff[b])
            if abs(jump) > limit:
                warnings.append({"axis": axis, "position": pos[edge],
                                 "side": side, "jump": jump})
        return warnings

    def _report_text(self):
        rep = self.report
        lines = [
            "Axis Twist Mesh analysis for '%s' (%d points, residual = "
            "correction - reference):" % (self.last.get("probe", "?"),
                                          rep["count"]),
            "  mean probe offset  %+.4f mm" % rep["mean_offset"],
            "  residual RMS       %.4f mm (max |r| %.4f)"
            % (rep["rms_residual"], rep["max_abs_residual"]),
            "  X effect           RMS %.4f, range %.4f mm"
            % (rep["x_effect_rms"], rep["x_effect_range"]),
            "  Y effect           RMS %.4f, range %.4f mm"
            % (rep["y_effect_rms"], rep["y_effect_range"]),
            "  interaction RMS    %.4f mm" % rep["interaction_rms"],
            "  X curve: " + "  ".join("%.1f:%+.4f" % pv for pv in
                                     zip(rep["xs"], rep["x_effect"])),
            "  Y curve: " + "  ".join("%.1f:%+.4f" % pv for pv in
                                     zip(rep["ys"], rep["y_effect"])),
        ]
        for w in rep["edge_warnings"]:
            lines.append(
                "  Edge warning: %s=%.1f breaks from the trend by %+.4f mm "
                "(possibly the probe misreading near the plate edge; move "
                "mesh_%s inward past it)" % (
                    w["axis"], w["position"], w["jump"], w["side"]))
        if rep["suspect_points"]:
            lines.append(
                "  Suspect points (off the smooth fit by much more than the "
                "rest; re-run or check the bed there): " + ", ".join(
                    "(%.1f, %.1f) %+.4f" % (p["x"], p["y"], p["comp"])
                    for p in rep["suspect_points"][:8]))
        lines.append(self._table_text(
            rep["xs"], rep["ys"], rep["comp_table"],
            "constant offset %+.4f removed" % -rep["mean_offset"]))
        return lines

    @staticmethod
    def _table_text(xs, ys, grid, note):
        lines = ["Per-point compensation (mm, added to the correction probe; "
                 "rows = Y, columns = X; %s):" % note,
                 "   Y \\ X " + "".join("%9.1f" % x for x in xs)]
        for j in range(len(ys) - 1, -1, -1):
            lines.append("%8.1f " % ys[j] + "".join(
                "        ." if v is None else "%+9.4f" % v for v in grid[j]))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Commands

    def _parse_probe_param(self, gcmd):
        if not self.correction_probes:
            raise gcmd.error("axis_twist_mesh: no correction probe; set "
                             "correction_probe in [axis_twist_mesh]")
        name = gcmd.get("PROBE", self._first_probe()).strip()
        if name not in self.correction_probes:
            raise gcmd.error("PROBE must be one of correction_probe: %s"
                             % ", ".join(self.correction_probes))
        return name

    def cmd_CALIBRATE(self, gcmd):
        if self._manual is not None:
            raise gcmd.error("A manual axis twist calibration is in progress "
                             "(ACCEPT each point, or ABORT)")
        probe_name = self._parse_probe_param(gcmd)
        if gcmd.get("PROFILE", None) is not None:
            raise gcmd.error("Maps are saved per probe now (like "
                             "[axis_twist_compensation <probe>]); use PROBE= "
                             "instead of PROFILE=")
        which = gcmd.get("PROBES", "both").lower()
        if which not in ("both", "main", "correction"):
            raise gcmd.error("PROBES must be both, main or correction")
        if gcmd.get("REFERENCE", None) is not None:
            reference = _parse_reference(gcmd.get("REFERENCE"))
        else:
            reference = (self.reference, self.reference_probe)
        if reference[1] in self.correction_probes:
            raise gcmd.error("The reference can't be a correction probe")
        if reference[0] == "manual" and which != "both":
            raise gcmd.error("REFERENCE=manual always measures both")
        grid = self._grid_key()
        if which != "both" and not (
                self.last and self.last["grid"] == grid
                and self.last["reference"] == reference
                and self.last["probe"] == probe_name):
            raise gcmd.error("PROBES=%s re-probes one side of the last "
                             "calibration of the same probe, grid and "
                             "reference; run PROBES=both first" % which)

        toolhead = self.printer.lookup_object("toolhead")
        eventtime = self.printer.get_reactor().monotonic()
        homed = toolhead.get_status(eventtime)["homed_axes"]
        if not all(a in homed for a in "xyz"):
            raise gcmd.error("Must home X, Y and Z before "
                             "AXIS_TWIST_MESH_CALIBRATE")
        roles = [r for r in ("main", "correction")
                 if which in ("both", r)]
        if reference[0] == "touch":
            modes = self._single_modes(probe_name)
            if modes is None:
                raise gcmd.error(
                    "REFERENCE=touch needs '%s' to be the Cartographer or "
                    "Beacon plugin" % probe_name)
            by_role = {"main": modes[0], "correction": modes[1]}
            probes = [(role, by_role[role]) for role in roles]
        else:
            if reference[0] == "manual":
                roles = ["correction"]
            names = {"main": reference[1], "correction": probe_name}
            probes = [(role, self._require_probe(
                gcmd, "reference" if role == "main" else "correction_probe",
                names[role])) for role in roles]
            probes = [(role, self._correction_reader(p)
                       if role == "correction" else p)
                      for role, p in probes]
        for role, probe in probes:
            self._check_reachable(gcmd, toolhead, role, probe)
            self._check_touch_area(gcmd, probe)
            precheck = getattr(probe, "precheck", None)
            if precheck is not None:
                precheck(gcmd.error)
        if reference[0] == "manual":
            self._check_reachable(gcmd, toolhead, "nozzle", _Nozzle())
            manual_probe.verify_no_manual_probe(self.printer)

        if self._kalico_manager() is None:
            self._install_hook()        # older Kalico: raw via the hook
        job = {"which": which, "reference": reference,
               "probe": probe_name, "grid": grid,
               "xs": self._grid_axis(0), "ys": self._grid_axis(1)}
        if reference[0] == "manual":
            self._start_manual(gcmd, toolhead, probes[0][1], job)
            return
        self.calibrating = True
        suspended = self._suspend_kalico_map(probe_name)
        try:
            results = self._probe_grid(gcmd, toolhead, probes)
        finally:
            self.calibrating = False
            self._resume_kalico_map(suspended)
        self._finish_calibration(
            gcmd, job, {role: h for role, (h, _) in results.items()})

    def _finish_calibration(self, gcmd, job, heights):
        """Analyze, save as the probe's map and activate it."""
        xs, ys, probe_name = job["xs"], job["ys"], job["probe"]
        last = dict(self.last) if job["which"] != "both" else {}
        last.update({"xs": xs, "ys": ys, "reference": job["reference"],
                     "probe": probe_name, "grid": job["grid"],
                     "radius": self.radius})
        last.update(heights)
        self.last = last
        self._analyze()
        rep = self.report

        # Kalico's maps hold the mean-removed difference; apply_offset
        # folds the constant probe offset in as well
        shift = -rep["mean_offset"] if self.apply_offset else 0.0
        prof = {
            "x_count": len(xs), "y_count": len(ys),
            "min_x": xs[0], "max_x": xs[-1],
            "min_y": ys[0], "max_y": ys[-1],
            "mesh_radius": self.radius, "mesh_origin": self.origin,
            "reference": _reference_value(job["reference"]),
            "points": [[v + shift for v in row] for row in rep["comp_grid"]],
            "algo": None, "mesh_x_pps": None, "mesh_y_pps": None,
            "tension": None,
        }
        lines = self._report_text()
        section = self._store_map(probe_name, prof)
        try:
            self._activate(probe_name, gcmd.error)
            state = "saved as [%s] and applied" % section
        except gcmd.error as exc:
            state = "saved as [%s] but NOT applied: %s" % (section, exc)
        if rep["edge_warnings"] or rep["suspect_points"]:
            lines.append("Review the warnings above; AXIS_TWIST_MESH_CLEAR "
                         "turns compensation off.")
        lines.append("Result %s. SAVE_CONFIG writes it to printer.cfg." %
                     state)
        gcmd.respond_info("\n".join(lines))

    # ------------------------------------------------------------------
    # Manual (paper) reference: interleaved, one point at a time. The
    # correction probe measures a point, then the nozzle is lowered there by
    # hand with TESTZ and ACCEPT; ACCEPT moves on to the next point.

    def _start_manual(self, gcmd, toolhead, corr, job):
        nodes = self._grid_nodes()
        self._manual = {
            "gcmd": gcmd, "toolhead": toolhead, "corr": corr, "job": job,
            "nodes": nodes, "index": 0, "main": {}, "correction": {},
            "extra": 0, "offsets": corr.get_offsets()[:3],
            "tap_gcmd": self._tap_gcmd(gcmd),
        }
        self.calibrating = True
        self._suspended = self._suspend_kalico_map(job["probe"])
        gcmd.respond_info(
            "axis_twist_mesh: manual calibration, %d points. At each point "
            "'%s' measures first, then lower the nozzle onto the paper with "
            "TESTZ Z=... and send ACCEPT (ABORT cancels; the current profile "
            "stays)." % (len(nodes), getattr(corr, "probe_name", "?")))
        self._manual_step()

    def _manual_step(self):
        m = self._manual
        try:
            if m["index"] >= len(m["nodes"]):
                self._manual = None
                self.calibrating = False
                self._resume_kalico_map(self._suspended)
                self._suspended = None
                # nozzle is still on the paper: lift before anything moves
                m["toolhead"].manual_move(
                    [None, None, self.horizontal_move_z], self.lift_speed)
                m["gcmd"].respond_info(
                    "axis_twist_mesh: manual points done; correction probe "
                    "needed %d extra readings" % m["extra"])
                self._finish_calibration(
                    m["gcmd"], m["job"],
                    {"main": m["main"], "correction": m["correction"]})
                return
            ix, iy, x, y = m["nodes"][m["index"]]
            th, corr = m["toolhead"], m["corr"]
            x_off, y_off, z_off = m["offsets"]
            th.manual_move([None, None, self.horizontal_move_z],
                           self.lift_speed)
            th.manual_move([x - x_off, y - y_off, None], self.speed)
            begin = getattr(corr, "multi_probe_begin", None)
            end = getattr(corr, "multi_probe_end", None)
            if begin is not None:
                begin()
            try:
                z, taps = self._agreeing_taps(m["gcmd"], m["tap_gcmd"], th,
                                              corr, x, y, "correction")
            finally:
                if end is not None:
                    end()
            m["extra"] += len(taps) - MIN_AGREEING_TAPS
            m["correction"][(ix, iy)] = z - z_off
            th.manual_move([None, None, self.horizontal_move_z],
                           self.lift_speed)
            th.manual_move([x, y, None], self.speed)
            m["gcmd"].respond_info(
                "axis_twist_mesh: point %d of %d at (%.1f, %.1f): lower the "
                "nozzle onto the paper, then ACCEPT"
                % (m["index"] + 1, len(m["nodes"]), x, y))
            self._new_manual_probe(m["gcmd"], self._manual_accepted)
        except Exception:
            self._abort_manual()
            raise

    def _new_manual_probe(self, gcmd, callback):
        manual_probe.ManualProbeHelper(self.printer, gcmd, callback)

    def _manual_accepted(self, kin_pos):
        m = self._manual
        if m is None:
            return
        if kin_pos is None:
            self._abort_manual()
            m["gcmd"].respond_info("axis_twist_mesh: manual calibration "
                                   "aborted; nothing was changed")
            return
        ix, iy = m["nodes"][m["index"]][:2]
        m["main"][(ix, iy)] = kin_pos[2]
        m["index"] += 1
        self._manual_step()

    def _abort_manual(self, *args):
        if self._manual is not None:
            self._manual = None
            self.calibrating = False
            self._resume_kalico_map(self._suspended)
            self._suspended = None

    # ------------------------------------------------------------------
    # Kalico's own axis_twist_compensation (line-based or mesh-based)

    def _native_owner(self):
        return getattr(self._orig_twist_fn, "__self__", None)

    def _legacy_lines(self):
        """Older Kalico's [axis_twist_compensation] with z_compensations /
        zy_compensations set, or None."""
        owner = self._native_owner() or self.printer.lookup_object(
            "axis_twist_compensation", None)
        if owner is None or owner is self or \
                getattr(owner, "get_compensation", None):
            return None
        if getattr(owner, "z_compensations", None) or \
                getattr(owner, "zy_compensations", None):
            return owner
        return None

    def cmd_IMPORT(self, gcmd):
        source = gcmd.get("SOURCE", "legacy").lower()
        if source != "legacy":
            raise gcmd.error("SOURCE must be legacy (maps are already stored "
                             "in Kalico's [axis_twist_compensation <probe>] "
                             "format)")
        probe_name = self._parse_probe_param(gcmd)
        self._install_hook()
        owner = self._legacy_lines()
        if owner is None:
            raise gcmd.error("No z_compensations / zy_compensations in "
                             "[axis_twist_compensation] to import")
        prof = self._legacy_profile(gcmd, owner, probe_name)
        prof.update({"mesh_radius": None, "mesh_origin": None})
        section = self._store_map(probe_name, prof, layout=False)
        self._activate(probe_name, gcmd.error)
        gcmd.respond_info(
            "axis_twist_mesh: imported the X / Y lines as [%s] (%dx%d) and "
            "applied it to '%s' in place of the lines. When you're happy with "
            "it, delete z_compensations / zy_compensations from "
            "[axis_twist_compensation] (they also shift the reference "
            "probe), then SAVE_CONFIG." % (
                section, prof["x_count"], prof["y_count"], probe_name))

    def _legacy_profile(self, gcmd, owner, probe_name):
        """Rebuild the old X / Y lines exactly as a direct (bilinear) map,
        including their linear extension past the ends, over the machine's
        XY range (the same conversion mesh-based Kalico does)."""
        toolhead = self.printer.lookup_object("toolhead")
        status = toolhead.get_kinematics().get_status(
            self.printer.get_reactor().monotonic())
        lo, hi = status["axis_minimum"], status["axis_maximum"]
        coords, corrections = [], []
        for axis, attr, start_attr, end_attr in (
                (0, "z_compensations", "compensation_start_x",
                 "compensation_end_x"),
                (1, "zy_compensations", "compensation_start_y",
                 "compensation_end_y")):
            values = list(getattr(owner, attr, None) or [])
            if not values:
                coords.append([lo[axis], hi[axis]])
                corrections.append([0.0, 0.0])
                continue
            start = getattr(owner, start_attr)
            end = getattr(owner, end_attr)
            if len(values) < 2 or start is None or end is None or \
                    start == end:
                raise gcmd.error("Invalid %s in [axis_twist_compensation]"
                                 % attr)
            if end < start:
                start, end, values = end, start, values[::-1]
            spacing = (end - start) / (len(values) - 1)
            first = int(math.floor((lo[axis] - start) / spacing))
            last = int(math.ceil((hi[axis] - start) / spacing))
            coords.append([start + k * spacing
                           for k in range(first, last + 1)])
            line = []
            for k in range(first, last + 1):
                e = max(0, min(k, len(values) - 2))
                line.append(values[e] + (k - e) * (values[e + 1] - values[e]))
            corrections.append(line)
        x_off, y_off = self._xy_offset(probe_name)
        return {
            "x_count": len(coords[0]), "y_count": len(coords[1]),
            "min_x": coords[0][0] + x_off, "max_x": coords[0][-1] + x_off,
            "min_y": coords[1][0] + y_off, "max_y": coords[1][-1] + y_off,
            "algo": "direct", "mesh_x_pps": 0, "mesh_y_pps": 0,
            "tension": self.tension, "reference": "manual",
            "points": [[cx + cy for cx in corrections[0]]
                       for cy in corrections[1]],
        }

    # ------------------------------------------------------------------

    def cmd_PROFILE(self, gcmd):
        """LOAD=<probe> turns a probe's saved map back on; REMOVE=<probe>
        deletes it (or an old [axis_twist_mesh <name>] profile)."""
        name = gcmd.get("LOAD", None)
        if name is not None:
            name = name.strip()
            self._activate(name, gcmd.error)
            gcmd.respond_info("axis_twist_mesh: map for '%s' applied" % name)
            return
        name = gcmd.get("REMOVE", None)
        if name is not None:
            name = name.strip()
            configfile = self.printer.lookup_object("configfile")
            if name in self.maps:
                section = self.maps[name]["section"]
                configfile.remove_section(section)
                self.maps = {p: m for p, m in self.maps.items() if p != name}
                self.active = {p: e for p, e in self.active.items()
                               if p != name}
                self._update_integration()
            elif name in self.old_profiles:
                section = "%s %s" % (self.name, name)
                configfile.remove_section(section)
                del self.old_profiles[name]
            else:
                raise gcmd.error("axis_twist_mesh: no saved map for '%s'"
                                 % name)
            gcmd.respond_info("axis_twist_mesh: [%s] removed. SAVE_CONFIG "
                              "writes the change to printer.cfg." % section)
            return
        raise gcmd.error("Use LOAD=<probe> or REMOVE=<probe>. Saved maps: %s"
                         % (", ".join(sorted(self.maps)) or "none"))

    def cmd_CLEAR(self, gcmd):
        if gcmd.get("PROBE", None) is None:
            self.active = {}
            self._update_integration()
            gcmd.respond_info("axis_twist_mesh: compensation off")
            return
        probe_name = self._parse_probe_param(gcmd)
        self.active = {p: e for p, e in self.active.items()
                       if p != probe_name}
        self._update_integration()
        gcmd.respond_info("axis_twist_mesh: compensation off for '%s'"
                          % probe_name)

    def _profile_table(self, prof):
        xs = _axis(prof["min_x"], prof["max_x"], prof["x_count"])
        ys = _axis(prof["min_y"], prof["max_y"], prof["y_count"])
        grid = [list(r) for r in prof["points"]]
        if prof.get("mesh_radius") is not None:
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    if not _in_round(x, y, prof["mesh_origin"],
                                     prof["mesh_radius"]):
                        grid[j][i] = None
        return xs, ys, grid

    def cmd_TABLE(self, gcmd):
        probe_name = gcmd.get("PROBE", self._first_probe())
        prof = self.maps.get(probe_name)
        if prof is None:
            raise gcmd.error("axis_twist_mesh: no saved map for '%s'"
                             % probe_name)
        xs, ys, grid = self._profile_table(prof)
        fmt = gcmd.get("FORMAT", "grid").lower()
        if fmt == "grid":
            gcmd.respond_info("[%s], reference %s\n" % (
                prof["section"], prof["reference"]) + self._table_text(
                    xs, ys, grid, "constant offset not included"
                    if not self.apply_offset else "includes the offset"))
        elif fmt == "twist":
            comps = []
            for i in range(len(xs)):
                col = [grid[j][i] for j in range(len(ys))
                       if grid[j][i] is not None]
                comps.append(sum(col) / len(col))
            x_off = self._xy_offset(probe_name)[0] \
                if self._resolve_probe(probe_name) else 0.0
            gcmd.respond_info(
                "X-only z_compensations equivalent of [%s] (one line, "
                "applies to ALL probes in older Kalico):\n"
                "z_compensations: %s\ncompensation_start_x: %.3f\n"
                "compensation_end_x: %.3f" % (
                    prof["section"], ", ".join("%.6f" % c for c in comps),
                    xs[0] - x_off, xs[-1] - x_off))
        else:
            raise gcmd.error("FORMAT must be grid or twist")

    def cmd_ANALYZE(self, gcmd):
        if self.report is None:
            raise gcmd.error("No calibration since restart. Use "
                             "AXIS_TWIST_MESH_TABLE to see a saved profile.")
        gcmd.respond_info("\n".join(self._report_text()))

    def _touch_area_status(self):
        if not self._touch_ref:
            return ""
        modes = self._single_modes()
        area = modes[0].touch_area() if modes else None
        if area is None:
            return ""
        fits = all(area[0] - 0.01 <= x <= area[1] + 0.01 and
                   area[2] - 0.01 <= y <= area[3] + 0.01
                   for _, _, x, y in self._grid_nodes())
        return ("\nCartographer touch area: X=[%.2f, %.2f] Y=[%.2f, %.2f] "
                "(grid %s)" % (area + ("fits" if fits else "DOES NOT FIT",)))

    def cmd_STATUS(self, gcmd):
        if self.radius is None:
            grid = "grid %g,%g..%g,%g probe_count %d,%d" % (
                self.mesh_min + self.mesh_max + self.probe_count)
        else:
            grid = "round grid r=%g around %g,%g, %d points" % (
                (self.radius,) + self.origin + (len(self._grid_nodes()),))
        lines = ["Axis Twist Mesh: %s" % grid,
                 "reference: %s (%s)" % (
                     self._main_label(),
                     "found" if self._main_found() else "MISSING")]
        if not self.correction_probes:
            lines.append("correction probe: none - set correction_probe "
                         "(Kalico's default probe isn't usable)")
        for probe_name in self.correction_probes:
            a = self.active.get(probe_name)
            lines.append("correction probe '%s' (%s): %s" % (
                probe_name,
                "found" if self._resolve_probe(probe_name) else "MISSING",
                "[%s] %dx%d, %s" % (a["name"], len(a["xs"]), len(a["ys"]),
                                    a["mesh"].get_mesh_params()["algo"])
                if a else "none (compensation off)"))
        lines.append("saved maps: %s" % (", ".join(
            "[%s]" % m["section"] for _, m in sorted(self.maps.items()))
            or "none"))
        if self.old_profiles:
            lines.append("old profiles not used (AXIS_TWIST_MESH_PROFILE "
                         "REMOVE=name deletes them): %s"
                         % ", ".join(sorted(self.old_profiles)))
        if self._legacy_lines() is not None:
            lines.append("note: z_compensations / zy_compensations in "
                         "[axis_twist_compensation] still apply to probes "
                         "without a map here (e.g. the reference); "
                         "AXIS_TWIST_MESH_IMPORT turns them into a map")
        if self._manual:
            lines.append("manual calibration in progress: point %d of %d"
                         % (self._manual["index"] + 1,
                            len(self._manual["nodes"])))
        gcmd.respond_info("\n".join(lines) + self._touch_area_status())

    def _cartographer_twist_adapter(self, probe_name):
        """The axis-twist adapter Cartographer's own BED_MESH_CALIBRATE uses
        (None if it was loaded without [axis_twist_compensation]), or 'n/a'
        when the Cartographer plugin isn't in use."""
        obj = self.printer.lookup_object(probe_name, None)
        for reg in getattr(obj, "macros", None) or []:
            macro = getattr(reg, "macro", None)
            if getattr(reg, "name", "") == "BED_MESH_CALIBRATE" and \
                    hasattr(macro, "axis_twist_compensation"):
                return macro.axis_twist_compensation
        return "n/a"

    def cmd_CHECK(self, gcmd):
        ok, lines = self._run_check()
        gcmd.respond_info("\n".join(lines))

    def _handle_ready(self):
        """Self-test at every start: an add-on can be broken by a Kalico or
        probe-plugin update without any error, so make that loud."""
        for old, probe_name in self._converted:
            self.gcode.respond_info(
                "axis_twist_mesh: converted old profile [%s %s] to [%s]; run "
                "SAVE_CONFIG to keep it" % (
                    self.name, old, self.maps[probe_name]["section"]))
        if not self.active:
            return
        try:
            ok, lines = self._run_check()
        except Exception as exc:
            ok, lines = False, ["self-test failed: %s" % (exc,)]
        if ok:
            logging.info("axis_twist_mesh: self-test passed")
            return
        logging.warning("axis_twist_mesh: self-test failed:\n%s",
                        "\n".join(lines))
        self.gcode.respond_raw(
            "!! axis_twist_mesh: compensation is NOT reaching the correction "
            "probe, so bed meshes are uncorrected. Run AXIS_TWIST_MESH_CHECK "
            "for details (a Kalico or probe plugin update may have changed "
            "how probes are corrected).\n")

    def _kalico_call_path(self):
        """Does Kalico's own probe code still ask axis_twist_compensation for
        each reading the way this add-on hooks it? True / False / None
        (can't tell)."""
        try:
            import inspect
            try:
                from . import probe as probe_module
            except ImportError:
                from klippy.extras import probe as probe_module
            src = inspect.getsource(probe_module.PrinterProbe)
        except Exception:
            return None
        return ('lookup_object(\n' in src or 'lookup_object(' in src) and \
            '"axis_twist_compensation"' in src and \
            '.get_z_compensation_value(' in src

    def _run_check(self):
        lines = ["Axis Twist Mesh check:"]
        atc = self.printer.lookup_object("axis_twist_compensation", None)
        wrapped = self._hook_active()
        needed = self._needs_wrap()
        hooked = wrapped or (atc is not None and not needed)
        lines.append("  hook on axis_twist_compensation: %s" % (
            "installed" if wrapped else
            "not needed (Kalico applies the maps itself)" if hooked else
            "MISSING"))
        if self._kalico_handles:
            lines.append("  Kalico applies: %s" % ", ".join(
                sorted(self._kalico_handles)))
        ok = hooked and bool(self.active)
        if not self.active:
            lines.append("  no active map - run AXIS_TWIST_MESH_PROFILE "
                         "LOAD=name")
        for probe_name in self.correction_probes:
            a = self.active.get(probe_name)
            corr = self._resolve_probe(probe_name)
            lines.append("  '%s': %s" % (
                probe_name, "[%s] %dx%d" % (a["name"], len(a["xs"]),
                                           len(a["ys"])) if a else
                "no active map"))
            if corr is None:
                continue
            adapter = self._cartographer_twist_adapter(probe_name)
            beacon = corr if self._is_beacon(corr) else None
            if beacon is not None:
                found = getattr(beacon, "mod_axis_twist_comp", None) \
                    is not None
                lines.append("    Beacon twist hook: %s" % (
                    "present" if found else "MISSING - Beacon started "
                    "without axis_twist_compensation; RESTART Klipper"))
                ok &= found
            elif self._is_cartographer_plugin(corr) or adapter != "n/a":
                if adapter is None or adapter == "n/a":
                    lines.append(
                        "    Cartographer BED_MESH_CALIBRATE twist hook: "
                        "MISSING - " + (self._support_problem(probe_name) or
                                        "Cartographer was loaded without an "
                                        "[axis_twist_compensation] section, "
                                        "so its scan meshes never ask for a "
                                        "correction. Add the section and "
                                        "restart."))
                    ok = False
                else:
                    lines.append("    Cartographer BED_MESH_CALIBRATE twist "
                                 "hook: present")
            if not hooked or a is None:
                continue
            x_off, y_off = self._xy_offset(probe_name)
            xs, ys = a["xs"], a["ys"]
            pts = [(xs[0], ys[0]), (xs[-1], ys[0]),
                   ((xs[0] + xs[-1]) / 2.0, (ys[0] + ys[-1]) / 2.0),
                   (xs[0], ys[-1]), (xs[-1], ys[-1])]
            for bx, by in pts:
                nx, ny = bx - x_off, by - y_off
                want = self.compensation_at(bx, by, probe_name)
                if beacon is not None:
                    # Beacon's own lookup, as used for every proximity sample
                    got = beacon.get_z_compensation_value([nx, ny, 0.0])
                    via = "via Beacon"
                elif adapter not in (None, "n/a"):
                    got = adapter.get_z_compensation_value(x=nx, y=ny)
                    via = "via Cartographer"
                else:
                    # the call Kalico's probe code makes for this probe
                    got = atc.get_z_compensation_value([nx, ny, 0.0], corr) \
                        if self._native_takes_probe() else \
                        self._as_kalico_reading(atc, corr, [nx, ny, 0.0])
                    via = "via Kalico"
                match = abs(got - want) < 1e-9
                ok &= match
                lines.append("    bed (%.1f, %.1f): gets %+.4f mm %s, "
                             "expected %+.4f %s" % (
                                 bx, by, got, via, want,
                                 "OK" if match else "MISMATCH"))
        uses_kalico = any(
            self._resolve_probe(n) is not None and
            not self._is_beacon(self._resolve_probe(n)) and
            not self._is_cartographer_plugin(self._resolve_probe(n))
            for n in self.active)
        if uses_kalico:
            path = self._kalico_call_path()
            if path is False:
                ok = False
            lines.append("  Kalico probe code asks axis_twist_compensation "
                         "per reading: %s" % {
                             True: "yes", None: "can't tell",
                             False: "NO - Kalico changed how probes are "
                             "corrected; this add-on needs an update"}[path])
        lines.append("Result: %s" % (
            "compensation reaches the correction probes" if ok
            else "NOT working, see above"))
        return ok, lines

    def _native_takes_probe(self):
        return self._kalico_manager() is not None

    def _as_kalico_reading(self, atc, corr, pos):
        """Older Kalico: a probe's reading is identified by its pending
        multi-probe session, so simulate one for the check."""
        pending = getattr(corr, "multi_probe_pending", None)
        if pending is None:
            return atc.get_z_compensation_value(pos)
        corr.multi_probe_pending = True
        try:
            return atc.get_z_compensation_value(pos)
        finally:
            corr.multi_probe_pending = pending

    def get_status(self, eventtime):
        rep = self.report
        a = self.active.get(self._first_probe())
        status = {
            "profile_name": a["name"] if a else "",
            "active": {p: e["name"] for p, e in self.active.items()},
            "calibrating": self.calibrating,
            # like bed_mesh, for visualizing the first probe's active map
            "probed_matrix": a["mesh"].get_probed_matrix() if a else [[]],
            "mesh_matrix": a["mesh"].get_mesh_matrix() if a else [[]],
            "mesh_params": dict(a["mesh"].get_mesh_params()) if a else {},
            "maps": {
                probe: {"section": m["section"],
                        "x_count": m["x_count"], "y_count": m["y_count"],
                        "min": [m["min_x"], m["min_y"]],
                        "max": [m["max_x"], m["max_y"]],
                        "mesh_radius": m.get("mesh_radius"),
                        "reference_probe": m["reference"]}
                for probe, m in self.maps.items()},
        }
        if rep is not None:
            status["last_calibration"] = {
                k: rep[k] for k in ("count", "mean_offset", "rms_residual",
                                    "x_effect_range", "y_effect_range",
                                    "interaction_rms")}
        return status


def load_config(config):
    return AxisTwistMesh(config)
