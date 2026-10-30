"""Tests for axis_twist_mesh against Kalico's real config parser, ProbeList
and G-code parameter parser, with simulated probes and toolhead.

Run from a Kalico checkout:  KALICO=~/kalico pytest test_axis_twist_mesh.py
"""
import configparser
import math
import os
import sys

import pytest

KALICO = os.path.expanduser(os.environ.get("KALICO", "~/kalico"))
sys.path.insert(0, KALICO)
sys.path.insert(0, os.path.dirname(__file__))

from klippy import configfile, gcode as kgcode  # noqa: E402
from klippy.extras import probe as kprobe  # noqa: E402

import axis_twist_mesh  # noqa: E402


def twist(nozzle_x):
    # gantry twist seen by the offset probe, a function of nozzle X
    return 0.0004 * (nozzle_x - 110.0) - 0.000004 * (nozzle_x - 110.0) ** 2


def bed(x, y):
    return 0.05 * math.sin(x / 40.0) + 0.03 * math.cos(y / 30.0)


class CmdError(Exception):
    pass


class FakeGcmd:
    error = CmdError

    def __init__(self, params=None):
        self.params = {k.upper(): str(v) for k, v in (params or {}).items()}
        self.out = []

    def get_int(self, name, default=configfile.sentinel, **kw):
        v = self.get(name, default)
        return None if v is None else int(v)

    def get_float(self, name, default=configfile.sentinel, **kw):
        v = self.get(name, default)
        return None if v is None else float(v)

    def get(self, name, default=configfile.sentinel):
        if name in self.params:
            return self.params[name]
        if default is configfile.sentinel:
            raise CmdError("missing %s" % name)
        return default

    def get_command_parameters(self):
        return dict(self.params)

    def respond_info(self, msg, log=True):
        self.out.append(msg)


class FakeToolhead:
    def __init__(self):
        self.pos = [0.0, 0.0, 10.0, 0.0]
        self.moves = []
        self.clock = 0.0
        self.history = []   # (time, x, y, z) after each move

    def manual_move(self, coord, speed):
        for i, v in enumerate(coord):
            if v is not None:
                self.pos[i] = v
        self.clock += 0.05
        self.moves.append(tuple(self.pos[:3]))
        self.history.append((self.clock,) + tuple(self.pos[:3]))

    def get_max_velocity(self):
        return 300.0, 3000.0

    def wait_moves(self):
        pass

    def dwell(self, t):
        self.clock += t

    def register_lookahead_callback(self, cb):
        cb(self.clock)

    def get_last_move_time(self):
        return self.clock

    def get_position(self):
        return list(self.pos)

    def get_status(self, eventtime):
        return {"homed_axes": "xyz"}

    hotend_temp = 25.0

    def get_extruder(self):
        th = self

        class Heater:
            def get_status(self, eventtime):
                return {"temperature": th.hotend_temp}

        class Extruder:
            def get_heater(self):
                return Heater()
        return Extruder()

    def get_kinematics(self):
        th = self

        class K:
            def get_status(self, eventtime):
                return {"axis_minimum": [0, 0, -5, 0],
                        "axis_maximum": [235, 235, 250, 0]}
        return K()


class SimProbe:
    """Duck-typed PrinterProbe: returns trigger Z of toolhead (before
    z_offset) plus any axis_twist_compensation, like Kalico's _probe()."""

    def __init__(self, printer, name, offsets, err_fn):
        self.printer, self.probe_name = printer, name
        self.offsets, self.err_fn = offsets, err_fn
        self.is_default_probe = False
        self.begun = 0
        self.multi_probe_pending = False
        self.speed = 5.0
        self.scan_count = 0
        sim = self

        class Collector:
            def __init__(self):
                self.start = sim.printer.objects["toolhead"].clock

            def sample_rate(self):
                return 1000.0

            def finish(self, before_time):
                sim.scan_count += 1
                th = sim.printer.objects["toolhead"]
                out = []
                for t, nx, ny, nz in th.history:
                    if self.start <= t <= before_time:
                        bx, by = nx + sim.offsets[0], ny + sim.offsets[1]
                        h = bed(bx, by) + sim.err_fn(nx, ny)
                        out.append(kprobe.ProbeDistanceResult(t, nz - h))
                return out

        class Mcu:
            def begin_collect_distance(self):
                return Collector()
        self.mcu_probe = Mcu()

    def get_lift_speed(self, gcmd=None):
        return 10.0

    def get_offsets(self):
        return self.offsets

    def multi_probe_begin(self, always_restore_toolhead=False):
        self.begun += 1
        self.multi_probe_pending = True

    def multi_probe_end(self):
        if self.multi_probe_pending:
            self.begun -= 1
        self.multi_probe_pending = False

    def run_probe(self, gcmd, retry_session=None):
        # Same begin/end bracketing as Kalico's PrinterProbe.run_probe
        must = not self.multi_probe_pending
        if must:
            self.multi_probe_begin()
        try:
            return self._read()
        finally:
            if must:
                self.multi_probe_end()

    def _read(self):
        th = self.printer.objects["toolhead"]
        nx, ny = th.pos[0], th.pos[1]
        bx, by = nx + self.offsets[0], ny + self.offsets[1]
        z = bed(bx, by) + self.offsets[2] + self.err_fn(nx, ny)
        atc = self.printer.objects.get("axis_twist_compensation")
        if atc is not None:
            z += atc.get_z_compensation_value([nx, ny, z])
        return [nx, ny, z]


class FakeConfigfile:
    """Records configfile.set / remove_section like Kalico's autosave, and
    renders the result as printer.cfg text (what SAVE_CONFIG would write)."""

    def __init__(self):
        self.sections = {}

    def set(self, section, option, value):
        self.sections.setdefault(section, {})[option] = str(value)

    def remove_section(self, section):
        self.sections.pop(section, None)

    def render(self):
        out = []
        for sec, opts in self.sections.items():
            out.append("[%s]" % sec)
            for k, v in opts.items():
                out.append("%s = %s" % (k, v))
            out.append("")
        return "\n".join(out)


class FakePrinter:
    def __init__(self, cfgfile):
        self.objects = {"toolhead": FakeToolhead(),
                        "configfile": FakeConfigfile()}
        self.cfgfile = cfgfile
        self.handlers = {}
        self.commands = {}
        printer = self

        class G:
            def register_command(self, name, func, desc=None):
                printer.commands[name] = func

            def register_mux_command(self, *a, **k):
                pass

            def create_gcode_command(self, cmd, line, params):
                return FakeGcmd(params)
        self.objects["gcode"] = G()

    def lookup_object(self, name, default=configfile.sentinel):
        if name in self.objects:
            return self.objects[name]
        if default is configfile.sentinel:
            raise KeyError(name)
        return default

    def load_object(self, config, name):
        if name in self.objects:
            return self.objects[name]

        class Macro:
            def load_template(self, config, option, default=None):
                return None
        return Macro()

    def add_object(self, name, obj):
        self.objects[name] = obj

    def register_event_handler(self, ev, cb):
        self.handlers.setdefault(ev, []).append(cb)

    command_error = CmdError

    def get_start_args(self):
        return {"config_file": self.cfgfile}

    def get_reactor(self):
        class R:
            def monotonic(self):
                return 0.0
        return R()


CFG = """
[axis_twist_mesh]
mesh_min: 20, 20
mesh_max: 200, 200
probe_count: 7, 5
main_probe: load_cell_probe
correction_probe: carto
"""


def make(tmp_path, cfg=CFG, carto_err=None, carto_offsets=(0.0, 15.0, 2.0)):
    cfgfile = tmp_path / "printer.cfg"
    cfgfile.write_text(cfg)
    printer = FakePrinter(str(cfgfile))
    fc = configparser.RawConfigParser()
    fc.read_string(cfg)
    access = {}
    config = configfile.ConfigWrapper(printer, fc, access, "axis_twist_mesh")
    # Real Kalico ProbeList with simulated probes registered in it
    plist = kprobe.ProbeList.get_list(printer)
    lc = SimProbe(printer, "load_cell_probe", (0.0, 0.0, -0.1),
                  lambda nx, ny: 0.0)
    carto = SimProbe(printer, "carto", carto_offsets,
                     carto_err or (lambda nx, ny: 0.3 + twist(nx)))
    plist.probes = {"load_cell_probe": lc, "carto": carto}
    atm = axis_twist_mesh.load_config(config)
    printer.access = access
    return printer, atm


def act(atm, probe=None):
    """The active map of a correction probe (first one by default)."""
    return atm.active.get(probe or atm.correction_probes[0])


def connect(printer):
    for h in printer.handlers.get("klippy:connect", []):
        h()


def saved_cfg(printer, base=CFG):
    """printer.cfg after SAVE_CONFIG: base config + autosaved sections."""
    return base + "\n" + printer.objects["configfile"].render()


# ---------------------------------------------------------------------------
# Config and probing

def test_config_parse_and_validation(tmp_path):
    _, atm = make(tmp_path)
    assert atm.probe_count == (7, 5)
    assert atm._grid_axis(0) == [20, 50, 80, 110, 140, 170, 200]


def test_reference_probe_options(tmp_path):
    _, atm = make(tmp_path, CFG.replace("main_probe: load_cell_probe",
                                        "main_probe: other_probe"))
    assert (atm.reference, atm.reference_probe) == ("probe", "other_probe")
    _, atm = make(tmp_path, CFG.replace("main_probe: load_cell_probe\n", "")
                  + "reference: load_cell\n")
    assert atm.reference_probe == "load_cell_probe"
    with pytest.raises(configparser.Error, match="can't also be"):
        make(tmp_path, CFG.replace("main_probe: load_cell_probe\n", "")
             + "reference: carto\n")


def test_old_options_accepted(tmp_path):
    make(tmp_path, CFG + "compensation_file: x.json\npair_tolerance: 0.5\n"
         "correction_method: scan\nscan_height: 2\n")


def test_calibrate_recovers_twist(tmp_path):
    printer, atm = make(tmp_path)
    connect(printer)
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    rep = atm.report
    tw = [twist(x) for x in atm._grid_axis(0)]
    m = sum(tw) / len(tw)
    assert abs(rep["mean_offset"] - (0.3 + m)) < 1e-9
    assert rep["interaction_rms"] < 1e-9 and rep["y_effect_rms"] < 1e-9
    for e, t in zip(rep["x_effect"], tw):
        assert abs(e - (t - m)) < 1e-9
    th = printer.objects["toolhead"]
    assert (20.0, 20.0 - 15.0, 5.0) in th.moves
    assert "Per-point compensation" in g.out[-1]
    assert "saved as [axis_twist_compensation carto] and applied" \
        in g.out[-1]


def test_interleaved_order_per_point(tmp_path):
    printer, atm = make(tmp_path)
    plist = printer.objects["probe_list"].probes
    order = []
    for name, pr in plist.items():
        orig = pr._read

        def wrapped(orig=orig, name=name):
            order.append(name)
            return orig()
        pr._read = wrapped
    atm.cmd_CALIBRATE(FakeGcmd())
    assert order == (["load_cell_probe"] * 3 + ["carto"] * 3) * 35
    assert all(p.begun == 0 for p in plist.values())


def nonadditive(nx, ny):
    # twist + Y pitch + a bump only a per-point grid can capture
    bx, by = nx, ny + 15.0
    bump = 0.03 * math.exp(-((bx - 140) ** 2 + (by - 110) ** 2) / 800.0)
    return 0.3 + twist(nx) + 0.0003 * (by - 110) + bump


def probe_at(printer, probe, bed_x, bed_y):
    th = printer.objects["toolhead"]
    xo, yo = probe.get_offsets()[:2]
    th.pos[0], th.pos[1] = bed_x - xo, bed_y - yo
    return probe.run_probe(None)[2] - probe.get_offsets()[2]


def test_three_readings_must_agree(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    lc = printer.objects["probe_list"].probes["load_cell_probe"]
    atm.cmd_CALIBRATE(FakeGcmd())
    clean = dict(atm.last["main"])
    calls = {"n": 0}
    orig = lc._read

    def flaky():
        calls["n"] += 1
        z = orig()
        if calls["n"] == 7:
            z[2] -= 0.06
        return z
    lc._read = flaky
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert any("main 1, correction 0" in m for m in g.out)
    assert max(abs(clean[k] - v) for k, v in atm.last["main"].items()) < 1e-12
    k = {"n": 0}

    def never():
        k["n"] += 1
        z = orig()
        z[2] += 0.05 * k["n"]
        return z
    lc._read = never
    with pytest.raises(CmdError, match="agreed within"):
        atm.cmd_CALIBRATE(FakeGcmd())


def test_unreachable_points_rejected(tmp_path):
    _, atm = make(tmp_path, carto_offsets=(0.0, 40.0, 2.0))
    with pytest.raises(CmdError, match="outside axis limits"):
        atm.cmd_CALIBRATE(FakeGcmd())


def test_missing_probe_lists_available(tmp_path):
    _, atm = make(tmp_path, CFG.replace("carto\n", "nope\n"))
    with pytest.raises(CmdError, match="load_cell_probe"):
        atm.cmd_CALIBRATE(FakeGcmd())


def test_partial_reprobe(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    with pytest.raises(CmdError, match="PROBES=both first"):
        atm.cmd_CALIBRATE(FakeGcmd({"PROBES": "main"}))
    atm.cmd_CALIBRATE(FakeGcmd())
    first = atm.report["comp_grid"]
    atm.cmd_CALIBRATE(FakeGcmd({"PROBES": "correction"}))
    assert atm.report["comp_grid"] == first


# ---------------------------------------------------------------------------
# Storage: Kalico's own [axis_twist_compensation <probe>] format

def test_map_saved_in_kalico_format_and_loaded_after_restart(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    cf = printer.objects["configfile"].sections
    sec = cf["axis_twist_compensation carto"]
    assert sec["version"] == "1" and sec["x_count"] == "7"
    assert sec["y_count"] == "5"
    assert sec["reference_probe"] == "load_cell_probe"
    assert sec["points"].startswith("\n  ") and sec["points"].count("\n") == 5
    assert sec["mesh_min"] == "20.000, 20.000"
    assert sec["probe_count"] == "7, 5" and sec["algorithm"] == "bicubic"
    assert not any(k.startswith("axis_twist_mesh") for k in cf)
    grid = act(atm)["grid"]

    # "SAVE_CONFIG" + restart: new plugin instance from the written config
    cfg2 = saved_cfg(printer)
    printer2, atm2 = make(tmp_path, cfg2, carto_err=nonadditive)
    assert "carto" in atm2.maps
    assert act(atm2) is None
    connect(printer2)
    assert act(atm2)["name"] == "axis_twist_compensation carto"
    for r1, r2 in zip(grid, act(atm2)["grid"]):
        for a, b in zip(r1, r2):
            assert abs(a - b) < 1e-6
    # and it corrects the probe after the restart
    carto = printer2.objects["probe_list"].probes["carto"]
    lc = printer2.objects["probe_list"].probes["load_cell_probe"]
    off = atm.report["mean_offset"]
    for x, y in ((20.0, 20.0), (110.0, 110.0), (200.0, 200.0)):
        c = probe_at(printer2, carto, x, y)
        m = probe_at(printer2, lc, x, y)
        assert abs(c - m - off) < 1e-5


def test_map_commands(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    connect(printer)
    with pytest.raises(CmdError, match="per probe"):
        atm.cmd_CALIBRATE(FakeGcmd({"PROFILE": "hot"}))
    atm.cmd_CALIBRATE(FakeGcmd())
    atm.cmd_CLEAR(FakeGcmd())
    assert act(atm) is None
    atm.cmd_PROFILE(FakeGcmd({"LOAD": "carto"}))
    assert act(atm) is not None
    atm.cmd_PROFILE(FakeGcmd({"REMOVE": "carto"}))
    assert "carto" not in atm.maps and act(atm) is None
    assert "axis_twist_compensation carto" not in printer.objects[
        "configfile"].sections
    with pytest.raises(CmdError, match="no saved map"):
        atm.cmd_PROFILE(FakeGcmd({"LOAD": "nope"}))
    with pytest.raises(CmdError, match="Saved maps: none"):
        atm.cmd_PROFILE(FakeGcmd())
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "saved maps: none" in g.out[0] and "compensation off" in g.out[0]


def test_bad_profile_in_config_rejected(tmp_path):
    bad = CFG + "\n[axis_twist_compensation carto]\nversion = 1\n" \
        "x_count = 3\ny_count = 2\nmin_x = 0\nmax_x = 10\nmin_y = 0\n" \
        "max_y = 10\nmesh_x_pps = 0\nmesh_y_pps = 0\nalgo = direct\n" \
        "tension = 0.2\nreference_probe = manual\n" \
        "points =\n  0.1, 0.2\n  0.3, 0.4\n"
    with pytest.raises(configparser.Error, match="don't match"):
        make(tmp_path, bad)


def test_max_compensation_refuses(tmp_path):
    printer, atm = make(tmp_path, CFG + "max_compensation: 0.01\n",
                        carto_err=nonadditive)
    connect(printer)
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert "NOT applied" in g.out[-1] and "max_compensation" in g.out[-1]
    assert act(atm) is None and "carto" in atm.maps


def test_table_formats(tmp_path):
    printer, atm = make(tmp_path)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    g = FakeGcmd()
    atm.cmd_TABLE(g)
    assert "[axis_twist_compensation carto], reference load_cell_probe" \
        in g.out[0]
    g = FakeGcmd({"FORMAT": "twist"})
    atm.cmd_TABLE(g)
    assert "z_compensations:" in g.out[0]


# ---------------------------------------------------------------------------
# Live application (standard Kalico probes)

def test_apply_corrects_only_correction_probe(tmp_path):
    cfg = CFG.replace("probe_count: 7, 5", "probe_count: 10, 10")
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    connect(printer)
    plist = printer.objects["probe_list"].probes
    lc, carto = plist["load_cell_probe"], plist["carto"]
    atm.cmd_CLEAR(FakeGcmd())
    lc_before = probe_at(printer, lc, 77.0, 133.0)
    atm.cmd_CALIBRATE(FakeGcmd())
    off = atm.report["mean_offset"]
    xs, ys = atm._grid_axis(0), atm._grid_axis(1)
    for x in xs:
        for y in ys:
            c = probe_at(printer, carto, x, y)
            m = probe_at(printer, lc, x, y)
            assert abs(c - m - off) < 1e-9
    worst = 0.0
    for bx in (33.0, 77.0, 131.0, 187.0):
        for by in (27.0, 99.0, 151.0, 197.0):
            c = probe_at(printer, carto, bx, by)
            m = probe_at(printer, lc, bx, by)
            worst = max(worst, abs(c - m - off))
    assert worst < 0.005
    assert probe_at(printer, lc, 77.0, 133.0) == lc_before
    atm.cmd_CLEAR(FakeGcmd())
    assert abs(probe_at(printer, carto, 20.0, 20.0)
               - probe_at(printer, lc, 20.0, 20.0) - off) > 0.01


def test_calibrate_while_applied_is_raw(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    first = atm.report["comp_grid"]
    assert act(atm) is not None
    atm.cmd_CALIBRATE(FakeGcmd())
    assert atm.report["comp_grid"] == first


def test_map_replaces_kalico_value_for_correction_probe(tmp_path):
    printer, atm = make(tmp_path, CFG + ATC_SECTION, carto_err=nonadditive)

    class ATC:
        def get_z_compensation_value(self, pos):
            return 0.001 * pos[0]
    printer.objects["axis_twist_compensation"] = ATC()
    connect(printer)
    plist = printer.objects["probe_list"].probes
    lc, carto = plist["load_cell_probe"], plist["carto"]
    atm.cmd_CALIBRATE(FakeGcmd())
    off = atm.report["mean_offset"]
    # other probes (the reference) still get Kalico's value
    m = probe_at(printer, lc, 110.0, 110.0)
    assert abs(m - (bed(110.0, 110.0) + 0.001 * 110.0)) < 1e-12
    # the correction probe gets our map instead of it, not on top of it
    c = probe_at(printer, carto, 80.0, 65.0)
    m = probe_at(printer, lc, 80.0, 65.0) - 0.001 * 80.0
    assert abs(c - m - off) < 1e-9


# ---------------------------------------------------------------------------
# Refinements, checked against the user's real 11x11 run

REAL_11 = """
210.0 +0.0951 +0.0593 +0.0425 +0.0524 +0.0441 +0.0427 +0.0306 +0.0280 +0.0396 -0.0020 +0.0401
193.5 +0.0674 +0.0579 +0.0490 +0.0336 +0.0265 -0.0542 +0.0172 +0.0075 -0.0111 -0.0160 +0.0103
177.0 +0.0590 +0.0311 +0.0481 +0.0279 +0.0383 +0.0079 +0.0158 +0.0163 +0.0067 +0.0057 +0.0070
160.5 +0.0570 +0.0668 +0.0319 +0.0369 +0.0068 -0.0089 +0.0153 -0.0087 +0.0091 -0.0231 +0.0395
144.0 +0.0436 +0.0333 +0.0358 +0.0409 +0.0081 +0.0105 +0.0383 -0.0027 -0.0082 -0.0359 +0.0180
127.5 +0.0295 +0.0251 +0.0258 +0.0269 +0.0073 -0.0290 +0.0045 -0.0245 -0.0172 -0.0302 +0.0186
111.0 +0.0372 +0.0396 +0.0010 +0.0074 +0.0128 -0.0399 -0.0178 -0.0338 -0.0296 -0.0360 +0.0046
94.5 +0.0127 -0.0195 +0.0075 -0.0289 -0.0286 -0.0489 -0.0698 -0.0479 -0.0297 -0.0367 +0.0332
78.0 +0.0158 -0.0060 -0.0176 -0.0229 -0.0275 -0.0433 -0.0532 -0.0307 -0.0403 -0.0463 +0.0040
61.5 +0.0084 -0.0093 -0.0392 -0.0335 -0.0365 -0.0528 -0.0483 -0.0315 -0.0493 -0.0607 -0.0031
45.0 -0.0030 -0.0276 -0.0350 -0.0423 -0.0329 -0.0524 -0.0562 -0.0473 -0.0605 -0.0630 -0.0101
"""
CFG_11 = CFG.replace("mesh_min: 20, 20", "mesh_min: 20, 45") \
            .replace("mesh_max: 200, 200", "mesh_max: 210, 210") \
            .replace("probe_count: 7, 5", "probe_count: 11, 11")


def load_real(atm):
    xs, ys = atm._grid_axis(0), atm._grid_axis(1)
    rows = {}
    for line in REAL_11.strip().splitlines():
        vals = [float(v) for v in line.split()]
        rows[vals[0]] = vals[1:]
    main, corr = {}, {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            main[(i, j)] = 0.0
            corr[(i, j)] = -0.4779 - rows[y][i]
    atm.last = {"xs": xs, "ys": ys, "main": main, "correction": corr}
    atm._analyze()


def test_flags_real_edge_column_and_bad_point(tmp_path):
    _, atm = make(tmp_path, CFG_11)
    load_real(atm)
    rep = atm.report
    edges = [(w["axis"], w["position"]) for w in rep["edge_warnings"]]
    assert ("X", 210.0) in edges and ("X", 20.0) not in edges
    sus = [(p["x"], p["y"]) for p in rep["suspect_points"]]
    assert sus == [(115.0, 193.5)]
    g = FakeGcmd()
    atm.cmd_ANALYZE(g)
    assert "Edge warning: X=210.0" in g.out[0] and "mesh_max" in g.out[0]
    assert "Suspect points" in g.out[0]


def test_no_false_warnings_on_clean_data(tmp_path):
    cfg = CFG.replace("probe_count: 7, 5", "probe_count: 9, 9")
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    atm.cmd_CALIBRATE(FakeGcmd())
    assert atm.report["edge_warnings"] == []
    assert atm.report["suspect_points"] == []


# ---------------------------------------------------------------------------
# Third-party Cartographer plugin on Kalico

import types  # noqa: E402

_adapter_mod = types.ModuleType("cartographer.adapters.kalico.atc")
exec(
    "def twist_value(compensation, x, y):\n"
    "    return compensation.get_z_compensation_value([x, y, 0])\n",
    _adapter_mod.__dict__)


class CartoPluginProbe:
    """Mimics cartographer.adapters.kalico.probe.KalicoCartographerProbe."""

    def __init__(self, printer, offsets, err_fn):
        self.printer, self.offsets, self.err_fn = printer, offsets, err_fn
        self.probe_name = "cartographer"

    def get_offsets(self):
        return self.offsets

    def multi_probe_begin(self):
        pass

    def multi_probe_end(self):
        pass

    def run_probe(self, gcmd, *a, **k):
        th = self.printer.objects["toolhead"]
        nx, ny = th.pos[0], th.pos[1]
        bx, by = nx + self.offsets[0], ny + self.offsets[1]
        z = bed(bx, by) + self.offsets[2] + self.err_fn(nx, ny)
        if getattr(self.printer, "carto_has_atc_section", False):
            z += _adapter_mod.twist_value(
                self.printer.objects["axis_twist_compensation"], nx, ny)
        return [nx, ny, z]


CartoPluginProbe.__module__ = "cartographer.adapters.kalico.probe"


class _CartoAdapter:
    """Like cartographer.adapters.kalico.axis_twist_compensation."""

    def __init__(self, printer):
        self.printer = printer

    def get_z_compensation_value(self, *, x, y):
        return _adapter_mod.twist_value(
            self.printer.objects["axis_twist_compensation"], x, y)


class _Reg:
    def __init__(self, name, macro):
        self.name, self.macro = name, macro


# a real printer.cfg declares the section; the test object stands in for it
ATC_SECTION = "\n[axis_twist_compensation]\n"
CARTO_CFG = CFG.replace("correction_probe: carto",
                        "correction_probe: cartographer")


def make_carto(tmp_path, with_section=True, cfg=CARTO_CFG):
    if with_section and ATC_SECTION not in cfg:
        cfg = cfg + ATC_SECTION
    printer, atm = make(tmp_path, cfg)
    plist = printer.objects["probe_list"].probes
    del plist["carto"]
    carto = CartoPluginProbe(printer, (0.0, 15.0, 2.0), nonadditive)
    plist["cartographer"] = carto

    class BedMeshMacro:
        axis_twist_compensation = _CartoAdapter(printer) if with_section \
            else None

    class PrinterCartographer:
        macros = [_Reg("BED_MESH_CALIBRATE", BedMeshMacro())]
    printer.objects["cartographer"] = PrinterCartographer()
    if with_section:
        class RealATC:   # Kalico's object with no compensation values
            def get_z_compensation_value(self, pos):
                return 0.0
        printer.objects["axis_twist_compensation"] = RealATC()
        printer.carto_has_atc_section = True
    connect(printer)
    return printer, atm, carto, BedMeshMacro.axis_twist_compensation


def test_carto_plugin_needs_twist_section(tmp_path):
    printer, atm, carto, _ = make_carto(tmp_path, with_section=False)
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert "NOT applied" in g.out[-1]
    assert "[axis_twist_compensation]" in g.out[-1]
    g = FakeGcmd()
    atm.cmd_CHECK(g)
    assert "BED_MESH_CALIBRATE twist hook: MISSING" in g.out[0]


def test_carto_plugin_gets_compensation_load_cell_does_not(tmp_path):
    printer, atm, carto, _ = make_carto(tmp_path)
    lc = printer.objects["probe_list"].probes["load_cell_probe"]
    atm.cmd_CLEAR(FakeGcmd())
    lc_before = probe_at(printer, lc, 77.0, 133.0)
    atm.cmd_CALIBRATE(FakeGcmd())
    off = atm.report["mean_offset"]
    raw = atm.report["comp_grid"]
    for y in atm._grid_axis(1):
        for x in atm._grid_axis(0):
            c = probe_at(printer, carto, x, y)
            m = probe_at(printer, lc, x, y)
            assert abs(c - m - off) < 1e-9
    assert probe_at(printer, lc, 77.0, 133.0) == lc_before
    atm.cmd_CALIBRATE(FakeGcmd())
    assert atm.report["comp_grid"] == raw


def test_carto_scan_mesh_in_forked_worker_gets_compensation(tmp_path):
    import multiprocessing
    printer, atm, carto, adapter = make_carto(tmp_path)
    atm.cmd_CALIBRATE(FakeGcmd())
    pts = [(31.0, 47.0), (150.0, 120.0), (199.0, 199.0)]

    def fn():
        return [adapter.get_z_compensation_value(x=x, y=y - 15.0)
                for x, y in pts]

    parent, child = multiprocessing.Pipe()

    def worker(conn):
        conn.send(fn())
        conn.close()
    proc = multiprocessing.get_context("fork").Process(
        target=worker, args=(child,), daemon=True)
    proc.start()
    got = parent.recv()
    proc.join()
    for (x, y), g in zip(pts, got):
        assert abs(g - atm.compensation_at(x, y)) < 1e-12
        assert abs(g) > 0


def test_kalico_driven_rapid_scan_gets_compensation(tmp_path):
    printer, atm, carto, adapter = make_carto(tmp_path)
    atm.cmd_CALIBRATE(FakeGcmd())
    atc = printer.objects["axis_twist_compensation"]
    v = atc.get_z_compensation_value((100.0, 100.0 - 15.0, 3.0))
    assert abs(v - atm.compensation_at(100.0, 100.0)) < 1e-12
    lc = printer.objects["probe_list"].probes["load_cell_probe"]
    lc.multi_probe_begin()
    try:
        assert atc.get_z_compensation_value((100.0, 85.0, 3.0)) == 0.0
    finally:
        lc.multi_probe_end()


def test_check_command(tmp_path):
    printer, atm, carto, adapter = make_carto(tmp_path)
    g = FakeGcmd()
    atm.cmd_CHECK(g)
    assert "NOT working" in g.out[0] and "no active map" in g.out[0]
    atm.cmd_CALIBRATE(FakeGcmd())
    g = FakeGcmd()
    atm.cmd_CHECK(g)
    assert "Result: compensation reaches the correction probe" in g.out[0]
    assert g.out[0].count("via Cartographer") == 5
    assert "MISMATCH" not in g.out[0]


def test_carto_profile_restart_round_trip(tmp_path):
    printer, atm, carto, adapter = make_carto(tmp_path)
    atm.cmd_CALIBRATE(FakeGcmd())
    want = atm.compensation_at(123.0, 77.0)
    cfg2 = saved_cfg(printer, CARTO_CFG)
    printer2, atm2, carto2, adapter2 = make_carto(tmp_path, cfg=cfg2)
    assert act(atm2)["name"] == "axis_twist_compensation cartographer"
    got = adapter2.get_z_compensation_value(x=123.0, y=77.0 - 15.0)
    assert abs(got - want) < 1e-5


# ---------------------------------------------------------------------------
# single_probe: Cartographer touch is the reference, scan is corrected

class _Pos:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class SimTouchMode:
    def __init__(self, printer):
        self.printer = printer
        self.offset = _Pos(0.0, 0.0, 0.0)
        self.calls = 0
        self.model_samples = 3

        class B:   # bed_mesh 10..220 shrunk by the +15 Y coil offset
            min_x, max_x, min_y, max_y = 10.0, 220.0, 10.0, 205.0
        self.boundaries = B()

    def get_model(self):
        mode = self

        class M:
            samples = mode.model_samples
            sample_range = 0.010
        return M()

    def perform_probe(self):
        self.calls += 1
        th = self.printer.objects["toolhead"]
        return bed(th.pos[0], th.pos[1])


class SimScanMode:
    def __init__(self, printer, err_fn):
        self.printer, self.err_fn = printer, err_fn
        self.offset = _Pos(0.0, 15.0, 2.5)
        self.calls = 0

    def perform_probe(self):
        self.calls += 1
        th = self.printer.objects["toolhead"]
        nx, ny = th.pos[0], th.pos[1]
        z = bed(nx, ny + 15.0) + self.err_fn(nx, ny)
        if getattr(self.printer, "carto_has_atc_section", False):
            z += _adapter_mod.twist_value(
                self.printer.objects["axis_twist_compensation"], nx, ny)
        return z


SimScanMode.__module__ = "cartographer.probe.scan_mode"


class KalicoCartoProbeSim:
    def __init__(self, scan):
        self.scan = scan
        self.probe_name = "cartographer"

    def get_offsets(self):
        o = self.scan.offset
        return (o.x, o.y, o.z)

    def multi_probe_begin(self):
        pass

    def multi_probe_end(self):
        pass

    def run_probe(self, gcmd, *a, **k):
        th = self.scan.printer.objects["toolhead"]
        return [th.pos[0], th.pos[1], self.scan.perform_probe()]


KalicoCartoProbeSim.__module__ = "cartographer.adapters.kalico.probe"


def make_single(tmp_path):
    cfg = CFG.replace("main_probe: load_cell_probe\n", "") \
             .replace("correction_probe: carto",
                      "correction_probe: cartographer")
    cfg += "single_probe: True\n" + ATC_SECTION
    printer, atm = make(tmp_path, cfg)
    plist = printer.objects["probe_list"].probes
    plist.clear()
    printer.objects["cartographer"] = pc = type("PC", (), {})()
    pc.touch_mode = SimTouchMode(printer)
    pc.scan_mode = SimScanMode(printer, nonadditive)
    plist["cartographer"] = KalicoCartoProbeSim(pc.scan_mode)

    class RealATC:
        def get_z_compensation_value(self, pos):
            return 0.0
    printer.objects["axis_twist_compensation"] = RealATC()
    printer.carto_has_atc_section = True
    connect(printer)
    return printer, atm, pc


def test_single_probe_touch_vs_scan_and_apply(tmp_path):
    printer, atm, pc = make_single(tmp_path)
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "reference: cartographer touch (found)" in g.out[0]
    assert "(grid fits)" in g.out[0]
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    n = 35
    assert pc.touch_mode.calls == n and pc.scan_mode.calls == 3 * n
    assert any("3 touches within 0.010 mm" in m for m in g.out)
    assert atm.maps["cartographer"]["reference"] == "touch"
    off = atm.report["mean_offset"]
    th = printer.objects["toolhead"]
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            th.pos[0], th.pos[1] = x, y - 15.0
            scan = pc.scan_mode.perform_probe()
            th.pos[0], th.pos[1] = x, y
            assert abs(scan - pc.touch_mode.perform_probe() - off) < 1e-9


def test_single_probe_touch_area_and_samples(tmp_path):
    printer, atm, pc = make_single(tmp_path)
    pc.touch_mode.model_samples = 2
    with pytest.raises(CmdError, match="samples: 3"):
        atm.cmd_CALIBRATE(FakeGcmd())
    pc.touch_mode.model_samples = 3
    pc.touch_mode.boundaries.max_y = 150.0
    pc.touch_mode.boundaries.min_x = 30.0
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "DOES NOT FIT" in g.out[0]
    th = printer.objects["toolhead"]
    before = len(th.moves)
    with pytest.raises(CmdError) as exc:
        atm.cmd_CALIBRATE(FakeGcmd())
    assert "mesh_min: 30.0, 20.0 and mesh_max: 200.0, 150.0" in str(exc.value)
    assert len(th.moves) == before


# ---------------------------------------------------------------------------
# Beacon plugin. The simulated BeaconProbe lives in a module named like the
# real one ("extras.beacon") and mirrors beacon.py: the twist lookup set up
# in _handle_connect, the value added in _probe_contact, and subtracted from
# every streamed proximity distance.

_beacon_mod = types.ModuleType("extras.beacon")
exec('''
class BeaconProbe:
    def __init__(self, printer, x_offset, y_offset, err_fn, default="proximity"):
        self.printer = printer
        self.x_offset, self.y_offset = x_offset, y_offset
        self.trigger_distance = 2.0
        self.default_probe_method = default
        self.mod_axis_twist_comp = None
        self.get_z_compensation_value = lambda pos: 0.0
        self.err_fn = err_fn
        self.calls = []
        class Contact:
            max_hotend_temp = 180.0
        self.mcu_contact_probe = Contact()
        self.toolhead = printer.objects["toolhead"]

    def _handle_connect(self):
        self.mod_axis_twist_comp = self.printer.lookup_object(
            "axis_twist_compensation", None)
        if self.mod_axis_twist_comp:
            if hasattr(self.mod_axis_twist_comp, "get_z_compensation_value"):
                self.get_z_compensation_value = lambda pos: (
                    self.mod_axis_twist_comp.get_z_compensation_value(pos))

    def get_offsets(self, _gcmd=None):
        return (0, 0, 0) if getattr(self, "_current", "") == "contact" \\
            else (self.x_offset, self.y_offset, self.trigger_distance)

    def run_probe(self, gcmd):
        method = gcmd.get("PROBE_METHOD", self.default_probe_method).lower()
        self._current = method
        self.calls.append((method, dict(gcmd.get_command_parameters())))
        if method == "proximity":
            return self._probe()
        n = gcmd.get_int("SAMPLES", 3)
        samples = [self._probe_contact(3.0)[2] for _ in range(n)]
        pos = self.toolhead.get_position()
        return pos[:2] + [sum(samples) / len(samples)]

    def _probe_contact(self, speed):
        start_pos = self.toolhead.get_position()
        pos = start_pos[:]
        pos[2] = -2
        epos = [start_pos[0], start_pos[1], self.bed(start_pos[0], start_pos[1])]
        epos[2] += self.get_z_compensation_value(pos)
        return epos[:3]

    def stream_sample(self, nozzle_pos):
        # what _handle_beacon_data does for each sample
        nx, ny, nz = nozzle_pos
        bx, by = nx + self.x_offset, ny + self.y_offset
        dist = nz - (self.bed(bx, by) + self.err_fn(nx, ny))
        dist -= self.get_z_compensation_value(nozzle_pos)
        return dist

    def _probe(self):
        pos = self.toolhead.get_position()
        z = self.trigger_distance
        dist = self.stream_sample([pos[0], pos[1], z])
        return [pos[0], pos[1], z + self.trigger_distance - dist]
''', _beacon_mod.__dict__)


def make_beacon(tmp_path, single=False, default="proximity", beacon_first=True,
                started=True):
    cfg = CFG.replace("correction_probe: carto", "correction_probe: beacon")
    if single:
        cfg = cfg.replace("main_probe: load_cell_probe\n", "") + \
            "single_probe: True\n"
    printer, atm = make(tmp_path, cfg)
    plist = printer.objects["probe_list"].probes
    del plist["carto"]
    if single:
        plist.clear()
    beacon = _beacon_mod.BeaconProbe(printer, 0.0, 20.0, nonadditive, default)
    beacon.bed = bed
    printer.objects["beacon"] = beacon
    if started:
        hs = printer.handlers.setdefault("klippy:connect", [])
        if beacon_first:
            hs.insert(0, beacon._handle_connect)
        else:
            hs.append(beacon._handle_connect)
    connect(printer)
    return printer, atm, beacon


def beacon_scan_height(printer, beacon, bx, by):
    th = printer.objects["toolhead"]
    th.pos[0], th.pos[1] = bx - beacon.x_offset, by - beacon.y_offset
    return beacon._probe()[2] - beacon.trigger_distance


def beacon_contact_height(printer, beacon, bx, by):
    th = printer.objects["toolhead"]
    th.pos[0], th.pos[1] = bx, by
    return beacon._probe_contact(3.0)[2]


@pytest.mark.parametrize("beacon_first", [True, False])
def test_beacon_with_load_cell(tmp_path, beacon_first):
    printer, atm, beacon = make_beacon(tmp_path, default="contact",
                                       beacon_first=beacon_first)
    assert beacon.mod_axis_twist_comp is atm   # found our hook either way
    lc = printer.objects["probe_list"].probes["load_cell_probe"]
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert "saved as [axis_twist_mesh beacon] and applied" in g.out[-1]
    # forced to proximity even though Beacon defaults to contact
    assert {m for m, _ in beacon.calls} == {"proximity"}
    off = atm.report["mean_offset"]
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            c = beacon_scan_height(printer, beacon, x, y)
            m = probe_at(printer, lc, x, y)
            assert abs(c - m - off) < 1e-9
            # contact readings are nozzle readings: never corrected
            assert beacon_contact_height(printer, beacon, x, y) == bed(x, y)
    # a bed-mesh scan sample between grid points is corrected too
    nozzle = [93.0, 61.0 - 20.0, 2.0]          # bed point (93, 61)
    d_on = beacon.stream_sample(nozzle)
    atm.cmd_CLEAR(FakeGcmd())
    d_off = beacon.stream_sample(nozzle)
    atm.cmd_PROFILE(FakeGcmd({"LOAD": "beacon"}))
    # Beacon subtracts it from the distance, so the height rises by comp
    assert abs((d_off - d_on) - atm.compensation_at(93.0, 61.0)) < 1e-12
    assert abs(atm.compensation_at(93.0, 61.0)) > 0.001
    g = FakeGcmd()
    atm.cmd_CHECK(g)
    assert "Beacon twist hook: present" in g.out[0]
    assert g.out[0].count("via Beacon") == 5 and "MISMATCH" not in g.out[0]


def test_beacon_reading_skipped_while_load_cell_probes(tmp_path):
    printer, atm, beacon = make_beacon(tmp_path)
    atm.cmd_CALIBRATE(FakeGcmd())
    lc = printer.objects["probe_list"].probes["load_cell_probe"]
    lc.multi_probe_begin()
    try:
        assert beacon.get_z_compensation_value([100.0, 80.0, 2.0]) == 0.0
    finally:
        lc.multi_probe_end()
    assert beacon.get_z_compensation_value([100.0, 80.0, 2.0]) != 0.0


def test_beacon_single_probe_contact_vs_proximity(tmp_path):
    printer, atm, beacon = make_beacon(tmp_path, single=True)
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "reference: beacon contact (found)" in g.out[0]
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    n = 35
    contact = [p for m, p in beacon.calls if m == "contact"]
    prox = [m for m, _ in beacon.calls if m == "proximity"]
    assert len(contact) == n and len(prox) == 3 * n
    assert contact[0]["SAMPLES"] == "3"
    assert float(contact[0]["SAMPLES_TOLERANCE"]) == 0.010
    assert any("3 touches within 0.010 mm" in m for m in g.out)
    assert atm.maps["beacon"]["reference"] == "touch"
    off = atm.report["mean_offset"]
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            c = beacon_scan_height(printer, beacon, x, y)
            t = beacon_contact_height(printer, beacon, x, y)
            assert abs(c - t - off) < 1e-9
            assert t == bed(x, y)


def test_beacon_contact_hotend_check(tmp_path):
    printer, atm, beacon = make_beacon(tmp_path, single=True)
    th = printer.objects["toolhead"]
    th.hotend_temp = 200.0
    before = len(th.moves)
    with pytest.raises(CmdError, match="below 180C"):
        atm.cmd_CALIBRATE(FakeGcmd())
    assert len(th.moves) == before


def test_beacon_not_started_reports_missing_hook(tmp_path):
    printer, atm, beacon = make_beacon(tmp_path, started=False)
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert "NOT applied" in g.out[-1] and "RESTART" in g.out[-1]
    g = FakeGcmd()
    atm.cmd_CHECK(g)
    assert "Beacon twist hook: MISSING" in g.out[0]


def test_beacon_profile_restart_round_trip(tmp_path):
    printer, atm, beacon = make_beacon(tmp_path)
    atm.cmd_CALIBRATE(FakeGcmd())
    want = atm.compensation_at(123.0, 77.0)
    cfg2 = saved_cfg(printer, CFG.replace("correction_probe: carto",
                                          "correction_probe: beacon"))
    printer2, atm2 = make(tmp_path, cfg2)
    del printer2.objects["probe_list"].probes["carto"]
    beacon2 = _beacon_mod.BeaconProbe(printer2, 0.0, 20.0, nonadditive)
    beacon2.bed = bed
    printer2.objects["beacon"] = beacon2
    printer2.handlers["klippy:connect"].insert(0, beacon2._handle_connect)
    connect(printer2)
    assert act(atm2)["name"] == "axis_twist_mesh beacon"
    got = beacon2.get_z_compensation_value([123.0, 77.0 - 20.0, 2.0])
    assert abs(got - want) < 1e-5


# ---------------------------------------------------------------------------
# Kalico's mesh-based axis_twist_compensation (commit ac18842): probe code
# calls get_z_compensation_value(pos, probe). Runs against the real class
# when KALICO points at a checkout that has it.

import inspect  # noqa: E402

from klippy.extras import axis_twist_compensation as katc  # noqa: E402

NEW_ATC = "probe" in inspect.signature(
    katc.AxisTwistCompensation.get_z_compensation_value).parameters


def test_hook_accepts_probe_argument(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    plist = printer.objects["probe_list"].probes
    lc, carto = plist["load_cell_probe"], plist["carto"]
    want = atm.compensation_at(100.0, 100.0)
    pos = [100.0, 100.0 - 15.0, 2.0]
    # Probe identified by Kalico: only the correction probe is corrected,
    # with no reliance on multi_probe_pending.
    assert abs(atm.get_z_compensation_value(pos, carto) - want) < 1e-12
    assert atm.get_z_compensation_value(pos, lc) == 0.0
    # Original function receives the probe when one is given
    seen = []
    atm._orig_twist_fn = lambda p, probe=None: seen.append(probe) or 0.0
    atm.get_z_compensation_value(pos, lc)
    atm.get_z_compensation_value(pos)
    assert seen == [lc, None]


@pytest.mark.skipif(not NEW_ATC, reason="Kalico without mesh-based twist comp")
def test_with_real_mesh_based_axis_twist_compensation(tmp_path):
    cfg = CFG + "\n[axis_twist_compensation]\n"
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    fc = configparser.RawConfigParser()
    fc.read_string(cfg)
    manager = katc.load_config(configfile.ConfigWrapper(
        printer, fc, {}, "axis_twist_compensation"))
    printer.objects["axis_twist_compensation"] = manager
    connect(printer)
    plist = printer.objects["probe_list"].probes
    lc, carto = plist["load_cell_probe"], plist["carto"]
    # patched in place, real manager still answers for its own maps
    assert manager.get_z_compensation_value.__self__ is atm
    atm.cmd_CALIBRATE(FakeGcmd())
    off = atm.report["mean_offset"]
    # Readings the way the new PrinterProbe._probe makes them
    for x, y in ((20.0, 20.0), (110.0, 110.0), (200.0, 200.0)):
        th = printer.objects["toolhead"]
        th.pos[0], th.pos[1] = x, y - 15.0
        raw = bed(x, y) + nonadditive(x, y - 15.0)
        c = raw + manager.get_z_compensation_value(list(th.pos), carto)
        th.pos[0], th.pos[1] = x, y
        m = bed(x, y) + manager.get_z_compensation_value(list(th.pos), lc)
        assert abs(c - m - off) < 1e-9


# ---------------------------------------------------------------------------
# Improvements taken from Kalico's mesh-based axis_twist_compensation

def test_bed_mesh_interpolation_auto_and_stored(tmp_path):
    printer, atm = make(tmp_path, carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())          # 7x5 grid -> bicubic
    a = act(atm)
    assert a["mesh"].get_mesh_params()["algo"] == "bicubic"
    sec = printer.objects["configfile"].sections[
        "axis_twist_compensation carto"]
    assert sec["algo"] == "bicubic" and sec["mesh_x_pps"] == "2"
    # exact at measured points
    for j, y in enumerate(a["ys"]):
        for i, x in enumerate(a["xs"]):
            assert abs(atm.compensation_at(x, y) - a["grid"][j][i]) < 1e-9
    # clamped outside the grid, like bed_mesh
    assert abs(atm.compensation_at(-50.0, -50.0) - a["grid"][0][0]) < 1e-9
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "bicubic" in g.out[0]
    st = atm.get_status(0)
    assert st["mesh_params"]["algo"] == "bicubic"
    assert len(st["probed_matrix"]) == 5 and len(st["probed_matrix"][0]) == 7
    assert len(st["mesh_matrix"]) == 5 + 4 * 2


def test_bicubic_closer_than_bilinear_between_points(tmp_path):
    # A smooth error sampled on a coarse grid: bicubic should follow it
    # between grid points better than straight lines do.
    def smooth(nx, ny):
        return 0.3 + 0.03 * math.sin(nx / 25.0) * math.cos((ny + 15) / 30.0)
    cfg = CFG.replace("probe_count: 7, 5", "probe_count: 6, 6")
    printer, atm = make(tmp_path, cfg, carto_err=smooth)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    off = atm.report["mean_offset"]
    mean_err = off - 0.3

    def worst():
        w = 0.0
        for bx in range(25, 200, 9):
            for by in range(25, 200, 11):
                truth = -(smooth(bx, by - 15) - 0.3 - mean_err)
                w = max(w, abs(atm.compensation_at(bx, by) - truth))
        return w
    bicubic = worst()
    atm.maps["carto"]["algo"] = "direct"
    atm.cmd_PROFILE(FakeGcmd({"LOAD": "carto"}))
    assert act(atm)["mesh"].get_mesh_params()["algo"] == "direct"
    assert bicubic < worst()


def test_algorithm_rules(tmp_path):
    rule = axis_twist_mesh._resolve_algorithm
    err = configparser.Error
    assert rule("auto", 3, 3, (2, 2), err) == "lagrange"
    assert rule("auto", 11, 11, (2, 2), err) == "bicubic"
    assert rule("auto", 3, 11, (2, 2), err) == "direct"
    assert rule("auto", 2, 5, (2, 2), err) == "direct"
    # direct must build with no sub-points (3x11 picks it automatically)
    cfg = CFG.replace("probe_count: 7, 5", "probe_count: 3, 11")
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    assert act(atm)["mesh"].get_mesh_params()["mesh_x_pps"] == 0
    assert rule("bicubic", 3, 3, (2, 2), err) == "lagrange"
    assert rule("bicubic", 5, 5, (0, 0), err) == "direct"
    with pytest.raises(err, match="lagrange"):
        rule("lagrange", 7, 7, (2, 2), err)
    with pytest.raises(configparser.Error, match="lagrange"):
        make(tmp_path, CFG.replace("probe_count: 7, 5", "probe_count: 9, 9")
             + "algorithm: lagrange\n")
    with pytest.raises(configparser.Error, match="1 mm apart"):
        make(tmp_path, CFG.replace("probe_count: 7, 5",
                                   "probe_count: 400, 5"))


def test_old_profile_without_interpolation_keys_loads(tmp_path):
    old = CFG + ("\n[axis_twist_mesh default]\nversion = 1\nx_count = 3\n"
                 "y_count = 3\nmin_x = 20\nmax_x = 200\nmin_y = 20\n"
                 "max_y = 200\noffset = -0.5\nreference = load_cell_probe\n"
                 "points =\n  0.01, 0.02, 0.03\n  0.0, 0.0, 0.0\n"
                 "  -0.01, -0.02, -0.03\n")
    printer, atm = make(tmp_path, old)
    connect(printer)
    assert act(atm)["mesh"].get_mesh_params()["algo"] == "lagrange"
    assert abs(atm.compensation_at(200.0, 20.0) - 0.03) < 1e-9


def test_single_probe_option_still_works(tmp_path):
    _, atm = make(tmp_path, CFG.replace("main_probe: load_cell_probe\n", "")
                  + "single_probe: True\n")
    assert atm.reference == "touch"
    with pytest.raises(configparser.Error, match="conflicts"):
        make(tmp_path, CFG + "single_probe: True\nreference: manual\n")


# --- manual (paper) reference

PAPER = 0.1


def make_manual(tmp_path, cfg=None):
    cfg = cfg or CFG.replace("main_probe: load_cell_probe\n", "") + \
        "reference: manual\n"
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    connect(printer)
    helpers = []

    def fake_helper(gcmd, callback):
        th = printer.objects["toolhead"]
        helpers.append((tuple(th.pos[:2]), callback))
    atm._new_manual_probe = fake_helper
    return printer, atm, helpers


def accept(printer, helpers):
    """The user lowers the nozzle onto the paper and sends ACCEPT."""
    (x, y), callback = helpers[-1]
    printer.objects["toolhead"].pos[2] = bed(x, y) + PAPER   # TESTZ down
    callback([x, y, bed(x, y) + PAPER])


def test_manual_reference_full_run(tmp_path):
    printer, atm, helpers = make_manual(tmp_path)
    carto = printer.objects["probe_list"].probes["carto"]
    order = []
    orig = carto._read

    def logged():
        order.append("carto")
        return orig()
    carto._read = logged
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert atm.calibrating and len(helpers) == 1
    assert helpers[0][0] == (20.0, 20.0)          # nozzle over the point
    assert order == ["carto"] * 3                 # probe measured it first
    with pytest.raises(CmdError, match="in progress"):
        atm.cmd_CALIBRATE(FakeGcmd())
    s = FakeGcmd()
    atm.cmd_STATUS(s)
    assert "point 1 of 35" in s.out[0]
    for _ in range(35):
        accept(printer, helpers)
    assert not atm.calibrating and atm._manual is None
    assert printer.objects["toolhead"].pos[2] == 5.0     # lifted off paper
    assert len(order) == 3 * 35
    assert "saved as [axis_twist_compensation carto] and applied" \
        in g.out[-1]
    assert atm.maps["carto"]["reference"] == "manual"
    # same result as a perfect load cell: paper thickness cancels out
    off = atm.report["mean_offset"]
    lc = printer.objects["probe_list"].probes["load_cell_probe"]
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            c = probe_at(printer, carto, x, y)
            m = probe_at(printer, lc, x, y)
            assert abs(c - m - (off + PAPER)) < 1e-9


def test_manual_abort_keeps_previous_profile(tmp_path):
    printer, atm, helpers = make_manual(tmp_path)
    for _ in range(1):
        atm.cmd_CALIBRATE(FakeGcmd())
        for _ in range(35):
            accept(printer, helpers)
    before = [r[:] for r in act(atm)["grid"]]
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    accept(printer, helpers)
    helpers[-1][1](None)                          # ABORT
    assert not atm.calibrating and atm._manual is None
    assert "aborted" in g.out[-1]
    assert act(atm)["grid"] == before


def test_manual_probe_error_resets_state(tmp_path):
    printer, atm, helpers = make_manual(tmp_path)
    carto = printer.objects["probe_list"].probes["carto"]
    atm.cmd_CALIBRATE(FakeGcmd())
    k = {"n": 0}
    orig = carto._read

    def never():
        k["n"] += 1
        z = orig()
        z[2] += 0.05 * k["n"]
        return z
    carto._read = never
    with pytest.raises(CmdError, match="agreed within"):
        accept(printer, helpers)
    assert not atm.calibrating and atm._manual is None


def test_reference_override_on_command(tmp_path):
    printer, atm, helpers = make_manual(tmp_path, CFG)   # config: load_cell
    atm.cmd_CALIBRATE(FakeGcmd({"REFERENCE": "manual"}))
    assert atm._manual is not None
    helpers[-1][1](None)
    with pytest.raises(CmdError, match="always measures both"):
        atm.cmd_CALIBRATE(FakeGcmd({"REFERENCE": "manual",
                                    "PROBES": "main"}))


# --- warning when Kalico's own compensation also corrects the probe

def test_legacy_lines_noted_and_replaced_for_correction_probe(tmp_path):
    printer, atm = make(tmp_path, CFG + ATC_SECTION, carto_err=nonadditive)

    class Legacy:
        z_compensations = [0.01, 0.0, -0.01]
        zy_compensations = []

        def get_z_compensation_value(self, pos):
            return 0.0
    printer.objects["axis_twist_compensation"] = Legacy()
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "note: z_compensations" in g.out[0]
    assert "WARNING" not in g.out[0]



@pytest.mark.skipif(not NEW_ATC, reason="Kalico without mesh-based twist comp")
def test_beacon_status_with_mesh_based_kalico(tmp_path):
    # BeaconProbe has no probe_name; Kalico's get_compensation needs one
    cfg = CFG.replace("correction_probe: carto", "correction_probe: beacon") \
        + "\n[axis_twist_compensation]\n"
    printer, atm = make(tmp_path, cfg)
    fc = configparser.RawConfigParser()
    fc.read_string(cfg)
    manager = katc.load_config(configfile.ConfigWrapper(
        printer, fc, {}, "axis_twist_compensation"))
    printer.objects["axis_twist_compensation"] = manager
    del printer.objects["probe_list"].probes["carto"]
    beacon = _beacon_mod.BeaconProbe(printer, 0.0, 20.0, nonadditive)
    beacon.bed = bed
    printer.objects["beacon"] = beacon
    printer.handlers["klippy:connect"].append(beacon._handle_connect)
    connect(printer)
    for cmd in (atm.cmd_STATUS, atm.cmd_CHECK, atm.cmd_CALIBRATE):
        cmd(FakeGcmd())
    assert act(atm) is not None


# ---------------------------------------------------------------------------
# Any registered probe as the reference

def test_any_probe_as_reference(tmp_path):
    cfg = CFG.replace("main_probe: load_cell_probe\n", "") + \
        "reference: klicky\n"
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    plist = printer.objects["probe_list"].probes
    plist["klicky"] = SimProbe(printer, "klicky", (0.0, 10.0, 0.5),
                               lambda nx, ny: 0.0)
    connect(printer)
    order = []
    for name in ("klicky", "carto"):
        orig = plist[name]._read

        def logged(orig=orig, name=name):
            order.append(name)
            return orig()
        plist[name]._read = logged
    atm.cmd_CALIBRATE(FakeGcmd())
    assert order[:6] == ["klicky"] * 3 + ["carto"] * 3
    assert atm.maps["carto"]["reference"] == "klicky"
    off = atm.report["mean_offset"]
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            c = probe_at(printer, plist["carto"], x, y)
            k = probe_at(printer, plist["klicky"], x, y)
            assert abs(c - k - off) < 1e-9
    # load cell still selectable per run
    atm.cmd_CALIBRATE(FakeGcmd({"REFERENCE": "load_cell"}))
    assert atm.maps["carto"]["reference"] == "load_cell_probe"


# ---------------------------------------------------------------------------
# Several correction probes, each with its own map

def eddy_err(nx, ny):
    return -0.2 - 0.5 * twist(nx) + 0.0002 * (ny - 100)


def make_two(tmp_path, cfg_extra=""):
    cfg = CFG.replace("correction_probe: carto",
                      "correction_probe: carto, eddy") + cfg_extra
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    plist = printer.objects["probe_list"].probes
    plist["eddy"] = SimProbe(printer, "eddy", (-20.0, 10.0, 1.5), eddy_err)
    connect(printer)
    return printer, atm, plist


def test_two_correction_probes_each_get_their_own_map(tmp_path):
    printer, atm, plist = make_two(tmp_path)
    lc, carto, eddy = plist["load_cell_probe"], plist["carto"], plist["eddy"]
    atm.cmd_CALIBRATE(FakeGcmd())                       # carto -> default
    off_c = atm.report["mean_offset"]
    g = FakeGcmd({"PROBE": "eddy"})
    atm.cmd_CALIBRATE(g)                                # -> default_eddy
    off_e = atm.report["mean_offset"]
    assert "[axis_twist_compensation eddy] and applied" in g.out[-1]
    assert atm.maps["carto"]["section"] == "axis_twist_compensation carto"
    assert atm.maps["eddy"]["section"] == "axis_twist_compensation eddy"
    assert set(atm.active) == {"carto", "eddy"}
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            m = probe_at(printer, lc, x, y)
            assert abs(probe_at(printer, carto, x, y) - m - off_c) < 1e-9
            assert abs(probe_at(printer, eddy, x, y) - m - off_e) < 1e-9
    # newer Kalico passes the probe: same answer
    pos = [80.0 + 20.0, 110.0 - 10.0, 2.0]
    assert abs(atm.get_z_compensation_value(pos, eddy)
               - atm.compensation_at(80.0 + 20.0 - 20.0, 110.0, "eddy")) \
        < 1e-12
    assert atm.get_z_compensation_value(pos, lc) == 0.0
    # clear one
    atm.cmd_CLEAR(FakeGcmd({"PROBE": "eddy"}))
    assert set(atm.active) == {"carto"}
    with pytest.raises(CmdError, match="PROBE must be one of"):
        atm.cmd_CALIBRATE(FakeGcmd({"PROBE": "nope"}))


def test_two_probes_restart_loads_each_default(tmp_path):
    printer, atm, plist = make_two(tmp_path)
    atm.cmd_CALIBRATE(FakeGcmd())
    atm.cmd_CALIBRATE(FakeGcmd({"PROBE": "eddy"}))
    cfg2 = saved_cfg(printer, CFG.replace("correction_probe: carto",
                                          "correction_probe: carto, eddy"))
    printer2, atm2 = make(tmp_path, cfg2, carto_err=nonadditive)
    printer2.objects["probe_list"].probes["eddy"] = SimProbe(
        printer2, "eddy", (-20.0, 10.0, 1.5), eddy_err)
    connect(printer2)
    assert {p: e["name"] for p, e in atm2.active.items()} == \
        {"carto": "axis_twist_compensation carto",
         "eddy": "axis_twist_compensation eddy"}
    g = FakeGcmd()
    atm2.cmd_STATUS(g)
    assert "'eddy' (found): [axis_twist_compensation eddy]" in g.out[0]
    g = FakeGcmd()
    atm2.cmd_CHECK(g)
    assert "MISMATCH" not in g.out[0]
    assert "reaches the correction probes" in g.out[0]


def test_cartographer_and_beacon_together(tmp_path):
    cfg = CFG.replace("correction_probe: carto",
                      "correction_probe: cartographer, beacon") + ATC_SECTION
    printer, atm = make(tmp_path, cfg)
    plist = printer.objects["probe_list"].probes
    del plist["carto"]
    carto = CartoPluginProbe(printer, (0.0, 15.0, 2.0), nonadditive)
    plist["cartographer"] = carto
    beacon = _beacon_mod.BeaconProbe(printer, 0.0, 20.0, eddy_err)
    beacon.bed = bed
    printer.objects["beacon"] = beacon

    class RealATC:
        def get_z_compensation_value(self, pos):
            return 0.0
    printer.objects["axis_twist_compensation"] = RealATC()
    printer.carto_has_atc_section = True
    printer.handlers["klippy:connect"].insert(0, beacon._handle_connect)
    connect(printer)
    lc = plist["load_cell_probe"]
    atm.cmd_CALIBRATE(FakeGcmd({"PROBE": "cartographer"}))
    off_c = atm.report["mean_offset"]
    atm.cmd_CALIBRATE(FakeGcmd({"PROBE": "beacon"}))
    off_b = atm.report["mean_offset"]
    for x in atm._grid_axis(0):
        for y in atm._grid_axis(1):
            m = probe_at(printer, lc, x, y)
            assert abs(probe_at(printer, carto, x, y) - m - off_c) < 1e-9
            assert abs(beacon_scan_height(printer, beacon, x, y) - m - off_b) \
                < 1e-9
            assert beacon_contact_height(printer, beacon, x, y) == bed(x, y)


# ---------------------------------------------------------------------------
# Round beds

ROUND_CFG = """
[axis_twist_mesh]
mesh_radius: 80
mesh_origin: 110, 110
round_probe_count: 7
main_probe: load_cell_probe
correction_probe: carto
"""


def test_round_bed(tmp_path):
    printer, atm = make(tmp_path, ROUND_CFG,
                        carto_err=lambda nx, ny: 0.3 + twist(nx))
    connect(printer)
    nodes = atm._grid_nodes()
    assert len(nodes) < 49
    assert all(math.hypot(x - 110, y - 110) <= 80 + 1e-6
               for _, _, x, y in nodes)
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    rep = atm.report
    assert rep["count"] == len(nodes)
    # pure X twist recovered on the circle by the additive fit
    assert rep["interaction_rms"] < 1e-6
    assert rep["y_effect_rms"] < 1e-6
    assert "        ." in g.out[-1]                 # outside the circle
    a = act(atm)
    tw = {x: twist(x) for x in atm._grid_axis(0)}
    for _, _, x, y in nodes:
        c = probe_at(printer, printer.objects["probe_list"].probes["carto"],
                     x, y)
        m = probe_at(printer, printer.objects["probe_list"].probes[
            "load_cell_probe"], x, y)
        assert abs(c - m - rep["mean_offset"]) < 1e-9
    sec = printer.objects["configfile"].sections[
        "axis_twist_compensation carto"]
    assert sec["mesh_radius"] == "80.000"
    assert sec["round_probe_count"] == "7"
    assert len(a["grid"]) == 7 and len(a["grid"][0]) == 7
    # restart round trip
    printer2, atm2 = make(tmp_path, saved_cfg(printer, ROUND_CFG))
    connect(printer2)
    assert act(atm2) is not None
    t = FakeGcmd()
    atm2.cmd_TABLE(t)
    assert "        ." in t.out[0]
    assert tw  # sanity
    with pytest.raises(configparser.Error, match="odd"):
        make(tmp_path, ROUND_CFG.replace("round_probe_count: 7",
                                         "round_probe_count: 6"))
    with pytest.raises(configparser.Error, match="not both"):
        make(tmp_path, ROUND_CFG + "mesh_min: 0, 0\n")


# ---------------------------------------------------------------------------
# Import from the old X / Y lines, and from / to mesh-based Kalico

class LegacyATC:
    """Older Kalico's line-based axis_twist_compensation (same math)."""
    z_compensations = [0.03, 0.0, -0.01, -0.03]
    compensation_start_x, compensation_end_x = 30.0, 180.0
    zy_compensations = [-0.02, 0.02]
    compensation_start_y, compensation_end_y = 40.0, 170.0

    @staticmethod
    def _line(c, values, start, end):
        n = len(values)
        spacing = (end - start) / (n - 1)
        t = (c - start) / spacing
        i = max(0, min(int(math.floor(t)), n - 2))
        t -= i
        return values[i] + t * (values[i + 1] - values[i])

    def get_z_compensation_value(self, pos):
        return (self._line(pos[0], self.z_compensations,
                           self.compensation_start_x,
                           self.compensation_end_x)
                + self._line(pos[1], self.zy_compensations,
                             self.compensation_start_y,
                             self.compensation_end_y))


def test_import_legacy_lines(tmp_path):
    printer, atm = make(tmp_path, CFG + ATC_SECTION, carto_err=nonadditive)
    legacy = LegacyATC()
    printer.objects["axis_twist_compensation"] = legacy
    connect(printer)
    plist = printer.objects["probe_list"].probes
    lc, carto = plist["load_cell_probe"], plist["carto"]
    line = LegacyATC.get_z_compensation_value.__get__(legacy)
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "z_compensations" in g.out[0]
    g = FakeGcmd({"SOURCE": "legacy"})
    atm.cmd_IMPORT(g)
    assert "as [axis_twist_compensation carto]" in g.out[0]
    prof = atm.maps["carto"]
    assert prof["algo"] == "direct" and prof["reference"] == "manual"
    # reproduces the lines exactly, including past their ends
    for nx in (0.0, 12.5, 30.0, 77.0, 180.0, 230.0):
        for ny in (-5.0, 40.0, 101.0, 170.0, 233.0):
            assert abs(atm.compensation_at(nx, ny + 15.0) - line([nx, ny, 0]))\
                < 1e-9
    # carto gets the map once (it replaces the lines), load cell still
    # gets the lines until they're deleted
    th = printer.objects["toolhead"]
    th.pos[0], th.pos[1] = 77.0, 101.0
    raw = bed(77.0, 116.0) + 2.0 + nonadditive(77.0, 101.0)
    assert abs(carto.run_probe(None)[2] - raw - line([77.0, 101.0, 0])) \
        < 1e-9
    assert abs(lc.run_probe(None)[2] - (bed(77.0, 101.0) - 0.1)
               - line([77.0, 101.0, 0])) < 1e-9
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "WARNING" not in g.out[0]
    g = FakeGcmd()
    atm.cmd_CHECK(g)
    assert "MISMATCH" not in g.out[0]
    with pytest.raises(CmdError, match="SOURCE must be"):
        atm.cmd_IMPORT(FakeGcmd({"SOURCE": "nope"}))


def test_import_legacy_needs_lines(tmp_path):
    printer, atm = make(tmp_path, CFG)
    connect(printer)
    with pytest.raises(CmdError, match="No z_compensations"):
        atm.cmd_IMPORT(FakeGcmd({"SOURCE": "legacy"}))


def make_new_kalico(tmp_path, extra_cfg=""):
    cfg = CFG + "\n[axis_twist_compensation]\n" + extra_cfg
    printer, atm = make(tmp_path, cfg, carto_err=nonadditive)
    fc = configparser.RawConfigParser()
    fc.read_string(cfg)
    printer.objects["axis_twist_compensation"] = katc.load_config(
        configfile.ConfigWrapper(printer, fc, {}, "axis_twist_compensation"))
    return printer, atm, fc


# ---------------------------------------------------------------------------
# Regression tests from the second review

_carto_fallback = types.ModuleType("cartographer.macros.fallback")
exec("def run(fn):\n    return fn()\n", _carto_fallback.__dict__)


def test_kalico_probe_under_plugin_command_gets_its_own_map(tmp_path):
    # e.g. Cartographer's BED_MESH_CALIBRATE falling back to Kalico's own
    # probing with PROBE=eddy: cartographer frames are deeper in the stack
    cfg = CARTO_CFG.replace("correction_probe: cartographer",
                            "correction_probe: cartographer, eddy")
    printer, atm, carto, _ = make_carto(tmp_path, cfg=cfg + ATC_SECTION)
    plist = printer.objects["probe_list"].probes
    eddy = SimProbe(printer, "eddy", (-20.0, 10.0, 1.5), eddy_err)
    plist["eddy"] = eddy
    atm.cmd_CALIBRATE(FakeGcmd({"PROBE": "cartographer"}))
    atm.cmd_CALIBRATE(FakeGcmd({"PROBE": "eddy"}))
    lc = plist["load_cell_probe"]
    off_e = atm.report["mean_offset"]
    for x, y in ((50.0, 65.0), (140.0, 110.0)):        # grid points
        e = _carto_fallback.run(lambda: probe_at(printer, eddy, x, y))
        m = probe_at(printer, lc, x, y)
        assert abs(e - m - off_e) < 1e-9


def test_per_probe_section_only_does_not_register_object(tmp_path):
    # mesh-based Kalico creates the object for per-probe sections itself;
    # older Kalico ignores them, so the add-on has to provide it
    cfg = CFG + "\n[axis_twist_compensation carto]\nreference_probe = x\n"
    printer, atm = make(tmp_path, cfg)
    assert ("axis_twist_compensation" in printer.objects) == (not NEW_ATC)


@pytest.mark.skipif(not NEW_ATC, reason="Kalico without mesh-based twist comp")
def test_cartographer_advice_on_mesh_based_kalico(tmp_path):
    printer, atm, carto, _ = make_carto(tmp_path, with_section=False)
    g = FakeGcmd()
    atm.cmd_CALIBRATE(g)
    assert "NOT applied" in g.out[-1] and "needs an update" in g.out[-1]


# ---------------------------------------------------------------------------
# Generic defaults (like Kalico's axis_twist_compensation)

BARE_CFG = """
[axis_twist_mesh]
mesh_min: 20, 20
mesh_max: 200, 200
"""


def test_generic_defaults(tmp_path):
    printer, atm = make(tmp_path, BARE_CFG)
    assert atm.reference == "manual" and atm.reference_probe is None
    assert atm.probe_count == (3, 3)
    assert atm.correction_probes == []           # resolved at connect
    plist = printer.objects["probe_list"]
    plist.default_probe = ("carto", plist.probes["carto"])
    connect(printer)
    assert atm.correction_probes == ["carto"]
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "reference: manual (paper)" in g.out[0]
    assert "correction probe 'carto'" in g.out[0]


def test_no_usable_default_probe(tmp_path):
    printer, atm = make(tmp_path, BARE_CFG + "reference: load_cell\n")
    plist = printer.objects["probe_list"]
    plist.default_probe = ("load_cell_probe", plist.probes["load_cell_probe"])
    connect(printer)
    assert atm.correction_probes == []
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "correction probe: none" in g.out[0]
    with pytest.raises(CmdError, match="set correction_probe"):
        atm.cmd_CALIBRATE(FakeGcmd())
    assert atm.get_z_compensation_value([1.0, 2.0, 3.0]) == 0.0


def test_beacon_as_default_probe(tmp_path):
    printer, atm = make(tmp_path, BARE_CFG + "reference: load_cell\n")
    del printer.objects["probe_list"].probes["carto"]
    beacon = _beacon_mod.BeaconProbe(printer, 0.0, 20.0, nonadditive)
    beacon.name = "beacon"
    printer.objects["beacon"] = beacon

    class Wrapper:          # what Beacon registers as "probe"
        pass
    w = Wrapper()
    w.beacon = beacon
    printer.objects["probe"] = w
    connect(printer)
    assert atm.correction_probes == ["beacon"]


def test_grid_is_required(tmp_path):
    with pytest.raises(configparser.Error, match="mesh_min and mesh_max"):
        make(tmp_path, "\n[axis_twist_mesh]\ncorrection_probe: carto\n")



@pytest.mark.skipif(not NEW_ATC, reason="Kalico without mesh-based twist comp")
def test_saved_map_is_kalicos_own_and_never_doubled(tmp_path):
    printer, atm = make(tmp_path, CFG, carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    want = {pt: atm.compensation_at(*pt) for pt in
            ((20.0, 20.0), (93.0, 61.0), (200.0, 200.0))}
    # restart with the SAVE_CONFIG output: Kalico's own code loads the map
    text = saved_cfg(printer)
    printer2, atm2 = make(tmp_path, text, carto_err=nonadditive)
    fc = configparser.RawConfigParser()
    fc.read_string(text)
    manager = katc.load_config(configfile.ConfigWrapper(
        printer2, fc, {}, "axis_twist_compensation"))
    printer2.objects["axis_twist_compensation"] = manager
    katc.load_config_prefix(configfile.ConfigWrapper(
        printer2, fc, {}, "axis_twist_compensation carto"))
    carto2 = printer2.objects["probe_list"].probes["carto"]
    lc2 = printer2.objects["probe_list"].probes["load_cell_probe"]
    for comp in manager.maps.values():
        comp.handle_connect()
    # Kalico alone (e.g. the hook broken by an update) still applies it
    for (bx, by), v in want.items():
        got = manager.get_z_compensation_value([bx, by - 15.0, 0.0], carto2)
        assert abs(got - v) < 1e-5
    # with the add-on hooked in: the same value, not twice
    connect(printer2)
    assert manager.get_z_compensation_value.__self__ is atm2
    for (bx, by), v in want.items():
        got = manager.get_z_compensation_value([bx, by - 15.0, 0.0], carto2)
        assert abs(got - v) < 1e-5
    assert manager.get_z_compensation_value([50.0, 50.0, 0.0], lc2) == 0.0
    g = FakeGcmd()
    atm2.cmd_CHECK(g)
    assert "MISMATCH" not in g.out[0] and "reaches" in g.out[0]


def test_apply_offset_is_folded_into_saved_points(tmp_path):
    printer, atm = make(tmp_path, CFG + "apply_offset: True\n"
                        "max_compensation: 1.0\n", carto_err=nonadditive)
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    pts = printer.objects["configfile"].sections[
        "axis_twist_compensation carto"]["points"]
    first = float(pts.split()[0].rstrip(","))
    off = atm.report["mean_offset"]
    assert abs(first - (atm.report["comp_grid"][0][0] - off)) < 1e-6
    assert abs(atm.compensation_at(20.0, 20.0) - first) < 1e-6


def test_old_default_profile_converted(tmp_path):
    old = CFG + ("\n[axis_twist_mesh default]\nversion = 1\nprobe = carto\n"
                 "x_count = 3\ny_count = 3\nmin_x = 20\nmax_x = 200\n"
                 "min_y = 20\nmax_y = 200\noffset = -0.5\n"
                 "reference = load_cell_probe\n"
                 "points =\n  0.01, 0.02, 0.03\n  0.0, 0.0, 0.0\n"
                 "  -0.01, -0.02, -0.03\n"
                 "\n[axis_twist_mesh spare]\nversion = 1\nprobe = carto\n"
                 "x_count = 2\ny_count = 2\nmin_x = 0\nmax_x = 10\n"
                 "min_y = 0\nmax_y = 10\npoints =\n  0, 0\n  0, 0\n")
    printer, atm = make(tmp_path, old)
    connect(printer)
    sec = printer.objects["configfile"].sections[
        "axis_twist_compensation carto"]
    assert sec["reference_probe"] == "load_cell_probe"
    assert abs(atm.compensation_at(200.0, 20.0) - 0.03) < 1e-9
    assert "spare" in atm.old_profiles
    out = []
    printer.objects["gcode"].respond_info = lambda m, log=True: out.append(m)
    printer.objects["gcode"].respond_raw = lambda m: out.append(m)
    for h in printer.handlers.get("klippy:ready", []):
        h()
    assert any("converted old profile [axis_twist_mesh default]" in m
               for m in out)
    g = FakeGcmd()
    atm.cmd_STATUS(g)
    assert "old profiles not used" in g.out[0] and "spare" in g.out[0]
    atm.cmd_PROFILE(FakeGcmd({"REMOVE": "spare"}))
    assert "spare" not in atm.old_profiles


def test_startup_self_test_warns_when_hook_is_lost(tmp_path):
    printer, atm = make(tmp_path, CFG + ATC_SECTION, carto_err=nonadditive)

    class ATC:
        def get_z_compensation_value(self, pos, probe=None):
            return 0.0
    atc = ATC()
    printer.objects["axis_twist_compensation"] = atc
    connect(printer)
    atm.cmd_CALIBRATE(FakeGcmd())
    out = []
    printer.objects["gcode"].respond_info = lambda m, log=True: out.append(m)
    printer.objects["gcode"].respond_raw = lambda m: out.append(m)
    for h in printer.handlers["klippy:ready"]:
        h()
    assert not any("NOT reaching" in m for m in out)
    # a Kalico update replaces the method: the next start says so
    atc.get_z_compensation_value = ATC.get_z_compensation_value.__get__(atc)
    for h in printer.handlers["klippy:ready"]:
        h()
    assert any(m.startswith("!! axis_twist_mesh: compensation is NOT")
               for m in out)


def test_kalico_call_path_detected():
    # the real Kalico probe code (whichever version KALICO points at) still
    # asks axis_twist_compensation for each reading
    assert axis_twist_mesh.AxisTwistMesh._kalico_call_path(None) is True
