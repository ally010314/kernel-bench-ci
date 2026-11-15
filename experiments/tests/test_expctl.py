import json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import expctl
import stairs
import tiles

SPEC = dict(name="t", models="m", fixed_tile="x", _models=[["a", "b", "384"]], targets={"a6000": dict(trt=["10.3.0"], gpus=[1])})


class Validate(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(expctl.validate(SPEC), [])

    def test_5090_rejects_103(self):
        s = dict(SPEC, targets={"rtx5090": dict(trt=["10.3.0"], gpus=[0])})
        self.assertIn("미지원", expctl.validate(s)[0])

    def test_gpu_of_other_users(self):
        s = dict(SPEC, targets={"ada": dict(trt=["10.3.0"], gpus=[3])})
        self.assertIn("허용되지 않은 GPU", expctl.validate(s)[0])

    def test_shipped_specs(self):
        for p in (Path(expctl.ME) / "specs").glob("*.yaml"):
            self.assertEqual(expctl.validate(expctl.load_spec(p.stem)), [], p.name)


class Tiles(unittest.TestCase):
    T = dict(fc1=["128x128"], fc2=["64x64", "128x128"], stages=[3, 4])
    S = dict(SPEC, fixed_tile="128x128_64x64_s4", tiles=T)

    def test_names(self):
        self.assertEqual(tiles.names(self.T), ["128x128_64x64_s3", "128x128_64x64_s4", "128x128_128x128_s3", "128x128_128x128_s4"])

    def test_macros(self):
        self.assertEqual(tiles.tile_flags("128x32_128x128_s4"), "-DTB1_M=128 -DWP1_M=64 -DTB1_N=32 -DWP1_N=16 -DTB2_M=128 -DWP2_M=64 -DTB2_N=128 -DWP2_N=64 -DST1=3 -DST2=4")

    def test_valid(self):
        self.assertEqual(expctl.validate(self.S), [])

    def test_fixed_tile_required(self):
        self.assertIn("고정 타일", expctl.validate(dict(self.S, fixed_tile="64x64_64x64_s3"))[0])

    def test_fc1_m256_rejected(self):
        self.assertIn("fc1 타일", tiles.check(dict(self.T, fc1=["256x64"]))[0])

    def test_bad_flags(self):
        self.assertIn("flags", tiles.check(self.T, "-DA; rm -rf x")[0])

    def test_all_devices_can_build(self):
        s = dict(self.S, targets={d: dict(trt=x["trt"], gpus=x["gpus"]) for d, x in expctl.DEV.items()})
        self.assertEqual([x for x in expctl.validate(s) if "빌드" in x], [])

    def test_some_failures_continue(self):
        self.assertEqual(tiles.split_result({"a": "", "b": "compile"}, "a"), (["b"], ""))

    def test_fixed_failure_stops(self):
        self.assertIn("고정", tiles.split_result({"a": "compile", "b": ""}, "a")[1])

    def test_all_failed_stops(self):
        self.assertIn("전부", tiles.split_result({"a": "compile"}, "b")[1])

    def test_failed_tile_record(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for r in (dict(ph=1, n="b1", x=None), dict(ph=1, n="t1", x=None), dict(ph=1, n="t2", x=dict(med=1)), dict(ph=2, n="c2", x=None)):
                f.write(json.dumps(r) + "\n")
        self.assertEqual(tiles.failed_tiles(f.name), ["t1"]); os.remove(f.name)


class TileRows(unittest.TestCase):
    R = [dict(ph=1, n="b1", x=dict(med=2)), dict(ph=1, n="a", x=dict(med=1), diff=0.01, top1=True),
         dict(ph=1, n="b", x=dict(med=1.1), diff=0.5, top1=True), dict(ph=1, n="c", x=None),
         dict(ph=2, n="a", m="s", x=dict(med=3)), dict(ph=2, n="a", m="s", x=dict(med=5)), dict(ph=2, n="a", m="s", x=dict(med=4)), dict(ph=2, n="c2:a", m="s", x=dict(med=9))]

    def test_every_candidate_kept(self):
        self.assertEqual(sorted(tiles.rows_by_tile(self.R)), ["a", "b", "c"])

    def test_output_check(self):
        v = tiles.rows_by_tile(self.R)
        self.assertTrue(v["a"]["selectable"]); self.assertFalse(v["b"]["selectable"]); self.assertFalse(v["c"]["ran"])

    def test_old_format(self):
        v = tiles.rows_by_tile([dict(ph=1, n="b1", ms=2), dict(ph=1, n="a", ms=1.5, diff=0.01, top1=True), dict(ph=1, n="b", ms=None)])
        self.assertEqual(v["a"]["metrics"]["graph_ms"], 1.5); self.assertFalse(v["b"]["ran"])


    def test_repeat_median(self):
        m = tiles.rows_by_tile(self.R)["a"]["metrics"]
        self.assertEqual((m["graph_ms"], m["repeat_stream_ms"], m["repeat_stream_n"]), (1, 4, 3.0))


class Stairs(unittest.TestCase):
    def test_ns_are_tile_multiples(self):
        self.assertEqual(stairs.ns_for("128x128_128x64_s3", 200), [64, 128, 192])

    def test_cta(self):
        self.assertEqual(stairs.cta("128x128_128x64_s3", 192), 24)
        self.assertEqual(stairs.cta("128x128_64x32_s3", 64), 32)

    def test_spec_needs_settings(self):
        s = dict(SPEC, type="stairs")
        self.assertTrue(any("stairs" in x for x in expctl.validate(s)))


class Stall(unittest.TestCase):
    def test_no_progress_is_stalled(self):
        self.assertEqual(expctl.classify(dict(state="RUNNING"), 0, True, (expctl.STALL_H + 1) * 3600), "STALLED")

    def test_recent_progress_is_running(self):
        self.assertEqual(expctl.classify(dict(state="RUNNING"), 0, True, 60), "RUNNING")


class Gpu(unittest.TestCase):
    def test_busy(self):
        self.assertIn("사용 중", expctl.judge_gpu({1: (1800, 60)}, {}, 1800, [1])[0])

    def test_clock(self):
        self.assertIn("클럭", expctl.judge_gpu({1: (1410, 0)}, {}, 1800, [1])[0])

    def test_other_user(self):
        self.assertIn("다른 사용자", expctl.judge_gpu({1: (1800, 0)}, {1: {"x"}}, 1800, [1])[0])

    def test_clean(self):
        self.assertEqual(expctl.judge_gpu({1: (1800, 0)}, {}, 1800, [1]), [])


class Classify(unittest.TestCase):
    def test_all_fail_is_not_done(self):
        self.assertEqual(expctl.classify(dict(state="FAILED", expected=13), 13, False), "FAILED")

    def test_claimed_success_needs_rows(self):
        self.assertEqual(expctl.classify(dict(state="SUCCEEDED", expected=13), 12, False), "PARTIAL")

    def test_dead_worker(self):
        self.assertEqual(expctl.classify(dict(state="RUNNING", expected=13), 3, False), "LOST")

    def test_running(self):
        self.assertEqual(expctl.classify(dict(state="RUNNING", expected=13), 3, True), "RUNNING")


class Auto(unittest.TestCase):
    def setUp(self):
        self.gpu, self.lock = expctl.check_gpu, expctl.locked_gpus

    def tearDown(self):
        expctl.check_gpu, expctl.locked_gpus = self.gpu, self.lock

    def test_skips_busy_and_locked(self):
        expctl.check_gpu = lambda d, g: ["사용 중"] if g == [2] else []
        expctl.locked_gpus = lambda d: {1}
        self.assertEqual(expctl.pick_gpus("a6000", 2), [0, 3])

    def test_none_free_stops(self):
        expctl.check_gpu = lambda d, g: ["사용 중"]
        expctl.locked_gpus = lambda d: set()
        s = dict(targets={"a6000": dict(trt=["10.3.0"], gpus="auto")})
        self.assertEqual(expctl.resolve_gpus(s, []), ["a6000"])

    def test_spec_accepts_auto(self):
        self.assertEqual(expctl.validate(dict(SPEC, targets={"a6000": dict(trt=["10.3.0"], gpus="auto")})), [])


class Kernel(unittest.TestCase):
    def test_all_match(self):
        self.assertEqual(expctl.kernel_mismatch({"ada": "ab", "orin": "ab"}, "ab"), {})

    def test_device_differs_from_repo(self):
        self.assertEqual(expctl.kernel_mismatch({"ada": "ab", "orin": "cd"}, "ab"), {"orin": "cd"})


class FailFast(unittest.TestCase):
    def test_rule(self):
        src = (Path(expctl.ME) / "tools/worker.py").read_text()
        ns = {}
        exec(src[src.index("def should_abort"):src.index("def env_for")], ns)
        self.assertTrue(ns["should_abort"](0, 2, 2))
        self.assertFalse(ns["should_abort"](1, 5, 2))   # 성공이 하나라도 있으면 끝까지 잰다
        self.assertFalse(ns["should_abort"](0, 1, 2))


class StairsFitTest(unittest.TestCase):
    def test_fit_finds_wave(self):
        import os; os.environ.setdefault("MLFLOW_TRACKING_URI", "http://x")
        import collect
        X = list(range(8, 201, 8)); Y = [10 + 50 * -(-c // 22) for c in X]
        R = max((collect.stairs_fit(X, Y, S), S) for S in range(4, 201))
        self.assertEqual(R[1], 22)
        self.assertLess(collect.stairs_fit(X, Y, 1), R[0])


if __name__ == "__main__":
    unittest.main()
