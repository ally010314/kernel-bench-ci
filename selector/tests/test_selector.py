import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from selector import dataset as D  # noqa: E402
from selector import evaluate as E  # noqa: E402
from selector import features as F  # noqa: E402
from selector.__main__ import main, train  # noqa: E402
from selector.model import GainModel, Policy, Selector, regret  # noqa: E402

DATA = ROOT / "selector" / "data" / "dataset.csv"
PRED = ROOT / "experiments" / "records" / "predictions"


class Features(unittest.TestCase):
    def test_v3_config_matches_registered_predictions(self):
        # 선택기의 설정은 사전 등록한 v3 예측 (3 · 4차) 과 같아야 한다
        for f in ("tile-40-r3_v3.csv", "tile-40-r4_v3.csv"):
            for r in csv.DictReader(open(PRED / f)):
                c = F.features(r["chip"], r["model"])
                self.assertEqual(f"{c['fc1']}_{c['fc2']}_s{c['stage']}", r["pred"], (f, r["chip"], r["model"]))

    def test_shape_dict_and_mixer_share(self):
        a = F.features("ada", dict(N=196, dim=1024, hidden=4096, blocks=24, token_hidden=512))
        b = F.features("ada", dict(N=197, dim=1024, hidden=4096, blocks=24))
        self.assertGreater(a["mlp_share"], b["mlp_share"])  # attention 이 없는 Mixer 가 MLP 비중이 크다

    def test_unknown_model(self):
        with self.assertRaises(KeyError):
            F.shape_of("nope")

    def test_round_of(self):
        self.assertEqual([D.round_of(s) for s in ("sep128", "sep", "tile-40-new-1107-2353", "tile-40-r3-1108-0637", "tile-40-r4-beit-1108-1435", "sep103", "res2")],
                         [0, 0, 1, 2, 3, None, None])


class Policies(unittest.TestCase):
    def test_decisions(self):
        p = Policy()
        self.assertEqual(p.decide("h100", dict(N=197), -5)[0], "measure")
        self.assertEqual(p.decide("rtx5090", dict(N=197), -9)[0], "measure")
        self.assertEqual(p.decide("ada", dict(N=50), -9)[0], "measure")
        self.assertEqual(p.decide("ada", dict(N=197), +3)[0], "custom")
        self.assertEqual(Policy(trt_above=2.0).decide("ada", dict(N=197), +3)[0], "trt")

    def test_regret(self):
        self.assertEqual(regret("measure", 5.0), 0.0)
        self.assertEqual(regret("custom", 1.5), 1.5)
        self.assertEqual(regret("custom", -3.0), 0.0)
        self.assertEqual(regret("trt", -3.0), 3.0)


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = D.load_csv(DATA)
        cls.sel, cls.rep = train(cls.rows)

    def test_dataset(self):
        self.assertGreaterEqual(len(self.rows), 187)
        self.assertEqual({r["chip"] for r in self.rows}, set(F.CHIPS))
        self.assertFalse(any(r["src"] == "sep103" for r in self.rows))

    def test_gate_passes(self):
        self.assertTrue(self.sel.meta["gate_ok"], self.sel.meta["gate"])

    def test_m3_beats_constant(self):
        f = self.rep["forms"]
        self.assertLess(f["M3"]["lomo"]["mae"], f["B0"]["lomo"]["mae"])
        self.assertLess(f["M3"]["lomo"]["mae"], 3.0)

    def test_policy_beats_always_custom(self):
        p = self.rep["policy"]
        self.assertLess(p["채택 정책"]["lomo"]["regret"], p["항상 커스텀"]["lomo"]["regret"])
        self.assertLessEqual(p["채택 정책"]["lomo"]["regret_worst"], 2.0)

    def test_save_load_predict(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "s.json"); self.sel.save(f); s2 = Selector.load(f)
            a, b = self.sel.predict("ada", "vitL"), s2.predict("ada", "vitL")
            self.assertEqual(a, b)
            self.assertEqual(a["action"], "custom")
            self.assertLess(a["interval"][0], a["expected_gain"])
            self.assertEqual(s2.predict("h100", "vitL")["action"], "measure")

    def test_gain_model_roundtrip(self):
        g = GainModel("M3").fit(self.rows)
        g2 = GainModel.from_dict(json.loads(json.dumps(g.to_dict())))
        self.assertAlmostEqual(g.predict(self.rows[0]), g2.predict(self.rows[0]), places=5)


class Cli(unittest.TestCase):
    def test_predict_model_file(self):
        with tempfile.TemporaryDirectory() as d:
            m = os.path.join(d, "s.json")
            self.assertEqual(main(["train", "--data", str(DATA), "--out", m]), 0)
            y = os.path.join(d, "x.yml")
            open(y, "w").write("name: x\ntimm: x\nshape: {N: 197, dim: 768, hidden: 3072, blocks: 12}\nchips: [ada, rtx5090]\n")
            out = os.path.join(d, "out.json")
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                main(["predict", y, "--model-file", m, "--format", "json"])
            P = json.loads(buf.getvalue())
            self.assertEqual([p["chip"] for p in P], ["ada", "rtx5090"])
            self.assertEqual([p["action"] for p in P], ["custom", "measure"])


if __name__ == "__main__":
    unittest.main()
