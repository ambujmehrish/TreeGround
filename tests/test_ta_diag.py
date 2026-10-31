import numpy as np
import pandas as pd

from test_s2 import box, load_script
from treeground import masks as M


def test_pick_relation_and_paired_bootstrap():
    td = load_script("ta_diag")
    rles = M.encode(np.stack([box(10, 90, 10, 70), box(10, 40, 10, 40), box(60, 90, 80, 120)]))
    areas = M.areas(rles).astype(float)
    ious = np.array([0.2, 0.9, 0.0])                                         # target = node 1 (a part)
    rows = [td.pick_rows("s", np.array(lg), ious, areas, rles, "q", "part", "im")
            for lg in ([0, 5, 1], [5, 0, 1], [0, 1, 5])]
    assert [r["relation"] for r in rows] == ["correct", "too_big", "elsewhere"]
    assert [r["success"] for r in rows] == [True, False, False]
    ref = pd.DataFrame([{"qid": f"q{i}", "image": f"im{i % 4}", "covered": True, "success": True,
                         "relation": "correct"} for i in range(40)])
    oth = ref.copy()
    oth.loc[:9, ["success", "relation"]] = [False, "too_big"]
    res = td.paired(ref, oth, 500, 0)
    assert res["queries"] == 40 and res["images"] == 4
    assert np.isclose(res["success_ref_minus_other"]["mean"], 0.25)
    assert np.isclose(res["wrong_level_other_minus_ref"]["mean"], 0.25)
    lo, hi = res["success_ref_minus_other"]["ci95"]
    assert lo <= 0.25 <= hi
