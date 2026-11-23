"""v2 observations never copy truth into observed columns (audit P1-3): tracker and the observed part of
products_v2 work from receiver detections only; truth is read only after the "# world side" marker (labels)."""

import inspect
import re

from sim.obs_model import observe_v2, tracker_v2

FORBIDDEN = re.compile(r"\b(e|best|em|emitter|truth)\.(protocol|bandwidth_khz|freq_mhz|kind|pair_id|eirp_dbm|id)\b")


def test_tracker_never_touches_emitters():
    src = inspect.getsource(tracker_v2)
    assert not FORBIDDEN.search(src)
    assert "emitter" not in src.lower().replace("emitters", "")


def test_observed_columns_of_products_v2_come_from_detections():
    src = inspect.getsource(observe_v2.products_v2)
    observed, _, world = src.partition("# world side")
    assert world, "products_v2 must mark where world-side labelling starts"
    assert not FORBIDDEN.search(observed)


def test_detection_record_has_no_truth_fields():
    fields = set(observe_v2.Detection.__dataclass_fields__)
    assert not fields & {"emitter_id", "kind", "pair_id", "true_protocol"}
