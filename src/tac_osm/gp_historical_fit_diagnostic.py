"""GP-UNC-006: real-source historical FIT diagnostic *only*.

A strictly gated sequential source-event evaluator. It admits separate issues
(with no truth field) and later observed target events. The original trained
base has already seen all FIT labels during historical training: these are
IN-SAMPLE diagnostics and CANNOT establish independent generalization, calibrated
uncertainty, or model superiority.

Two 64D two-inducing projected GP controls:
  dynamic: genuine frozen original native associative state from FIT observations
  blinded: constant 64D zero input with matched 2x64 inducing storage
Both update from the same source events using identical hyperparameters and
real label arrival chronology. No new neural parameters are fitted.
"""
from __future__ import annotations
import heapq
import math
from collections.abc import Mapping

from .gp_rankone_nofit import RankOneProjectedResidualGP
from .gp_uncertainty_nofit import _vector

ISSUE_FIELDS=frozenset(("kind","row_id","entity_id","features","base_prediction",
                        "feature_at","issued_at","checkpoint_sha256"))
LABEL_FIELDS=frozenset(("kind","row_id","truth","available_at"))
ARMS=("frozen_native","constant_gp","native_memory_gp")

class HistoricalFITSidecar:
    """No independent TEST path. Caller authenticates original checkpoint/source."""

    def __init__(self, *, source_sha256, checkpoint_sha256, native_inducing,
                 noise_variance=.5, max_records=10000):
        if not (isinstance(source_sha256,str) and len(source_sha256)==64 and
                isinstance(checkpoint_sha256,str) and len(checkpoint_sha256)==64 and
                all(c in "0123456789abcdef" for c in source_sha256+checkpoint_sha256)):
            raise ValueError("pinned SHA256 source/checkpoint required")
        if type(max_records) is not int or not 1<=max_records<=100000:
            raise ValueError("bounded source record count required")
        if not isinstance(native_inducing,Mapping) or not native_inducing:
            raise ValueError("frozen original inducing dictionary required")
        self.sha=checkpoint_sha256
        self.source=source_sha256
        self.dynamic={}
        self.constant={}
        for key,z in native_inducing.items():
            if not isinstance(key,str) or not key:
                raise ValueError("source entity id missing")
            z=tuple(_vector(a) for a in z)
            if len(z)!=2 or any(len(a)!=64 for a in z):
                raise ValueError("two frozen 64D native inducing points required")
            self.dynamic[key]=RankOneProjectedResidualGP(
                z,noise_variance=noise_variance,jitter=1e-8)
            self.constant[key]=RankOneProjectedResidualGP(
                ([0.]*64,[1.]*64),noise_variance=noise_variance,jitter=1e-8)
        self.max_records=max_records
        self.seen=set()
        self.issued={}
        self.pending=[]
        self.counts={arm:0 for arm in ARMS}
        self.abs_error={arm:0.0 for arm in ARMS}
        self.square_error={arm:0.0 for arm in ARMS}
        self.labels_missing=0
        self.updates=0
        self._serial=0
        self.last_at=-math.inf
        self.last_issue_time=-math.inf

    def _num(self,value):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError("nonfinite real-source datum/time rejected")
        return float(value)

    def _clock(self,now):
        when=self._num(now)
        if when < self.last_at:
            raise ValueError("events reversed actual original source chronology")
        self.last_at=when
        return when

    def issue(self,event):
        if not isinstance(event,Mapping) or set(event)!=ISSUE_FIELDS or event.get("kind")!="issue":
            raise ValueError("forecast issue must not carry truth, future mask or outcome")
        now=self._clock(event["issued_at"])
        if now < self.last_issue_time:
            raise ValueError("neural issues must remain chronological")
        self.last_issue_time=now
        rid,entity=event["row_id"],event["entity_id"]
        if (not isinstance(rid,str) or not rid or rid in self.seen or
                len(self.seen)>=self.max_records or entity not in self.dynamic):
            raise ValueError("unknown entity, duplicate or unbounded forecast")
        if event["checkpoint_sha256"]!=self.sha:
            raise ValueError("original trained checkpoint differs")
        feat=_vector(event["features"])
        if len(feat)!=64 or self._num(event["feature_at"])>now:
            raise ValueError("invalid native memory feature issue-time boundary")
        base=self._num(event["base_prediction"])
        # Past label events may update models ONLY on a LATER issue timestamp.
        while self.pending and self.pending[0][0] < now:
            at,_,old_entity,old_feat,residual,original_at=heapq.heappop(self.pending)
            for gp,x in ((self.dynamic[old_entity],old_feat),
                         (self.constant[old_entity],(0.0,)*64)):
                gp.observe(x,residual,feature_at=original_at,issued_at=original_at,
                           label_at=at,cutoff=now)
            self.updates+=1
        memory_p=self.dynamic[entity].predict(feat).mean
        constant_p=self.constant[entity].predict((0.0,)*64).mean
        prediction={"frozen_native":base,
                    "constant_gp":base+constant_p,
                    "native_memory_gp":base+memory_p}
        if any(not math.isfinite(v) for v in prediction.values()):
            raise ValueError("nonfinite model output")
        self.issued[rid]=(entity,feat,base,now,prediction)
        self.seen.add(rid)
        return prediction

    def label(self,event):
        if not isinstance(event,Mapping) or set(event)!=LABEL_FIELDS or event.get("kind")!="label_arrived":
            raise ValueError("label requires a distinct outcome-arrival event")
        now=self._clock(event["available_at"])
        rid=event["row_id"]
        if rid not in self.issued:
            raise ValueError("outcome lacks pre-existing original neural forecast")
        entity,feat,base,issued_at,preds=self.issued[rid]
        if now<=issued_at:
            raise ValueError("label cannot become available at/before decision")
        y=self._num(event["truth"])
        del self.issued[rid]
        for arm in ARMS:
            error=preds[arm]-y
            self.abs_error[arm]+=abs(error)
            self.square_error[arm]+=error*error
            self.counts[arm]+=1
        # The true residual cannot affect any prior evaluation output.
        heapq.heappush(self.pending,(now,self._serial,entity,feat,y-base,issued_at))
        self._serial+=1

    def missing_label(self,row_id):
        """Explicitly report a missing original measured outcome; no GP update."""
        if row_id not in self.issued:
            raise ValueError("unknown or duplicated missing-label receipt")
        del self.issued[row_id]
        self.labels_missing+=1

    def report(self):
        n=self.counts["frozen_native"]
        if n<1 or any(self.counts[x]!=n for x in ARMS):
            raise ValueError("source diagnostic requires aligned scored outcomes")
        if self.issued:
            raise ValueError("all forecast issues require observed or missing outcome receipt")
        output={a:{"n":n,"mae_standardized_log_units":self.abs_error[a]/n,
                   "rmse_standardized_log_units":math.sqrt(self.square_error[a]/n)}
                for a in ARMS}
        # Descriptive in-FIT difference only, no superiority hypothesis test.
        return {"evidence":"AUTHENTIC_HISTORICAL_FIT_ONLY_IN_SAMPLE",
                "frozen_checkpoint_sha256":self.sha,
                "original_source_sha256":self.source,
                "original_label_events_scored":n,
                "original_missing_label_receipts":self.labels_missing,
                "gp_real_fit_residual_update_events":self.updates,
                "remaining_unused_late_residuals":len(self.pending),
                "all_arm_same_outcomes":True,
                "gp_parameters_preselected_without_scoring":True,
                "arms":output,
                "independent_evaluation":False,
                "trained_native_weights_changed":False,
                "model_head_to_head_superiority":False,
                "probabilistic_calibration_scored":False,
                "protected_test_access":False,
                "scientific_promotion":"NOT_QUALIFIED"}
