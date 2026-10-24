import Std

/-!
Full typed dependency graph of the ESWAM013 **observational** native forward:
recorded actor features -> shared parallel encoder -> recent fast + causal
motion -> 256-frame slow + eight observed associative source-ID slots ->
learned relevance -> gated fusion -> per-actor exchange decoder ->
bounded real physical velocity continuation correction.

This is a mathematical structural/causality specification. Neural affine
weights, floating-point softmax, gradients and source dynamics are represented
as abstract operators; Lean kernel checking of these theorems is NOT a proof of
real-world prediction accuracy, torch numerical equivalence or causality of
logged interventions.
-/
namespace ESWAM013

abbrev Vector (α : Type) (dim : Nat) := Fin dim → α
abbrev Actors (α : Type) (dim : Nat) := Fin 4 → Vector α dim
abbrev SourceFrames (α : Type) := Fin 256 → Actors α 8
abbrev EmbeddedFrames (α : Type) := Fin 256 → Actors α 66
abbrev SourceMemory (α : Type) := Fin 4 → Fin 8 → Vector α 32
abbrev PredictedPhysical (α : Type) := Fin 4 → Fin 2 → Vector α 2

structure ObservedInput (α : Type) where
  sourceRecordedFrames : SourceFrames α
  originalActorObserved : Fin 4 → Bool
  measuredVelocityPrior : PredictedPhysical α

structure WorldForecast (α : Type) where
  predictedPhysical : PredictedPhysical α
  predictedScale : PredictedPhysical α

structure Operators (α : Type) where
  encodeParallel : SourceFrames α → EmbeddedFrames α
  recentFast : EmbeddedFrames α → Actors α 66
  causalMotionDifference : SourceFrames α → Actors α 32
  motionAugment : Actors α 66 → Actors α 32 → Actors α 66
  longSlow : EmbeddedFrames α → Actors α 66
  observedEightKeys : EmbeddedFrames α → SourceMemory α
  observedEightValues : EmbeddedFrames α → SourceMemory α
  queryRecent : Actors α 66 → Actors α 32
  sourceIdScopedRead : SourceMemory α → SourceMemory α →
                       Actors α 32 → Actors α 32
  trustLongEvidence : Actors α 66 → Actors α 66 →
                      Actors α 32 → Fin 4 → α
  relevantSlow : Actors α 66 → Actors α 66 →
                 (Fin 4 → α) → Actors α 66
  relevantMemory : Actors α 32 → (Fin 4 → α) → Actors α 32
  gatedFastSlowFusion : Actors α 66 → Actors α 66 →
                        Actors α 32 → Actors α 66
  actorExchangeHead : Actors α 66 → SourceFrames α →
                      ObservedInput α → WorldForecast α
  recordedSpeed : SourceFrames α → Fin 4 → α
  sourceMotionOpen : α → α

def wholeNativeStep {α : Type} (ops : Operators α)
    (input : ObservedInput α) : WorldForecast α := Id.run do
  let encoded := ops.encodeParallel input.sourceRecordedFrames
  let fast := ops.recentFast encoded
  let observedMotion := ops.causalMotionDifference input.sourceRecordedFrames
  let motionFast := ops.motionAugment fast observedMotion
  let slow := ops.longSlow encoded
  let observedKeys := ops.observedEightKeys encoded
  let observedValues := ops.observedEightValues encoded
  let ownQuery := ops.queryRecent motionFast
  let ownMemory := ops.sourceIdScopedRead observedKeys observedValues ownQuery
  let relevance := ops.trustLongEvidence motionFast slow observedMotion
  let selectedSlow := ops.relevantSlow motionFast slow relevance
  let selectedMemory := ops.relevantMemory ownMemory relevance
  let worldCore := ops.gatedFastSlowFusion motionFast selectedSlow selectedMemory
  let proposal := ops.actorExchangeHead worldCore input.sourceRecordedFrames input
  let masked := fun actor horizon coordinate =>
    input.measuredVelocityPrior actor horizon coordinate +
      ops.sourceMotionOpen (ops.recordedSpeed input.sourceRecordedFrames actor) *
      (proposal.predictedPhysical actor horizon coordinate -
        input.measuredVelocityPrior actor horizon coordinate)
  return {predictedPhysical := masked, predictedScale := proposal.predictedScale}

-- Rat is a concrete exact field already available in Lean core (no mathlib
-- dependency), so the algebra below can be kernel-checked independently.
def relevanceMix (fast slow trust : Rat) : Rat :=
  fast + trust * (slow - fast)

theorem zero_relevance_selects_fast (fast slow : Rat) :
    relevanceMix fast slow 0 = fast := by
  simp [relevanceMix]

theorem unit_relevance_selects_slow (fast slow : Rat) :
    relevanceMix fast slow 1 = slow := by
  simp [relevanceMix]

def physicalMotionGate (measuredSpeed positiveOffset : Rat) : Rat :=
  measuredSpeed / (measuredSpeed + positiveOffset)

def physicalResidual
    (originalMeasuredVelocityPrior predictedCorrection measuredSpeed positiveOffset : Rat) : Rat :=
  originalMeasuredVelocityPrior +
    physicalMotionGate measuredSpeed positiveOffset * predictedCorrection

theorem zero_measured_speed_exact_physical_prior
    (sourceMeasuredVelocityPrior correction offset : Rat) :
    physicalResidual sourceMeasuredVelocityPrior correction 0 offset =
      sourceMeasuredVelocityPrior := by
  simp [physicalResidual, physicalMotionGate]

-- Actor index is a source-authorized array address; there is no learned
-- ID-classification theorem. Model has no do(action) input or inferred action
-- authority and the three-role robot state is not fabricated here.

end ESWAM013
