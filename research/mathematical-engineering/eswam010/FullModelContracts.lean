import Std

namespace ESWAMFull

structure Shape where
  batch : Nat
  observation : Nat
  action : Nat
  width : Nat
  slots : Nat
  memoryWidth : Nat
  deriving Repr, DecidableEq

structure Contract where
  shape : Shape
  finiteInputs : Prop
  validObservedMask : Prop
  validEntityIds : Prop
  positiveStateScale : Prop

def encoderInputWidth (s : Shape) : Nat := 2*s.observation + s.action + 1
def encoderOutputWidth (s : Shape) : Nat := s.width
def fastGateOutputWidth (s : Shape) : Nat := 2*s.width
def slowGateOutputWidth (s : Shape) : Nat := 2*s.width
def memoryRoleCount : Nat := 3
def decoderInputWidth (s : Shape) : Nat := 2*s.width + s.memoryWidth
def physicalHeadWidth (s : Shape) : Nat := 2*s.observation
def actionHeadWidth (s : Shape) : Nat := s.action

theorem physical_head_is_mean_and_scale (s : Shape) :
    physicalHeadWidth s = s.observation + s.observation := by
  simp [physicalHeadWidth, Nat.two_mul]

theorem decoder_input_is_fast_slow_and_memory (s : Shape) :
    decoderInputWidth s = s.width + s.width + s.memoryWidth := by
  simp [decoderInputWidth, Nat.two_mul, Nat.add_assoc]

theorem encoder_includes_observation_mask_action_time (s : Shape) :
    encoderInputWidth s = s.observation + s.observation + s.action + 1 := by
  simp [encoderInputWidth, Nat.two_mul, Nat.add_assoc]

theorem gate_splits_fast_candidate_and_gate (s : Shape) :
    fastGateOutputWidth s = s.width + s.width := by
  simp [fastGateOutputWidth, Nat.two_mul]

theorem gate_splits_slow_candidate_and_gate (s : Shape) :
    slowGateOutputWidth s = s.width + s.width := by
  simp [slowGateOutputWidth, Nat.two_mul]

inductive EvidenceRole where
  | observed | predicted | verified
  deriving Repr, DecidableEq

def roleIndex : EvidenceRole → Fin 3
  | .observed => ⟨0, by decide⟩
  | .predicted => ⟨1, by decide⟩
  | .verified => ⟨2, by decide⟩

theorem prediction_role_not_observation :
    roleIndex .predicted ≠ roleIndex .observed := by decide

theorem verification_role_not_prediction :
    roleIndex .verified ≠ roleIndex .predicted := by decide

def canWriteObservation (hasMeasuredCoordinate : Bool) : Bool :=
  hasMeasuredCoordinate

theorem missing_observation_cannot_write :
    canWriteObservation false = false := by rfl

end ESWAMFull
