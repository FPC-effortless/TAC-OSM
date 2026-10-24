import Std

/-!
A typed abstract composition of the entire actual ES-WAM observation-to-output
world step. Real-valued matrix/activation semantics, real-time numerical stability,
source ID admission, gradient correctness and trained intelligence are NOT proved
by these structural lemmas. No sorry, admit or custom axioms.
-/
namespace ESWAM

abbrev Vec (α : Type) (n : Nat) := Fin n → α

structure Input (α : Type) (d a : Nat) where
  observation : Vec α d
  known : Fin d → Bool
  loggedAction : Vec α a
  deltaTime : α
  sourceId : Nat
  goal : Vec α a

structure Memory (α : Type) (slots m : Nat) where
  ids : Fin slots → Option Nat
  keys : Fin slots → Vec α m
  observed : Fin slots → Option (Vec α m)
  predicted : Fin slots → Option (Vec α m)
  verified : Fin slots → Option (Vec α m)

structure State (α : Type) (w slots m : Nat) where
  fast : Vec α w
  slow : Vec α w
  memory : Memory α slots m

structure Forecast (α : Type) (d a w slots m : Nat) where
  meanNext : Vec α d
  positiveScale : Vec α d
  proposedAction : Vec α a
  nextState : State α w slots m
  previousPredictionPresent : Bool
  observationWritten : Bool
  predictionWritten : Bool

def storeObserved {α : Type} {s m : Nat} (memory : Memory α s m)
    (slot : Fin s) (value : Vec α m) : Memory α s m :=
  { memory with observed := Function.update memory.observed slot (some value) }

def storePredicted {α : Type} {s m : Nat} (memory : Memory α s m)
    (slot : Fin s) (value : Vec α m) : Memory α s m :=
  { memory with predicted := Function.update memory.predicted slot (some value) }

theorem predicted_write_preserves_observed {α : Type} {s m : Nat}
    (memory : Memory α s m) (slot : Fin s) (value : Vec α m) :
    (storePredicted memory slot value).observed = memory.observed := by rfl

theorem predicted_write_preserves_verified {α : Type} {s m : Nat}
    (memory : Memory α s m) (slot : Fin s) (value : Vec α m) :
    (storePredicted memory slot value).verified = memory.verified := by rfl

theorem observed_write_preserves_predicted {α : Type} {s m : Nat}
    (memory : Memory α s m) (slot : Fin s) (value : Vec α m) :
    (storeObserved memory slot value).predicted = memory.predicted := by rfl

theorem observed_write_preserves_verified {α : Type} {s m : Nat}
    (memory : Memory α s m) (slot : Fin s) (value : Vec α m) :
    (storeObserved memory slot value).verified = memory.verified := by rfl

-- Operator signatures give the true dependency order and tensor dimensions.
-- All nonlinear and learned operators are abstract functions: neither their
-- implementations nor their numerical properties are proved by type checking.
structure Operators (α : Type) (d a w slots m : Nat) where
  encode : Input α d a → Vec α w
  key : Vec α w → Vec α m
  observedCode : Vec α w → Vec α m
  readPrediction : Memory α slots m → Nat → Vec α m → Vec α m × Bool
  verificationCorrection : Vec α w → Vec α m → Vec α m → Bool → Vec α w
  fastGate : Vec α w → Vec α w → Vec α w
  slowGate : Vec α w → Vec α w → Vec α w
  admit : Memory α slots m → Nat → Option (Fin slots)
  query : Vec α w → Vec α m
  readObserved : Memory α slots m → Vec α m → Vec α m
  decode : Vec α w → Vec α w → Vec α m → Vec α 64
  physical : Input α d a → Vec α 64 → Vec α d × Vec α d
  action : Vec α 64 → Vec α a → Vec α a
  predictedCode : Vec α 64 → Vec α m

def hasObservation {α : Type} {d a : Nat} (input : Input α d a) : Bool :=
  (List.finRange d).any input.known

def step {α : Type} {d a w slots m : Nat}
    (op : Operators α d a w slots m) (input : Input α d a)
    (old : State α w slots m) : Forecast α d a w slots m := Id.run do
  let encoded := op.encode input
  let sourceKey := op.key encoded
  let (oldPrediction, hasPrediction) :=
    op.readPrediction old.memory input.sourceId sourceKey
  let observedValue := op.observedCode encoded
  let corrected := op.verificationCorrection old.fast observedValue oldPrediction hasPrediction
  let fast := op.fastGate encoded corrected
  let slow := op.slowGate fast old.slow
  let slot := op.admit old.memory input.sourceId
  let wasObserved := hasObservation input
  let mem1 := match slot with
    | some s => if wasObserved then storeObserved old.memory s observedValue else old.memory
    | none => old.memory
  let factualWritten := wasObserved && slot.isSome
  let remembered := op.readObserved mem1 (op.query encoded)
  let decoded := op.decode fast slow remembered
  let (mean, scale) := op.physical input decoded
  let proposal := op.action decoded input.goal
  let predicted := op.predictedCode decoded
  let mem2 := match slot with
    | some s => storePredicted mem1 s predicted
    | none => mem1
  return {
    meanNext := mean, positiveScale := scale, proposedAction := proposal,
    nextState := {fast := fast, slow := slow, memory := mem2},
    previousPredictionPresent := hasPrediction,
    observationWritten := factualWritten, predictionWritten := slot.isSome
  }

end ESWAM
