import Std

namespace ESWAM

def matches (slots : List (Option Nat)) (id : Nat) : Bool :=
  slots.any (fun s => s == some id)

def hasFree (slots : List (Option Nat)) : Bool :=
  slots.any (fun s => s == none)

def admissible (slots : List (Option Nat)) (id : Nat) (enabled : Bool) : Bool :=
  enabled && (matches slots id || hasFree slots)

theorem disabled_refused (slots : List (Option Nat)) (id : Nat) :
    admissible slots id false = false := by
  simp [admissible]

theorem full_collision_refused (slots : List (Option Nat)) (id : Nat)
    (hfull : hasFree slots = false) (habsent : matches slots id = false) :
    admissible slots id true = false := by
  simp [admissible, hfull, habsent]

theorem existing_identity_admitted (slots : List (Option Nat)) (id : Nat)
    (hmatch : matches slots id = true) :
    admissible slots id true = true := by
  simp [admissible, hmatch]

theorem free_slot_admitted (slots : List (Option Nat)) (id : Nat)
    (hfree : hasFree slots = true) :
    admissible slots id true = true := by
  simp [admissible, hfree]

end ESWAM
