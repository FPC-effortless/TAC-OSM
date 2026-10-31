import Mathlib

/-!
LANGMATH101 candidate forward formulations over exact reals.

This is a typed one-stream forward specification, not an IEEE/AD proof.
The Python functional references test finite-precision correspondence against
the actual source. Surrogate derivatives, runtime validation and owner ACLs are
separate obligations; no learned capability follows from these definitions.
`coord` totalizes out-of-range indexing. Registered geometries/layouts only use
in-range indices; the source validates its corresponding shape contracts.
-/
namespace PNDS.LANGMATH101
noncomputable section
open scoped BigOperators

abbrev Vec (n : ℕ) := Fin n → ℝ
abbrev Byte := Fin 256

def coord {n} (x : Vec n) (i : ℕ) : ℝ := if h : i < n then x ⟨i,h⟩ else 0
def elems {n} (x : Vec n) : List ℝ := (List.finRange n).map x
def pack {n} (xs : List ℝ) : Vec n := fun i => (xs[i.val]?).getD 0
def vmap {n} (f : ℝ → ℝ) (x : Vec n) : Vec n := fun i => f (x i)
def vadd {n} (x y : Vec n) : Vec n := fun i => x i+y i
def vmul {n} (x y : Vec n) : Vec n := fun i => x i*y i
def vscale {n} (a : ℝ) (x : Vec n) : Vec n := fun i => a*x i
def vzero (n : ℕ) : Vec n := fun _ => 0
def slice {n m} (x : Vec n) (offset : ℕ) : Vec m := fun i => coord x (offset+i.val)
def dot {n} (x y : Vec n) : ℝ := ∑ i, x i*y i
def sigmoid (x : ℝ) := 1/(1+Real.exp (-x))
def squash (x : ℝ) := 2*sigmoid (2*x)-1
def silu (x : ℝ) := x*sigmoid x
def softmax {n} (x : Vec n) : Vec n := fun i => Real.exp (x i)/(∑ j,Real.exp (x j))
def erf (x : ℝ) := 2/Real.sqrt Real.pi * ∫ y in (0:ℝ)..x, Real.exp (-y*y)
def gelu (x : ℝ) := x*(1+erf (x/Real.sqrt 2))/2

structure Linear (input output : ℕ) where
  weight : Fin output → Vec input
  bias : Vec output
def linear {i o} (p : Linear i o) (x : Vec i) : Vec o :=
  fun j => dot (p.weight j) x+p.bias j

structure Norm (n : ℕ) where
  weight : Vec n
  bias : Vec n
def layerNorm {n} (p : Norm n) (x : Vec n) : Vec n :=
  let mean := (∑ i,x i)/(n:ℝ)
  let variance := (∑ i,(x i-mean)^2)/(n:ℝ)
  fun i => (x i-mean)/Real.sqrt (variance+0.00001)*p.weight i+p.bias i

structure Head (input width : ℕ) where
  norm : Norm input
  up : Linear input width
  gate : Linear input width
  out : Linear width 256
def head {d w} (p : Head d w) (x : Vec d) : Vec 256 :=
  let z := layerNorm p.norm x
  linear p.out (vmul (linear p.up z) (vmap silu (linear p.gate z)))

structure Bank (keys values : ℕ) where
  matrix : Fin values → Vec keys
  mass : Vec keys
def bankUpdate {k v} (s : Bank k v) (a : ℝ) (key : Vec k) (value : Vec v) : Bank k v :=
  ⟨fun i j => a*s.matrix i j+(1-a)*value i*key j,
   fun j => a*s.mass j+(1-a)*key j⟩
def bankRead {k v} (s : Bank k v) (query : Vec k) : Vec v :=
  fun i => dot (s.matrix i) query/(dot s.mass query+0.0001)
def retention (bank : Fin 2) (raw : ℝ) : ℝ :=
  if bank=0 then 0.97+(0.995-0.97)*sigmoid raw
  else 0.995+(0.99995-0.995)*sigmoid raw

structure Geometry where
  embed : ℕ
  context : ℕ
  key : ℕ
  value : ℕ
  width : ℕ
def small : Geometry := ⟨32,48,8,16,138⟩

structure NativeParams (g : Geometry) where
  embedding : Byte → Vec g.embed
  lag : Fin 3 → Vec g.embed
  contextInput : Linear g.embed g.context
  decay : Vec g.context
  address : Linear (2*g.embed+g.context) (4*g.key+2*g.value+2)
  compare : Linear (g.embed+g.context+3*g.value+2) g.value
  readout : Head (2*g.embed+g.context+3*g.value+2) g.width
  bankMin : Vec 2
  bankMax : Vec 2

structure NativeState (g : Geometry) where
  clock : ℕ
  context : Vec g.context
  lexical : Fin 3 → Vec g.embed
  previousKey : Fin 2 → Vec g.key
  banks : Fin 2 → Bank g.key g.value

def nextLexical {d} (old : Fin 3 → Vec d) (e : Vec d) : Fin 3 → Vec d :=
  fun l => if l=0 then old 1 else if l=1 then old 2 else e
def localStem {d} (e : Vec d) (old lag : Fin 3 → Vec d) : Vec d :=
  fun i => squash (e i+∑ l:Fin 3,squash (lag l i)*old ⟨2-l.val,by omega⟩ i)

/-- Exact source layout: current embedding/local/prior context; lag-key/current-value binding. -/
def nativeStep {g} (p : NativeParams g) (s : NativeState g) (byte : Byte)
    (writeAllowed : Bool := true) : Vec 256 × NativeState g :=
  let e := vmap squash (p.embedding byte)
  let localValue := localStem e s.lexical p.lag
  let offer := vmap squash (linear p.contextInput localValue)
  let delta := vmap (fun x => sigmoid (-x)) p.decay
  let context := fun i => delta i*s.context i+(1-delta i)*offer i
  let features : Vec (2*g.embed+g.context) := pack (elems e++elems localValue++elems s.context)
  let raw := linear p.address features
  let key : Fin 2 → Vec g.key := fun h => softmax (slice raw (h.val*g.key))
  let query : Fin 2 → Vec g.key := fun h => softmax (slice raw (2*g.key+h.val*g.key))
  let value : Fin 2 → Vec g.value := fun h => vmap squash (slice raw (4*g.key+h.val*g.value))
  let alpha : Fin 2 → ℝ := fun h =>
    if s.clock=0 ∨ writeAllowed=false then 1 else
      p.bankMin h+(p.bankMax h-p.bankMin h)*sigmoid (coord raw (4*g.key+2*g.value+h.val))
  let banks := fun h => bankUpdate (s.banks h) (alpha h) (s.previousKey h) (value h)
  let reads := fun h => bankRead (banks h) (query h)
  let evidence : Vec 2 := fun h => dot (banks h).mass (query h)
  let comparison : Vec (g.embed+g.context+3*g.value+2) := pack
    (elems localValue++elems s.context++elems (reads 0)++elems (reads 1)++
      elems (fun i => |reads 0 i-reads 1 i|)++elems evidence)
  let gate := vmap sigmoid (linear p.compare comparison)
  let mixed := fun i => gate i*reads 0 i+(1-gate i)*reads 1 i
  let z : Vec (2*g.embed+g.context+3*g.value+2) := pack
    (elems e++elems localValue++elems s.context++elems (reads 0)++elems (reads 1)++elems mixed++elems evidence)
  (head p.readout z,⟨s.clock+1,context,nextLexical s.lexical e,key,banks⟩)

/-- A1 receives only a bounded window of the current invocation, with absent positions zero padded. -/
def statelessFeatures {window d} (embedding : Byte → Vec d)
    (observed : Fin window → Option Byte) : Vec (window*d) :=
  pack ((List.finRange window).flatMap (fun lag =>
    elems (match observed lag with | none => vzero d | some b => vmap squash (embedding b))))
def statelessForward {window d w} (embedding : Byte → Vec d)
    (readout : Head (window*d) w) (observed : Fin window → Option Byte) : Vec 256 :=
  head readout (statelessFeatures embedding observed)

structure BlockState (c k v : ℕ) where
  context : Vec c
  key : Fin 2 → Vec k
  banks : Fin 2 → Bank k v
structure AssociativeParams (d c k v : ℕ) where
  contextInput : Linear d c
  decay : Vec c
  address : Linear (d+c) (4*k+2*v+2)
  compare : Linear (d+c+3*v+2) v
  norm : Norm (d+c+3*v+2)
  out : Linear (d+c+3*v+2) d
  update : Option (Linear d 1)
def rate {d} (p : Option (Linear d 1)) (h : Vec d) : ℝ :=
  match p with | none => 1 | some q => sigmoid (linear q h 0)

def associativeStep {d c k v} (p : AssociativeParams d c k v)
    (clock : ℕ) (s : BlockState c k v) (h : Vec d) : Vec d × BlockState c k v :=
  let u := rate p.update h
  let delta := vmap (fun x => sigmoid (-x)) p.decay
  let a := fun i => 1-u*(1-delta i)
  let offer := vmap squash (linear p.contextInput h)
  let context := fun i => a i*s.context i+(1-a i)*offer i
  let raw := linear p.address (pack (elems h++elems s.context))
  let keys : Fin 2 → Vec k := fun b => softmax (slice raw (b.val*k))
  let queries : Fin 2 → Vec k := fun b => softmax (slice raw (2*k+b.val*k))
  let values : Fin 2 → Vec v := fun b => vmap squash (slice raw (4*k+b.val*v))
  let alpha := fun b : Fin 2 => if clock=0 then 1 else
    1-u*(1-retention b (coord raw (4*k+2*v+b.val)))
  let banks := fun b => bankUpdate (s.banks b) (alpha b) (s.key b) (values b)
  let reads := fun b => bankRead (banks b) (queries b)
  let evidence : Vec 2 := fun b => dot (banks b).mass (queries b)
  let compareInput : Vec (d+c+3*v+2) := pack
    (elems h++elems s.context++elems (reads 0)++elems (reads 1)++
      elems (fun i => |reads 0 i-reads 1 i|)++elems evidence)
  let gate := vmap sigmoid (linear p.compare compareInput)
  let mixed := fun i => gate i*reads 0 i+(1-gate i)*reads 1 i
  let z := pack (elems h++elems s.context++elems (reads 0)++elems (reads 1)++elems mixed++elems evidence)
  (vadd h (vmap squash (linear p.out (layerNorm p.norm z))),⟨context,keys,banks⟩)

structure RecurrentParams (d c width : ℕ) where
  contextInput : Linear d c
  decay : Vec c
  norm : Norm (d+c)
  up : Linear (d+c) width
  gate : Linear (d+c) width
  out : Linear width d
  update : Option (Linear d 1)
def recurrentStep {d c w} (p : RecurrentParams d c w) (state : Vec c)
    (h : Vec d) : Vec d × Vec c :=
  let u := rate p.update h
  let a := fun i => 1-u*(1-sigmoid (-p.decay i))
  let offer := vmap squash (linear p.contextInput h)
  let next := fun i => a i*state i+(1-a i)*offer i
  let z := layerNorm p.norm (pack (elems h++elems state))
  (vadd h (linear p.out (vmul (linear p.up z) (vmap silu (linear p.gate z)))),next)

/-- Each optional upward projection is applied before that layer. The first is absent. -/
def assocLayers {d c k v} (clock : ℕ)
    (layers : List (Option (Linear d d) × AssociativeParams d c k v))
    (states : List (BlockState c k v)) (h : Vec d) :
    Option (Vec d × List (BlockState c k v) × List (Vec d)) :=
  match layers,states with
  | [],[] => some (h,[],[])
  | (up,p)::ps,s::ss =>
      let input := match up with | none => h | some u => vmap squash (linear u h)
      let result := associativeStep p clock s input
      do let (top,next,levels) ← assocLayers clock ps ss result.1
         pure (top,result.2::next,result.1::levels)
  | _,_ => none
def recurrentLayers {d c w}
    (layers : List (Option (Linear d d) × RecurrentParams d c w))
    (states : List (Vec c)) (h : Vec d) : Option (Vec d × List (Vec c) × List (Vec d)) :=
  match layers,states with
  | [],[] => some (h,[],[])
  | (up,p)::ps,s::ss =>
      let input := match up with | none => h | some u => vmap squash (linear u h)
      let result := recurrentStep p s input
      do let (top,next,levels) ← recurrentLayers ps ss result.1
         pure (top,result.2::next,result.1::levels)
  | _,_ => none
def downward {d} (top : Vec d) (levels : List (Vec d))
    (projections : List (Linear d d)) : Vec d :=
  ((levels.dropLast.zip projections).reverse).foldl
    (fun h pair => vadd pair.1 (vmap squash (linear pair.2 h))) top

/-- A3/A4 full decoder: the downward path affects readout, not stored next states. -/
def associativeStackForward {d c k v width} (clock : ℕ)
    (embedding : Byte → Vec d) (lag lexical : Fin 3 → Vec d)
    (layers : List (Option (Linear d d) × AssociativeParams d c k v))
    (states : List (BlockState c k v)) (down : List (Linear d d))
    (readout : Head d width) (byte : Byte) :
    Option (Vec 256 × (List (BlockState c k v) × (Fin 3 → Vec d))) := do
  let e := vmap squash (embedding byte)
  let (top,next,levels) ← assocLayers clock layers states (localStem e lexical lag)
  pure (head readout (downward top levels down),(next,nextLexical lexical e))
def recurrentStackForward {d c w width}
    (embedding : Byte → Vec d) (lag lexical : Fin 3 → Vec d)
    (layers : List (Option (Linear d d) × RecurrentParams d c w))
    (states : List (Vec c)) (down : List (Linear d d))
    (readout : Head d width) (byte : Byte) :
    Option (Vec 256 × (List (Vec c) × (Fin 3 → Vec d))) := do
  let e := vmap squash (embedding byte)
  let (top,next,levels) ← recurrentLayers layers states (localStem e lexical lag)
  pure (head readout (downward top levels down),(next,nextLexical lexical e))

structure GRU (input hidden : ℕ) where
  inputAffine : Linear input (3*hidden)
  hiddenAffine : Linear hidden (3*hidden)
def gruStep {d h} (p : GRU d h) (x : Vec d) (old : Vec h) : Vec h :=
  let gi := linear p.inputAffine x
  let gh := linear p.hiddenAffine old
  let r := fun i : Fin h => sigmoid (coord gi i.val+coord gh i.val)
  let z := fun i : Fin h => sigmoid (coord gi (h+i.val)+coord gh (h+i.val))
  let n := fun i : Fin h => squash (coord gi (2*h+i.val)+r i*coord gh (2*h+i.val))
  fun i => (1-z i)*n i+z i*old i
def gruForward {width} (embedding : Byte → Vec 32) (gru : GRU 32 64)
    (readout : Head 64 width) (state : Vec 64) (byte : Byte) : Vec 256 × Vec 64 :=
  let next := gruStep gru (vmap squash (embedding byte)) state
  (head readout next,next)
structure SSM where
  embedding : Byte → Vec 32
  retain : Linear 32 64
  offer : Linear 32 64
  readout : Linear 64 64
  current : Linear 32 64
def ssmForward {width} (p : SSM) (readout : Head 64 width)
    (old : Vec 64) (byte : Byte) : Vec 256 × Vec 64 :=
  let e := vmap squash (p.embedding byte)
  let a := vmap (fun x => 0.999*sigmoid x) (linear p.retain e)
  let offer := vmap squash (linear p.offer e)
  let next := fun i => a i*old i+(1-a i)*offer i
  let h := vmap squash (vadd (linear p.readout next) (linear p.current e))
  (head readout h,next)

structure Transformer (width : ℕ) where
  embedding : Byte → Vec 32
  attentionNorm : Norm 32
  query : Linear 32 32
  key : Linear 32 32
  value : Linear 32 32
  attentionOut : Linear 32 32
  ffnNorm : Norm 32
  ffnUp : Linear 32 64
  ffnOut : Linear 64 32
  readout : Head 32 width
def position (t : ℕ) : Vec 32 := fun i =>
  let phase := (t:ℝ)*Real.exp (((2*(i.val/2):ℕ):ℝ)*(-Real.log 10000/32))
  if i.val%2=0 then Real.sin phase else Real.cos phase
def transformerForward {width n} (p : Transformer width)
    (observed : Fin (n+1) → Byte) (t : Fin (n+1)) : Vec 256 :=
  let input := fun j => vadd (vscale (Real.sqrt 32) (p.embedding (observed j))) (position j.val)
  let z := fun j => layerNorm p.attentionNorm (input j)
  let q := linear p.query (z t)
  let k := fun j => linear p.key (z j)
  let v := fun j => linear p.value (z j)
  let score := fun (bank : Fin 4) j =>
    (∑ i:Fin 8,coord q (bank.val*8+i.val)*coord (k j) (bank.val*8+i.val))/Real.sqrt 8
  let weight := fun bank j => if j.val≤t.val then Real.exp (score bank j) else 0
  let attended : Vec 32 := fun i =>
    (∑ j,weight ⟨i.val/8,by omega⟩ j*coord (v j) i.val)/(∑ j,weight ⟨i.val/8,by omega⟩ j)
  let h := vadd (input t) (linear p.attentionOut attended)
  let h := vadd h (linear p.ffnOut (vmap gelu (linear p.ffnUp (layerNorm p.ffnNorm h))))
  head p.readout h

/-- Explicit temporal metadata for the single-state conventional controls. -/
structure TemporalState (dimension : ℕ) where
  clock : ℕ
  hidden : Vec dimension
def gruByteStep {width} (embedding : Byte → Vec 32) (gru : GRU 32 64)
    (readout : Head 64 width) (s : TemporalState 64) (byte : Byte) :
    Vec 256 × TemporalState 64 :=
  let result := gruForward embedding gru readout s.hidden byte
  (result.1,⟨s.clock+1,result.2⟩)
def ssmByteStep {width} (p : SSM) (readout : Head 64 width)
    (s : TemporalState 64) (byte : Byte) : Vec 256 × TemporalState 64 :=
  let result := ssmForward p readout s.hidden byte
  (result.1,⟨s.clock+1,result.2⟩)

/-- Transformer state is the entire observed raw prefixTokens, with clock equal to its length. -/
structure TransformerState where
  observed : List Byte
def transformerByteStep {width} (p : Transformer width) (s : TransformerState)
    (byte : Byte) : Vec 256 × TransformerState :=
  let history := s.observed++[byte]
  let observed : Fin (s.observed.length+1) → Byte := fun i =>
    history.get ⟨i.val,by simpa [history] using i.isLt⟩
  (transformerForward p observed ⟨s.observed.length,by omega⟩,⟨history⟩)

/-! A5 primal graphs. Straight-through derivatives are deliberately NOT encoded
as derivatives of argmax: that would be mathematically false. The Python
functional reference verifies the actual surrogate's autograd behavior.
-/
inductive PatchMode where
  | byte | discrete | latent | adaptive
inductive PatchKind where
  | discrete | latent
  deriving DecidableEq
structure PatchParams (hidden : ℕ) where
  embedding : Byte → Vec 24
  recurrent : GRU 24 hidden
  readout : Linear hidden 256
  pool : Option (Linear 96 24)
  codebook : Option (Fin 16 → Vec 24)
  controller : Option (Linear (hidden+49) 3)
structure PatchState (hidden : ℕ) where
  clock : ℕ
  hidden : Vec hidden
  pending : List (Vec 24)
  pendingKind : PatchKind
  pendingBytes : List Byte
def PatchState.Valid {h} (s : PatchState h) : Prop :=
  s.pending.length=s.pendingBytes.length ∧ s.pending.length<4
def patchPool {h} (p : PatchParams h) (items : List (Vec 24)) : Option (Vec 24) :=
  match p.pool with
  | none => none
  | some pool => some (vmap squash (linear pool (pack (items.flatMap elems))))
def squaredDistance {d} (x y : Vec d) : ℝ := ∑ i,(x i-y i)^2
def codeIndex (codebook : Fin 16 → Vec 24) (pooled : Vec 24) : Fin 16 :=
  (List.finRange 16).foldl
    (fun best i => if squaredDistance pooled (codebook i)<squaredDistance pooled (codebook best)
      then i else best) 0
def patchRepresentation {h} (p : PatchParams h) (items : List (Vec 24))
    (kind : PatchKind) : Option (Vec 24) :=
  match items with
  | [] => none
  | [single] => some single
  | _ => do
      let pooled ← patchPool p items
      match kind,p.codebook with
      | PatchKind.latent,_ => some pooled
      | PatchKind.discrete,some codes => some (codes (codeIndex codes pooled))
      | _,_ => none
def argmax3 (score : Vec 3) : Fin 3 :=
  (List.finRange 3).foldl (fun best i => if score best<score i then i else best) 0
def patchRoute {h} (p : PatchParams h) (mode : PatchMode)
    (s : PatchState h) (current : Vec 24) (allItems : List (Vec 24)) : Option (Fin 3) :=
  match mode with
  | PatchMode.byte => some 0
  | PatchMode.discrete => some 1
  | PatchMode.latent => some 2
  | PatchMode.adaptive => do
      let pooled ← patchPool p allItems
      let gate ← p.controller
      let probability := softmax (linear p.readout s.hidden)
      let entropy := -(∑ i,probability i*Real.log (probability i))/5.545177444479562
      let features : Vec (h+49) := pack (elems s.hidden++elems current++elems pooled++[entropy])
      pure (argmax3 (linear gate features))
def patchReadout {h} (p : PatchParams h) (s : PatchState h) : Vec 256 :=
  linear p.readout s.hidden
def patchStep {h} (p : PatchParams h) (mode : PatchMode) (s : PatchState h)
    (byte : Byte) : Option (Vec 256 × PatchState h) := do
  let current := p.embedding byte
  let allItems := s.pending++[current]
  let route ← patchRoute p mode s current allItems
  if route=0 then
    let before ← if s.pending.isEmpty then some s.hidden else do
      let rep ← patchRepresentation p s.pending s.pendingKind
      pure (gruStep p.recurrent rep s.hidden)
    let next := gruStep p.recurrent current before
    pure (linear p.readout next,⟨s.clock+1,next,[],s.pendingKind,[]⟩)
  else
    let kind := if route=1 then PatchKind.discrete else PatchKind.latent
    if allItems.length=4 then
      let rep ← patchRepresentation p allItems kind
      let next := gruStep p.recurrent rep s.hidden
      pure (linear p.readout next,⟨s.clock+1,next,[],kind,[]⟩)
    else
      pure (linear p.readout s.hidden,⟨s.clock+1,s.hidden,allItems,kind,s.pendingBytes++[byte]⟩)

abbrev BPEToken := Fin 289
def mergePair (pair : BPEToken × BPEToken) (newToken : BPEToken)
    (tokens : List BPEToken) : List BPEToken :=
  match tokens with
  | a::b::rest => if a=pair.1 ∧ b=pair.2 then newToken::mergePair pair newToken rest
      else a::mergePair pair newToken (b::rest)
  | short => short
termination_by tokens.length
decreasing_by all_goals simp_wf
def tokenizeBPE (merges : Fin 32 → BPEToken × BPEToken) (observed : List Byte) : List BPEToken :=
  (List.finRange 32).foldl
    (fun ids rank => mergePair (merges rank) ⟨256+rank.val,by omega⟩ ids)
    (observed.map (fun b => ⟨b.val,by omega⟩))
def bpeFold {h} (embedding : BPEToken → Vec 24) (recurrent : GRU 24 h)
    (initial : Vec h) (tokens : List BPEToken) : Vec h :=
  tokens.foldl (fun state token => gruStep recurrent (embedding token) state) initial
def bpeForward {h} (embedding : BPEToken → Vec 24) (recurrent : GRU 24 h)
    (readout : Linear h 256) (merges : Fin 32 → BPEToken × BPEToken)
    (observed : List Byte) : Vec 256 :=
  linear readout (bpeFold embedding recurrent (vzero h) (288::tokenizeBPE merges observed))

/-- Cached prefixTokens equality assumes unchanged parameters and an actual prefixTokens
state, as checked by the runtime version/identity guard. It does not prove
that a mutable cache is immune to arbitrary storage tampering. -/
theorem bpe_prefix_cache {h} (embedding : BPEToken → Vec 24) (recurrent : GRU 24 h)
    (initial : Vec h) (prefixTokens suffix : List BPEToken) :
    bpeFold embedding recurrent initial (prefixTokens++suffix)=
    bpeFold embedding recurrent (bpeFold embedding recurrent initial prefixTokens) suffix := by
  simp [bpeFold,List.foldl_append]

/-- Hidden-state-only terminal decoding: pending raw bytes have no direct head path. -/
theorem patch_readout_noninterference {h} (p : PatchParams h)
    (s t : PatchState h) (sameHidden : s.hidden=t.hidden) :
    patchReadout p s=patchReadout p t := by
  simp [patchReadout,sameHidden]

def unroll {State Input} (step : State → Input → State) (s : State) (xs : List Input) := xs.foldl step s
theorem unroll_chunks {State Input} (step : State → Input → State)
    (s : State) (xs ys : List Input) :
    unroll step s (xs++ys)=unroll step (unroll step s xs) ys := by
  simp [unroll,List.foldl_append]

/-- The stored native transition does not depend on final-head parameters. -/
theorem native_readout_does_not_update_state {g} (p : NativeParams g)
    (other : Head (2*g.embed+g.context+3*g.value+2) g.width)
    (s : NativeState g) (x : Byte) (allowed : Bool) :
    (nativeStep {p with readout:=other} s x allowed).2=(nativeStep p s x allowed).2 := by
  rfl

end
end PNDS.LANGMATH101

#print axioms PNDS.LANGMATH101.unroll_chunks
#print axioms PNDS.LANGMATH101.native_readout_does_not_update_state
#print axioms PNDS.LANGMATH101.bpe_prefix_cache
#print axioms PNDS.LANGMATH101.patch_readout_noninterference
