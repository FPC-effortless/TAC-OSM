# TACOSM-PLM-C5-STATE-ADMISSION-006

## Purpose

Transfer the already-developed C5 OR-LSH admission mechanism into persistent
PLM state addressing.

This is explicitly not a new LSH search. The adapter inherits C5's dense CDL
student, OR-LSH construction, K90 calibration procedure and rho-derived table
schedule. The experiment asks whether that machinery transfers when the
candidate objects are treated as persistent state records.

## Protocol

- seeds: 0..4
- M: 1024, 2048, 4096, 8192
- calibration M: 1024
- 64 trials per M/seed
- K90 alpha: 0.10
- OR-LSH cap: 128
- target query noise: 0.0625

The index remains resident during each cell's repeated queries; there is no
query-time rebuild.

## Metrics

- inherited K90;
- inherited rho;
- requested/actual tables;
- target admission rate;
- raw candidates;
- admitted candidates;
- routing operations;
- routing fraction relative to M.

## Scientific constraint

This is a mechanism-transfer test. It does not select a new representation,
does not establish semantic-language retrieval, and does not establish an
asymptotic law.

A positive result means only that the already-measured C5 admission machinery
survives when embedded in the persistent-state layer.
