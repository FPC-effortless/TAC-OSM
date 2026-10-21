# ME042 — cross-domain real transformer measurements, full TAC architecture portfolio

Registered 2026-10-21 before ME042 fits. Stacked as separate draft on ME041 without modifying its active 88-fit Tetouan run.

## Objective

Research owner prioritizes full model training and accuracy/compute improvements, not diagnostic-only experiments. Train the **same eleven full predictive models** from ME041 on authentic electricity transformer hourly load and oil temperature measurements. Eight exactly same-active-capacity trained ideas (small adapter baseline, causal pair relevance, CDL-inspired innovation write, sparse assemblies, phase modulation, role-filler multiplicative binding, iterative latent readout, unified) plus unchanged ME038 full native, ME040 paired native, and original context-only. Original fast32/slow32/8x8 memory/96-input AR/context and correction architecture unchanged. No claim of actual counterfactual decisions, native execution of learned CASM actions, fully emergent entities, verified System2 or symbolic reasoning from this forecasting-only task.

## Verified real data/exposure

Original ETDataset ETTh1.csv 17,420 real hourly measurements from one transformer station, seven numeric channels: six load measurements and measured oil temperature OT. Original URL https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv . Original bytes SHA256 f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066 from prior ME036 source ledger. Verify original byte/hash, header, 1-hour cadence and no missing values BEFORE training. Parse numerics ONLY from chronological FIT rows [0,8640) and previously inspected DEV rows [8640,11520). Do NOT parse original benchmark test [11520,14400) or later numeric values; those periods were previously explored by ME035/ME038, so this is NOT a newly sealed independent test. FIT-only seven-channel standardization. Ninety-six genuinely observed hourly history values predict actual subsequent 96h or 720h, all seven real channels; primary separate standardized error on observed oil-temperature channel OT, secondary seven-channel error. No future temperature or load provided at prediction time.

## Locked 88 real model fits

Seeds (42001,42002), horizons (96,720). Four real FIT-only original linear AR parent fits, 500 batch4 updates each, select from fixed historical DEV steps100/250/500. Each parent reused without replacement for the exact matched cohort. Eleven arms x two horizons x two seeds x two separately initialized lanes =88 real complete model fits.

Exposure lane: exactly 250 AdamW optimizer updates, true batch4 FIT histories and forecast targets, same randomly drawn exact row indexes per seed/horizon/arm, identical recorded loss forecast MSE +10 original AR anchor +.01 correction squared, common AR lr0.0001/others0.001, 1e-4 weight decay, clip1. No DEV-driven early stopping. CPU lane: train from the same AR source for approximately 5 measured optimizer CPU-seconds, finishing update crossing budget, max4000 updates; deliberately unequal real target presentations, record observed counts. Record parent pretraining time separately, and all optimizer CPU, end-to-end CPU and checkpoint replay.

All eight ME041 treatment models have exactly identical active parameter counts within this task; original references differ <1% total trainables. Matched samples and active parameters do NOT imply matched training FLOPs or CPU. Equal optimizer CPU means unequal training data. No false fully fair architecture superiority claims.

## Preregistered model performance selection

Use historical DEV and select by means across four horizon/seed combinations at exact exposure update250: primary oil-temperature standardized MSE, secondary all-seven-channel MSE. Rank all eleven, preserve per-horizon scores and per-seed errors, return registered CPU versus error Pareto frontier. Candidate accepted as historical engineering lead only if it beats original native AND original context mean oil MSE, stays within 1.05x original native at each horizon, and beats context mean oil MSE at equal five-second CPU. No candidate automatically promoted, merged, qualified against modern Transformers, deployed or claimed general. Preserve every negative arm, weights, optimizer/RNG and real predictions. Independent new-site validation remains separate from these previously exposed development cohorts.
