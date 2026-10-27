# MODEL01-UCI235-TRANSFER-001: frozen cross-source robustness stress

**Registration before opening UCI235 original measurements in this workflow.** Original different household: [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), DOI 10.24432/C58K54, authors Georges Hebrail and Alice Berard, CC BY 4.0. Original publisher compressed ZIP endpoint `https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip`.

## Frozen evaluation, source and hypothesis

- Use original **unchanged** UCI374 trained six-feature classical ridge checkpoint and its original UCI374 calibration half-width. No FIT, recalibration, selection, normalization changes, gradient updates or parameter tuning on UCI235; exact Git checkpoint digest stays enforced.
- Read only source minutes from 2006-12-17 00:00 local naive timestamp to before 2007-01-16 00:00. The publisher compressed ZIP may contain later data, but never parse source member rows after this selected period. The selected evaluation dates are 2007-01-01 inclusive to 2007-01-16 exclusive, allowing earlier history.
- Original UCI235 `Global_active_power` in kilowatts is converted to **whole-house** watt-hours for each 10-minute period by summing ten one-minute mean active power observations multiplied by 1000/60. Reject missing or invalid minute readings: no imputation, and discard any prediction case with an incomplete current/past 145-block history or incomplete 60-minute-ahead target block.
- Frozen candidate is original ridge using its six original features, with source index measured from the UCI235 first selected 10-minute observation. This original daily phase index is **not** aligned to original UCI374 source origin, so source-clock mapping is an additional shift; no post-hoc correction in this test.
- Controls: unmodified latest observed energy (persistence), same time previous day. No control fitted on transfer source. Horizon: six ten-minute periods (60min). Primary reported point metric: MAE Wh (whole house); nominal 90% interval empirical coverage of original frozen CAL half-width is a transfer warning, not a calibrated guarantee. Report each separate calendar day's MAE, source missingness and exact original ZIP hash.
- A 300-valid-window minimum and at least seven source evaluation calendar days are necessary to report the stress. Fail closed otherwise. No significance testing on overlapping cases is implied.

## Pre-run claim constraints

This is a **different prediction target** from the UCI374 `Appliances` Wh per ten minutes: UCI235 whole-house active energy includes other loads. The site, meter type, weather, dates, timezone semantics and phase origin differ. Therefore even a positive numerical result **cannot establish independent same-task confirmation**, TAC architecture superiority, production forecasting quality or customer generalization. It is an explicitly exploratory OOD transfer stress that can reveal failure and target mismatch. An independent same-task target/feature contract needs actual compatible customer/site measurements and frozen audit.

UCI235 minute timestamps are source-local naive without independently established publisher real-time available-at lag; no claim of prospective operational prediction is allowed. Once read, this source/date becomes exposed development evidence for future work.

Implementation `specialist_models/model01_uci235_transfer_stress.py`. Unit tests contain constructed archives only. CI fetches the original public source under a 30MB compressed / 150MB member bound and writes only aggregate metrics and original source hashes; no original raw electricity records are committed.
