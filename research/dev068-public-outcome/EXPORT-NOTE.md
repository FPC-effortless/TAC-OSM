# Export dependency correction

The initial smoke export used `torch.use_deterministic_algorithms(True)`. In PyTorch2.14 this imports its optional compiler stack, which imports NumPy; the isolated no-research-dependency check rejected it. The failure log is retained.

The export now uses the supported public `torch.set_deterministic_debug_mode("error")` eager interface and asserts that deterministic algorithms are enabled. The same strict eager kernel flag is set without importing the optional compiler. The import guard, tolerance, model source, weights, fitting source and registered criteria were unchanged. Eight smoke checkpoints, 16 clock cases, eight interface cases, 124 native actions and 31 standalone decisions replay successfully with zero numeric drift. Smoke is not performance evidence.
