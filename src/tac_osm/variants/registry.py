"""Explicit isolation and historical source mapping for TAC-OSM research lanes.

Research intent is not a trained model. Never infer evidence from a class name.
All new trainable variants stay subject to the prospective 95k-105k policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

class Variant(str, Enum):
    SYSTEM1 = "system1"
    PHYSICAL = "physical"
    ENTERPRISE = "enterprise"
    TEMPORAL = "temporal"
    EDGE_LITE = "edge_lite"
    COUNTERFACTUAL = "counterfactual"

@dataclass(frozen=True)
class VariantSpec:
    identifier: Variant
    source_modules: tuple[str, ...]
    target: str
    readout: str
    current_evidence: str
    archived_studies: tuple[str, ...]
    deployment_state: str = "not_qualified"
    def __post_init__(self):
        if self.deployment_state == "production":
            raise ValueError("no production-qualified TAC-OSM variant in this registry")

VARIANTS: Mapping[Variant, VariantSpec] = {
    Variant.SYSTEM1: VariantSpec(
        Variant.SYSTEM1, ("tac_osm.real_robot_me061_reliable",),
        "single-observation risk/response from authentic live state",
        "one-step risk or proposed reaction; NEVER authorization",
        "ME074 fast-state readout sensitivity is not latency or policy evidence",
        ("ME054", "ME061", "ME074"),
    ),
    Variant.PHYSICAL: VariantSpec(
        Variant.PHYSICAL, ("tac_osm.real_robot_me061_reliable", "tac_osm.real_robot_me063_regime"),
        "observed joint/sensor trajectories conditioned on executed motor commands",
        "delta state, physical uncertainty and reliable servo/persistence fallback",
        "ME064/ME066 real transfer lost strongly to physical persistence",
        ("ME054", "ME056", "ME060", "ME061", "ME062", "ME063", "ME064", "ME066", "ME067", "ME068"),
    ),
    Variant.ENTERPRISE: VariantSpec(
        Variant.ENTERPRISE, ("tac_osm.workflow_me069", "tac_osm.workflow_me071",
                            "tac_osm.workflow_me072", "tac_osm.workflow_me073"),
        "real process future mark/time sequence from available observation prefixes",
        "event-conditional forecasts; published record ending is not verified business completion",
        "ME073 exposed-DEV native trajectory generation; early h1 at age2 weak",
        ("ME069", "ME070", "ME071", "ME072", "ME073", "ME074"),
    ),
    Variant.TEMPORAL: VariantSpec(
        Variant.TEMPORAL, ("tac_osm.woyengi_forecast_contract",
                           "tac_osm.variants.entity_sensor_world",
                           "tac_osm.variants.belief_sensor_world",
                           "tac_osm.variants.reconstruction_sensor_world",
                           "tac_osm.variants.median_split_sensor_world",
                           "tac_osm.variants.protected_sensor_world",
                           "tac_osm.variants.point_priority_sensor_world"),
        "persistent sensor histories; proposed as-known-at/as-valid-at corrections",
        "six-pollutant sensor distribution; separate proposed governance dual-clock input",
        "ME105 primary gap coverage passes15/15;4/15fullseedgates but noall5recipe qualifies; multichannel calibration and bitemporal correction remain unqualified",
        ("ME069", "ME071", "ME073", "ME100", "ME101", "ME102", "ME103", "ME104", "ME105", "ME106"),
    ),
    Variant.EDGE_LITE: VariantSpec(
        Variant.EDGE_LITE, ("tac_osm.parameter_budget",),
        "hardware-verified constrained deployment of frozen licensed forecasts",
        "resource/audit manifest; hardware latency must be measured",
        "no on-device INT8 original benchmark or qualified 15k model",
        ("ME055", "ME056", "ME060"),
    ),
    Variant.COUNTERFACTUAL: VariantSpec(
        Variant.COUNTERFACTUAL, ("tac_osm.casm_runtime", "tac_osm.workflow_me073"),
        "candidate-action conditional scenarios and policy uncertainty",
        "sandbox observational or simulated rollouts only until interventions qualify",
        "ME073 imagined futures are passive; Veritas synthetic tasks are diagnostics",
        ("DEV016", "DEV017", "DEV020", "ME049", "ME061", "ME073"),
    ),
}

def describe_variant(name: str | Variant) -> VariantSpec:
    try:
        return VARIANTS[Variant(name)]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"unknown TAC research variant: {name}") from exc
