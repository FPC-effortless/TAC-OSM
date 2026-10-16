"""DEV-024: learning attribution within the fixed masked-local graph."""
from .entity_aging import EntityAgingControl

ARMS = ("joint", "frozen_writer", "frozen_encoder", "both_frozen", "schema41", "schema41_frozen")


def component_active(arm, prefix):
    if arm not in ARMS or prefix not in ("event_encoder.", "memory."):
        raise ValueError("unregistered freeze boundary")
    return (arm not in ("frozen_encoder", "both_frozen") if prefix == "event_encoder."
            else arm not in ("frozen_writer", "both_frozen", "schema41_frozen"))


class WriterAttributionControl(EntityAgingControl):
    def __init__(self, learning_arm, seed=0):
        if learning_arm not in ARMS:
            raise ValueError("unregistered learning intervention")
        super().__init__("schema41" if learning_arm.startswith("schema41") else "masked_local", seed)
        self.learning_arm = learning_arm
        for prefix, module in (("event_encoder.", self.event_encoder), ("memory.", self.memory)):
            if not component_active(learning_arm, prefix):
                module.requires_grad_(False)
