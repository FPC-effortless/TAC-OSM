"""Preserve actual optimizer effects even when the accounting exit fails."""

def preserve(C, output, model, optimizer, metadata, journal, completed, attempted, elapsed, error):
    steps = {str(key): int(slot['step']) for key, slot in optimizer.state_dict()['state'].items()
             if 'step' in slot}
    meta = {**metadata, 'stage': 'FAILED_RETAINED_NOT_EVALUABLE',
            'model_sha256': C.state_digest(model), 'optimizer_steps': completed,
            'optimizer_step_attempts': attempted, 'completed_ledger_entries': len(journal),
            'optimizer_slot_steps': steps}
    checkpoint_sha = C.checkpoint(output/'FAILED_MODEL.pt', model, optimizer, meta)
    C.save(output/'FAILURE.json', {
        'exception': type(error).__name__, 'message': str(error),
        'actual_steps': completed, 'optimizer_steps': completed,
        'optimizer_step_attempts': attempted, 'completed_ledger_entries': len(journal),
        'optimizer_slot_steps': steps, 'elapsed': elapsed,
        'failed_checkpoint_sha256': checkpoint_sha,
        'status': 'FAILED_RETAINED_NOT_EVALUABLE',
        'partial_optimizer_step_possible': attempted != completed})
