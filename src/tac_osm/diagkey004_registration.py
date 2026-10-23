"""Exact owner-authorized DIAGKEY004 exception; not a general synthetic lane."""
STUDY='DIAGKEY004_ALIGNED_GATED_ZERO_REPAIR_SYNTHETIC_ONLY'
COUNTS={'aligned_tac':101527,'tac':101554,'tac_noassoc':101560,'gru':101500,'transformer':101616}


def validate(manifest):
    def require(value,message):
        if not value:raise ValueError('DIAGKEY004: '+message)
    require(manifest.get('study_id')==STUDY,'unregistered synthetic experiment')
    for k in ('registered_before_first_fit','explicit_owner_requested_single_exception',
              'no_architecture_superiority_claim','no_real_world_capability_claim'):
        require(manifest.get(k) is True,k+' must be true')
    rows=manifest.get('models',[])
    require(len(rows)==25 and {(r.get('architecture'),r.get('seed')) for r in rows}==
            {(a,s) for a in COUNTS for s in range(75001,75006)},'fixed 25 arm/seed roster required')
    for row in rows:
        require(row.get('trainable_parameters')==row.get('deployed_parameters')==COUNTS[row['architecture']],
                'exact live whole-model size required')
    budget=manifest.get('training_budget',{})
    require(budget=={'steps':4096,'batch_size':16,'learning_rate':.001,'weight_decay':.0001,
                     'optimizer':'AdamW','clip_gradient_norm':1.,'trained_distractor_length':0},
            'fixed longer budget, original learning rate and zero distractors required')
    require(manifest.get('milestone_checkpoints')==[640,4096],'preserve both milestones')
    require(manifest.get('test_open_only_after_all_25_fits_frozen') is True,'freeze all originals before TEST')
    require(manifest.get('all_five_seed_threshold')==.9,'do not lower the original stopping gate')
    require(manifest.get('full_compute_parity_claim') is False,'no unsupported FLOP-matched winner claim')
    return rows
