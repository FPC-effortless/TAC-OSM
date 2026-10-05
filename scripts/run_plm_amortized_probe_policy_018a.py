    c=load_contract(EXPERIMENT_ID)
    if smoke: train_seeds=(0,); val_seeds=(20,); test_seeds=(25,); levels=(32,128); subsets=1; tasks=4
    else:
        train_seeds=TRAIN_SEEDS; val_seeds=VAL_SEEDS; test_seeds=TEST_SEEDS; levels=M_LEVELS; subsets=TRAIN_SUBSETS_PER_M; tasks=TASKS_PER_M
        c.require_levels(M_LEVELS); c.require_seeds(TEST_SEEDS); c.require_steps(1); c.require_eval_steps(TASKS_PER_M); c.require_arms(["greedy_budget_teacher","greedy_information_teacher","amortized_policy","fixed_random"])
    train_x,train_y,train_scores=build_training_examples(train_seeds,levels,subsets)
    val_x,val_y,val_scores=build_training_examples(val_seeds,levels,1)
    model,history=train_model(train_x,train_y,train_scores,val_x,val_y,val_scores)
    evaluated=evaluate(model,test_seeds,levels,tasks); summary=summarize(evaluated,test_seeds,levels)
    result={"experiment_id":EXPERIMENT_ID,"status":"measured","provenance":{"tacosm_commit":os.environ.get("GITHUB_SHA","local"),"run_id":os.environ.get("GITHUB_RUN_ID","local"),"generator_commit":GENERATOR_COMMIT,"python":sys.version,"torch":torch.__version__},"protocol":{"training_seeds":list(train_seeds),"validation_seeds":list(val_seeds),"test_seeds":list(test_seeds),"M_levels":list(levels),"B":BUDGET,"tasks_per_seed_M":tasks,"feature_dim":int(train_x.shape[-1]),"hidden_dim":HIDDEN,"epochs":EPOCHS,"learning_rate":LR,"weight_decay":WEIGHT_DECAY,"one_step_only":True},"split_integrity":{"train_eval_structure_disjoint":True,"train_eval_truth_disjoint":True,"test_teacher_outputs_generated_after_checkpoint_freeze":True,"test_teacher_outputs_used_for_training":False,"target_identity_used_by_policy":False,"realized_target_evidence_used_before_action":False},"training_history":history,"summary":summary,"results":evaluated,"scope":{"finite_domain_amortized_policy":True,"target_blind_action_selection":True,"learned_probe_policy_claim":True,"general_active_learning_claim":False,"submodularity_claim":False,"asymptotic_claim":False,"external_generalization_claim":False,"language_image_audio_claim":False,"hardware_speedup_claim":False},"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}
    Path("artifacts").mkdir(exist_ok=True); Path("artifacts",f"{EXPERIMENT_ID}.json").write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); main(p.parse_args().smokedef main(smoke=False):
    from tac_osm.contract import load_contract
    c=load_contract(EXPERIMENT_ID)
    if smoke:
        train_seeds=(0,); val_seeds=(20,); test_seeds=(25,); levels=(32,128); subsets=1; tasks=4
    else:
        train_seeds=TRAIN_SEEDS; val_seeds=VAL_SEEDS; test_seeds=TEST_SEEDS; levels=M_LEVELS; subsets=TRAIN_SUBSETS_PER_M; tasks=TASKS_PER_M
        c.require_levels(M_LEVELS); c.require_seeds(TEST_SEEDS); c.require_steps(1); c.require_eval_steps(TASKS_PER_M); c.require_arms(["greedy_budget_teacher","greedy_information_teacher","amortized_policy","fixed_random"])
    split_sets={}
    for split_name, seeds in (("train",train_seeds),("validation",val_seeds),("test",test_seeds)):
        all_structures=set(); all_truths=set()
        for seed in seeds:
            ss,tt=library_signatures(seed)
            if all_structures & ss or all_truths & tt:
                raise RuntimeError(f"cross-seed collision inside {split_name} population")
            all_structures |= ss; all_truths |= tt
        split_sets[split_name]=(all_structures,all_truths)
    if split_sets["train"][0] & split_sets["validation"][0] or split_sets["train"][1] & split_sets["validation"][1]:
        raise RuntimeError("train/validation exact population collision")
    if split_sets["train"][0] & split_sets["test"][0] or split_sets["train"][1] & split_sets["test"][1]:
        raise RuntimeError("train/test exact population collision")
    if split_sets["validation"][0] & split_sets["test"][0] or split_sets["validation"][1] & split_sets["test"][1]:
        raise RuntimeError("validation/test exact population collision")
    train_x,train_y,train_scores=build_training_examples(train_seeds,levels,subsets)
    val_x,val_y,val_scores=build_training_examples(val_seeds,levels,1)
    model,history=train_model(train_x,train_y,train_scores,val_x,val_y,val_scores)
    evaluated=evaluate(model,test_seeds,levels,tasks); summary=summarize(evaluated,test_seeds,levels)
    result={"experiment_id":EXPERIMENT_ID,"status":"measured","provenance":{"tacosm_commit":os.environ.get("GITHUB_SHA","local"),"run_id":os.environ.get("GITHUB_RUN_ID","local"),"generator_commit":GENERATOR_COMMIT,"python":sys.version,"torch":torch.__version__},"protocol":{"training_seeds":list(train_seeds),"validation_seeds":list(val_seeds),"test_seeds":list(test_seeds),"M_levels":list(levels),"B":BUDGET,"tasks_per_seed_M":tasks,"feature_dim":int(train_x.shape[-1]),"hidden_dim":HIDDEN,"epochs":EPOCHS,"learning_rate":LR,"weight_decay":WEIGHT_DECAY,"one_step_only":True},"split_integrity":{"train_eval_structure_disjoint":True,"train_eval_truth_disjoint":True,"test_teacher_outputs_generated_after_checkpoint_freeze":True,"test_teacher_outputs_used_for_training":False,"target_identity_used_by_policy":False,"realized_target_evidence_used_before_action":False},"training_history":history,"summary":summary,"results":evaluated,"scope":{"finite_domain_amortized_policy":True,"target_blind_action_selection":True,"learned_probe_policy_claim":True,"general_active_learning_claim":False,"submodularity_claim":False,"asymptotic_claim":False,"external_generalization_claim":False,"language_image_audio_claim":False,"hardware_speedup_claim":False},"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}
    Path("artifacts").mkdir(exist_ok=True); Path("artifacts",f"{EXPERIMENT_ID}.json").write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); main(p.parse_args().smoke)