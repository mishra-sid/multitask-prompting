
def get_num_trainable_params(args, model):
    param_groups = []
    if args.tune_plm:
        param_groups.append(model.plm.parameters())
    
    if args.model_type == "prompt":
        param_groups.extend([p for name, p in model.template.named_parameters() if 'raw_embedding' not in name])
        param_groups.extend([model.verbalizer.group_parameters_2])

    return sum([sum([p.numel() for p in group if p.requires_grad]) for group in param_groups])

def get_uniq_str(args):
    return f"model={args.model}_plm_path={args.base_plm_path}_plm-frozen={args.tune_plm}_verbalizer_init={args.verbalizer_init}_prompt_text={args.prompt_text.replace(' ', '_')}"
