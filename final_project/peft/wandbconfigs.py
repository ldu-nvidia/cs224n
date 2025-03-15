from peft import LoraConfig, TaskType, LoHaConfig, LoKrConfig, AdaLoraConfig

def get_all_lora_configs(model_name):  
    # target module name is different for different models  
    if model_name == 'BERT':
        modules = ['attention.self.query', 
        'attention.self.key', 
        'attention.self.value']
    elif model_name == 'DistilBERT':
        modules = ["attention.q_lin", 
        "attention.k_lin", 
        "attention.v_lin"]
    else:
        modules = ["attention.self.query", 
        "attention.self.key",    
        "attention.self.value"]
    
    lora_config = LoraConfig(
    # sweeping parameter: LORA rank
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias='none',
    target_modules=modules,
    task_type="SEQ_CLS"
    )

    loha_config = LoHaConfig(
    r=16,
    alpha=16,
    target_modules=modules,
    module_dropout=0.1,
    modules_to_save=["classifier"],)

    lokr_config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=modules,
    module_dropout=0.1,
    modules_to_save=["classifier"],)

    adalora_config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=modules,
    modules_to_save=["classifier"],)

    return (lora_config, loha_config, lokr_config, adalora_config, None)