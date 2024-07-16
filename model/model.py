from transformers import AutoConfig, AutoModelForSequenceClassification

def load_model(model_args):
    config=AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                    num_labels=3,
                                    finetuning_task='text-classification',
                                    #cache_dir=model_args.cache_dir)
    )                
    model=AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                            from_tf=bool('.ckpt' in model_args.model_name_or_path),
                                                            #num_labels=num_labels,
                                                            config=config,
                                                            #cache_dir=model_args.cache_dir)
    )
    return model

    