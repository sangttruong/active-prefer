from src.llmtuner import run_exp


def select_active_queries(dataset, question_id):
    pass

def main():
    stage = 'sft' 
    do_train = True
    model_name_or_path = "meta-llama/Llama-2-7b-hf" 
    dataset =  "alpaca_gpt4_en"
    dataset_dir  = "data"
    template = "default"
    finetuning_type  = "lora" 
    lora_target = "q_proj,v_proj" 
    output_dir = f"saves/LLaMA2-7B/lora/{stage}" 
    cutoff_len  = 1024 
    per_device_train_batch_size = 1 
    per_device_eval_batch_size = 1 
    gradient_accumulation_steps = 8 
    lr_scheduler_type = "cosine" 
    logging_steps = 10 
    save_steps = 100 
    eval_steps = 100 
    evaluation_strategy = "steps" 
    load_best_model_at_end = True
    learning_rate = 5e-5 
    num_train_epochs  = 1.0 
    max_samples = 1000 
    val_size = 0.1 
    quantization_bit = 4 
    plot_loss = True
    fp16 = True

    # Inference
    run_exp(dict(
        stage = 'rm',
        do_train  = True,
        do_predict = True,
        model_name_or_path = model_name_or_path,
        overwrite_output_dir = True,
        dataset = dataset,
        dataset_dir  = dataset_dir,
        template = template,
        finetuning_type  = finetuning_type,
        lora_target = lora_target,
        output_dir = output_dir,
        cutoff_len  = cutoff_len,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        lr_scheduler_type = lr_scheduler_type,
        logging_steps = logging_steps,
        save_steps = save_steps,
        eval_steps = eval_steps,
        evaluation_strategy = evaluation_strategy,
        learning_rate = learning_rate,
        num_train_epochs  = num_train_epochs,
        max_samples = max_samples, # setup small step for warm up
        val_size = val_size,
        quantization_bit = quantization_bit,
        plot_loss = plot_loss,
        fp16 = fp16
    ))



    # Selection
    
    # Train DPO
    
    # Train Reward

    


    


if __name__ == "__main__":
    main()

