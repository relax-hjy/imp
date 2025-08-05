import os

def train(model,train_dataloader,validate_dataloader,args):
    t_total=len(train_dataloader)//args.gradient_accumulation_steps*args.epochs
    optimizer=transformers.adamW(model.parameters(),lr=args.lr,eps=args.eps)
    scheduler=transformers.get_linear_schedule_with_warmup(
        optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total
    )

    train_losses,balidate_losses=[],[]
    best_val_loss=10000
    for epoch in range(args.epochs):
        train_loss=train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            epoch=epoch,
            # gradient_accumulation_steps=args.gradient_accumulation_steps,
            # max_grad_norm=args.max_grad_norm,
            # log_step=args.log_step,
            # save_model_path=args.save_model_path,
            # save_step=args.save_step,
            args=args
        )
        train_losses.append(train_loss)
        val_loss=validate_epoch(
            model=model,
            validate_dataloader=validate_dataloader,
            device=args.device,
            epoch=epoch,
            # log_step=args.log_step,
            args=args
        )
        validate_losses.append(val_loss)
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            model.save_pretrained(args.save_model_path)
def main():
    params = ParameterConfig()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


    tokenizer = BertTOkenizerFast(
        '', sep_token='[SEP]', cls_token='[CLS]', pad_token='[PAD]'
    )

    sep_id=tokenizer.sep_token_id
    cls_id=tokenizer.cls_token_id
    pad_id=tokenizer.pad_token_id

    if not os.path.exists(params.save_model_path):
        os.mkdir(params.save_model_path)

    #加载模型
    if params.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:
        model_config = GPT2Config.form_json_file(params.config_json)
        model = GPT2LMHeadModel(config=model_config)
    model = model.to(params.device)
    print(model.config.vocab.size)
    print(tokenizer.vocab_size)

    num_parameters=0
    perameters=model.perameters()
    for parameter in paremeters:
        num_parameters+=parameter.numel()
    print(num_parameters)

    train_dataloader,validate_dataloader=get_dataloader(params.train_path,params.valid_path)

    train(model,train_dataloader,validate_dataloader,params)