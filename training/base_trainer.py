import os
import torch
import locale
from peft import (
    get_peft_model,
    LoraConfig,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from datetime import datetime
from accelerate import Accelerator
from abc import ABC, abstractmethod

from customlogger.logger import Logger
from utils.load_config import Config
from utils.data_utils import DatasetLoader

locale.getpreferredencoding = lambda: "UTF-8"

logger = Logger(__name__)


class BaseTrainer(ABC):
    def __init__(self, config_path):
        self.config = Config(config_path)
    def get_trainer(self):
        if self.config.use_4bit_bnb:
            model = self.get_model(self.config.BASE_MODEL, use_4bit_bnb=True)
        else:
            model = self.get_model(self.config.BASE_MODEL)

        tokenizer = self.get_tokenizer(self.config.BASE_MODEL)

        peft_config = None
        if self.config.USE_LORA:
            model, peft_config = self.peft_model(model)

        output_dir = self._make_run_dir()

        if self.config.USE_WANDB:
            training_arguments = TrainingArguments(
                output_dir=output_dir,
                learning_rate=self.config.LEARNING_RATE,
                num_train_epochs=self.config.NUM_EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
                gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
                optim=self.config.OPTIMIZER,
                weight_decay=self.config.WEIGHT_DECAY,
                max_grad_norm=self.config.MAX_GRAD_NORM,
                bf16=self.config.use_bf16,
                warmup_ratio=self.config.WARMUP_RATIO,
                lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
                save_strategy=self.config.SAVE_STRATEGY,
                save_steps=self.config.SAVE_STEPS,
                # load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
                # evaluation_strategy=self.config.SAVE_STRATEGY,
                # eval_steps=self.config.EVAL_STEPS,
                dataloader_pin_memory=True,
                dataloader_num_workers=4,
                logging_steps=self.config.LOGGING_STEPS,
                report_to=self.config.REPORT_TO
            )
        else:
            training_arguments = TrainingArguments(
                output_dir=output_dir,
                learning_rate=self.config.LEARNING_RATE,
                num_train_epochs=self.config.NUM_EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                gradient_accumulation_steps=self.config.GRAD_ACCUMULATION_STEPS,
                gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
                optim=self.config.OPTIMIZER,
                weight_decay=self.config.WEIGHT_DECAY,
                max_grad_norm=self.config.MAX_GRAD_NORM,
                bf16=self.config.use_bf16,
                warmup_ratio=self.config.WARMUP_RATIO,
                lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
                save_strategy=self.config.SAVE_STRATEGY,
                save_steps=self.config.SAVE_STEPS,
                # load_best_model_at_end=self.config.LOAD_BEST_MODEL_AT_END,
                # evaluation_strategy=self.config.SAVE_STRATEGY,
                # eval_steps=self.config.EVAL_STEPS,
                dataloader_pin_memory=True,
                dataloader_num_workers=4,
                logging_steps=self.config.LOGGING_STEPS
            )

        train_dataset, test_dataset = DatasetLoader(self.config.DATASET_PATH, self.config.USE_HF, tokenizer).get_dataset()
        logger.debug(train_dataset[0])
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            peft_config=peft_config,

            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="text",

            args=training_arguments,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            packing=self.config.PACKING,
            neftune_noise_alpha=5.0
        )

        self._calculate_steps(train_dataset)

        return trainer, self.config.NEW_MODEL

    def get_bnb_config(self):
        """
        Get the BitsAndBytesConfig

        :return: BiteAndBytesConfig
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return bnb_config

    def get_model(self, model: str, use_4bit_bnb: bool = False) -> AutoModelForCausalLM:
        """
        Get the model

        :return: AutoModelForCausalLM
        """
        if use_4bit_bnb:
            model = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=self.get_bnb_config(),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        self._model_loading_postprocessing(model)
        self._model_loading_checks(model)
        self._set_more_layers_trainable(model)

        return model

    def get_tokenizer(self, model: str) -> AutoTokenizer:
        """
        Get the tokenizer

        :return: AutoTokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            trust_remote=True
        )
        self._set_padding_token(tokenizer)
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        return tokenizer

    def peft_model(self, model):
        peft_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias=self.config.LORA_BIAS,
            task_type=self.config.LORA_TASK_TYPE,
            use_dora=self.config.USE_DORA
        )
        model = get_peft_model(model, peft_config)
        self._get_model_trainable_parameters(model)
        return model, peft_config

    def _model_loading_checks(self, model):
        """
        Perform checks on the model

        :param model: AutoModelForCausalLM
        """
        for n, p in model.named_parameters():
            if p.device.type == "meta":
                logger.error(f'{n} is on meta')

    def _model_loading_postprocessing(self, model):
        """
        Perform postprocessing on the model
        :param model:
        :return:
        """
        for params in model.parameters():
            params.requires_grad = False
            if params.ndim == 1:
                params.data = params.data.to(torch.float32)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        logger.info(f"Model loaded - {model}")

    def _get_model_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        non_trainable_params = 0
        all_param = 0

        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()
        logger.info(
            f"Trainable params: {trainable_params} || All params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _set_padding_token(self, tokenizer):
        if '<pad>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<pad>'
        elif '<|pad|>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<|pad|>'
        elif '<unk>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<unk>'
        else:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

    def _set_more_layers_trainable(self, model):
        """
        Set more layers to be trainable

        :param model: AutoModelForCausalLM
        """
        trainable_layers = ['embed_tokens', 'input_layernorm', 'post_attention_layernorm']
        for n, p in model.named_parameters():
            if any(k in n for k in trainable_layers):
                p.requires_grad_(True)
        logger.info(f"More layers set to trainable - {trainable_layers}")

    def _make_run_dir(self):
        """
        Using DateTime to create a directory to store the run logs
        :return:
        """
        dirname = f"checkpoints/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(dirname)
        return dirname

    def _calculate_steps(self, train_dataset):
        dataset_size = len(train_dataset)
        steps_per_epoch = dataset_size / (self.config.BATCH_SIZE * self.config.GRAD_ACCUMULATION_STEPS)
        total_steps = steps_per_epoch * self.config.NUM_EPOCHS

        logger.info(f"Total number of steps: {total_steps}")


