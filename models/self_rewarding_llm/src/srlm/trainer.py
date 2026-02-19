from transformers import TrainerCallback, TrainingArguments
import torch.nn as nn
import os
from trl import SFTTrainer, SFTConfig

class Trainer:
    def __init__(self, output):
        self.output = output

    def train(
        self,
        model,
        tokenizer,
        dataset
    ):
        batch_size = 1
        effective_batch_size = 16
        max_seq_length = 4096

        training_args = SFTConfig(
            output_dir=self.output,
            per_device_train_batch_size=batch_size,
            learning_rate=5.5e-6,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            gradient_accumulation_steps=effective_batch_size // batch_size,
            warmup_steps=30,
            logging_steps=1,
            save_steps=1000,
            max_seq_length=max_seq_length,
            dataset_text_field='text',
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
        )

        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        trainer.train()

        output_dir = os.path.join(self.output, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)