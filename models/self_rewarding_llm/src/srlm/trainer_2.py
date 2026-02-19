from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
import os

class Trainer:
    def __init__(self, output):
        self.output = output

    def train(
            self,
            model,
            tokenizer,
            dataset
    ):
        learning_rate=2e-4
        batch_size = 1
        max_seq_length = 4096 # we can use a very long sequence length for now

        training_args = SFTConfig(
            output_dir=self.output,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            logging_steps=1,
            num_train_epochs=1,
            save_steps=50,
            max_seq_length=max_seq_length,
            dataset_text_field='text'
        )
        # NOTE: these dataset_text_fields and max_seq_legnths are present in the SFTConfig argument, we just use the default
        # values for now
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
