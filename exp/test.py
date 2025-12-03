from datasets import load_dataset
from trl import GRPOTrainer


# Define a simple reward function (count unique chars as example)
def reward_function(completions, **kwargs):
    return [len(set(completion.lower())) for completion in completions]


trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=load_dataset("trl-lib/tldr", split="train"),
    reward_funcs=reward_function,
)
trainer.train()
