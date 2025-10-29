# Character generation LLM fine-tune pipeline

WIP: The final code and documentation is still under construction

Create LLM-based characters to chat with. Start with a `JSONL` database file of knowledge from a character and another one with conversation data (up to the user, but [this repo](https://github.com/DargorAbraxas/conversation_data_generation) can be used) to create a fine-tuned LLM.

Currently, there are 2 "big" options supported:
- Full fine-tuning
- Adapter tuning

## Full fine-tuning

Start from the base foundational model and use the character knowledge database to fine-tune it. This fine-tuned model, containing specific information about the character, is then used with the conversational dataset to create an instruct model. For this second step, it is possible to add a `system` prompt to the training data, and to choose between a full LLM loss (i.e., train over *all* the information in the conversation pairs, both user and assistant) or an `assistant-only` loss (i.e., train only over the assistant part of the conversation data pairs).

## Adapter fine-tuning

This version works by first creating a base adapter on top of the base foundational model instead of a full fine-tuning process. Then, there are two options:
- Place the base adapter as weighted merge on top of the *instruct* model and train a new adapter using the conversational data
- Train a fully new, independent adapter from the instruct model and then merge it with the base adapter to be placed on top of the instruct model.

## DPO

A DPO step will be added soon.

