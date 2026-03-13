from datasets import load_dataset

prompts = []

ds = load_dataset("allenai/real-toxicity-prompts", split="train")

for example in ds:

    prompt_text = example

    if prompt_text and len(prompt_text) > 10:
        prompts.append(prompt_text)

prompts = list(set(prompts))

print("Total prompts:", len(prompts))

with open("adversarial_prompts.txt", "w", encoding="utf-8") as f:
    for p in prompts:
        f.write(p.replace("\n", " ") + "\n")

print("File created successfully")