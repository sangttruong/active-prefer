import re

def extract_dialogues(dialogue_string):
    human_dialogues = re.findall(r'Human: (.+?)(?=(?:Assistant:|Human:)|$)', dialogue_string)
    assistant_dialogues = re.findall(r'Assistant: (.+?)(?=(?:Assistant:|Human:)|$)', dialogue_string)
    return human_dialogues, assistant_dialogues

def join_dialogues(human_dialogues, assistant_dialogues):
    joined_dialogue = f"\nAssistant: {assistant_dialogues[0]}\n"
    dialogue_pairs = zip(human_dialogues, assistant_dialogues[1:])
    for human, assistant in dialogue_pairs:
        joined_dialogue += f"Human: {human}\nAssistant: {assistant}\n"
    return joined_dialogue


dialogue_string = "Human: What kind of noises did dinosaurs make? Assistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made would be Human: yes they did Assistant: to guess, and that would probably require lots of reading and a certain amount of imagination, so we’re not really prepared to do that. Human: you cant read Assistant: You can read?"

human_dialogues, assistant_dialogues = extract_dialogues(dialogue_string)

print(f"Human Dialogues have {len(human_dialogues)}: ")
for dialogue in human_dialogues:
    print(dialogue)

print(f"\nAssistant Dialogues have {len(assistant_dialogues)}:")
for dialogue in assistant_dialogues:
    print(dialogue)

# Example usage:
print("Dialogues:===============")
joined_dialogue = join_dialogues(human_dialogues[1:], assistant_dialogues)
print(joined_dialogue)
