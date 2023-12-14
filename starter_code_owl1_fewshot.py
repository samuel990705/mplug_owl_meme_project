import os
import json
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor


args = argparse.ArgumentParser()
args.add_argument("--prompt", type=int, default=1)
args.add_argument("--shot", type=int, default=0)
args = args.parse_args()


pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
finetuned_checkpoint = 'output/test_finetune/checkpoint-500'
model = MplugOwlForConditionalGeneration.from_pretrained(
    finetuned_checkpoint,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True
    )
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Load model successfully!")

image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
print("Load image processor successfully!")
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
print("Load tokenizer successfully!")
processor = MplugOwlProcessor(image_processor, tokenizer)


text = ["THERE ARE ONLY TWO GENDERS\n\nFEMALE \n\nMALE\n", "This is not an accident!", "TILL DEATH DO US PART\n\nNO TO VIOLENCE AGAINST WOMEN\n"]

image_list = ['data/train/128_image.png', 'data/train/162_image.png', 'data/train/189_image.png']

# generate kwargs (the same in transformers) can be passed in the do_generate()
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}

with open('data/test/test_set_task3.txt', "r") as f:
    gt = json.load(f)
data = {example['id'] : example for example in gt}

generated_text = []
print(f"Start generating with prompt {args.prompt} and {args.shot} shot(s)...")
memes_idx=[]
for i,key in enumerate(data.keys()):
    print("Generating for meme "+str(i)+":", key)
    dev_images = image_list + ['data/test/'+data[key]['image']]
    # The first three examples are used for one-shot and three-shot learning
    dev_text = text + [data[key]['text']]
    memes_idx.append(key)
    # prompt1
    if args.prompt == 1:
        if args.shot == 0:
            # zero shot
            prompts = [
                f'''The following is a conversation between a curious human and AI assistant. The assistant identify persuasion techniques in image and text given by the human. 
                List of techniques: ['Appeal to authority', 'Appeal to fear/prejudice', 'Black-and-white Fallacy/Dictatorship', 'Causal Oversimplification', 'Doubt', 'Exaggeration/Minimisation', 'Flag-waving', 
                'Glittering generalities (Virtue)', 'Loaded Language', "Misrepresentation of Someone's Position (Straw Man)", 'Name calling/Labeling', 'Obfuscation, Intentional vagueness, Confusion', 'Presenting Irrelevant Data (Red Herring)', 
                'Reductio ad hitlerum', 'Repetition', 'Slogans', 'Smears', 'Thought-terminating cliché', 'Whataboutism', 'Bandwagon', 'Transfer', 'Appeal to (Strong) Emotions']
                Human: Image: <image>. Text: {dev_text[3]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI:'''
            ]
        elif args.shot == 1:
            # one shot
            prompts = [
                f'''The following is a conversation between a curious human and AI assistant. The assistant identify persuasion techniques in image and text given by the human. 
                List of techniques: ['Appeal to authority', 'Appeal to fear/prejudice', 'Black-and-white Fallacy/Dictatorship', 'Causal Oversimplification', 'Doubt', 'Exaggeration/Minimisation', 'Flag-waving', 
                'Glittering generalities (Virtue)', 'Loaded Language', "Misrepresentation of Someone's Position (Straw Man)", 'Name calling/Labeling', 'Obfuscation, Intentional vagueness, Confusion', 'Presenting Irrelevant Data (Red Herring)', 
                'Reductio ad hitlerum', 'Repetition', 'Slogans', 'Smears', 'Thought-terminating cliché', 'Whataboutism', 'Bandwagon', 'Transfer', 'Appeal to (Strong) Emotions']
                Human: Image: <image>. Text: {dev_text[1]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Appeal to (Strong) Emotions", "Appeal to fear/prejudice", "Loaded Language", "Slogans"]
                Human: Image: <image>. Text: {dev_text[3]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI:'''
            ]
        elif args.shot == 3:
            # three shot
            prompts = [
                f'''The following is a conversation between a curious human and AI assistant. The assistant identify persuasion techniques in image and text given by the human. 
                List of techniques: ['Appeal to authority', 'Appeal to fear/prejudice', 'Black-and-white Fallacy/Dictatorship', 'Causal Oversimplification', 'Doubt', 'Exaggeration/Minimisation', 'Flag-waving', 
                'Glittering generalities (Virtue)', 'Loaded Language', "Misrepresentation of Someone's Position (Straw Man)", 'Name calling/Labeling', 'Obfuscation, Intentional vagueness, Confusion', 'Presenting Irrelevant Data (Red Herring)', 
                'Reductio ad hitlerum', 'Repetition', 'Slogans', 'Smears', 'Thought-terminating cliché', 'Whataboutism', 'Bandwagon', 'Transfer', 'Appeal to (Strong) Emotions']
                Human: Image: <image>. Text: {dev_text[0]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Black-and-white Fallacy/Dictatorship", "Name calling/Labeling", "Smears"]
                Human: Image: <image>. Text: {dev_text[1]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Appeal to (Strong) Emotions", "Appeal to fear/prejudice", "Loaded Language", "Slogans"]
                Human: Image: <image>. Text: {dev_text[2]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Causal Oversimplification"]
                Human: Image: <image>. Text: {dev_text[3]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI:'''
            ]
        else:
            raise ValueError("Invalid shot number!")
    # prompt2
    elif args.prompt == 2:
        if args.shot == 0:
            # zero shot
            prompts = [
                f'''The following scenario involves a conversation between a curious human and an AI assistant. The AI is tasked with identifying persuasion techniques in both an image and accompanying text provided by the human. The human has presented a list of possible techniques, and the AI is expected to analyze the content and select relevant techniques from the list. Here is the list of techniques for reference:
                Appeal to authority
                Appeal to fear/prejudice
                Black-and-white Fallacy/Dictatorship
                Causal Oversimplification
                Doubt
                Exaggeration/Minimisation
                Flag-waving
                Glittering generalities (Virtue)
                Loaded Language
                Misrepresentation of Someone's Position (Straw Man)
                Name calling/Labeling
                Obfuscation, Intentional vagueness, Confusion
                Presenting Irrelevant Data (Red Herring)
                Reductio ad hitlerum
                Repetition
                Slogans
                Smears
                Thought-terminating cliché
                Whataboutism
                Bandwagon
                Transfer
                Appeal to (Strong) Emotions
                Human: Image: <image>. Text: {dev_text[3]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: '''
            ]
        elif args.shot == 1:
            # one shot
            prompts = [
                f'''The following scenario involves a conversation between a curious human and an AI assistant. The AI is tasked with identifying persuasion techniques in both an image and accompanying text provided by the human. The human has presented a list of possible techniques, and the AI is expected to analyze the content and select relevant techniques from the list. Here is the list of techniques for reference:
                Appeal to authority
                Appeal to fear/prejudice
                Black-and-white Fallacy/Dictatorship
                Causal Oversimplification
                Doubt
                Exaggeration/Minimisation
                Flag-waving
                Glittering generalities (Virtue)
                Loaded Language
                Misrepresentation of Someone's Position (Straw Man)
                Name calling/Labeling
                Obfuscation, Intentional vagueness, Confusion
                Presenting Irrelevant Data (Red Herring)
                Reductio ad hitlerum
                Repetition
                Slogans
                Smears
                Thought-terminating cliché
                Whataboutism
                Bandwagon
                Transfer
                Appeal to (Strong) Emotions
                Human: Image: <image>. Text: {dev_text[1]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Appeal to (Strong) Emotions", "Appeal to fear/prejudice", "Loaded Language", "Slogans"]
                Human: Image: <image>. Text: {dev_text[3]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: '''
            ]
        elif args.shot == 3:
            # three shot
            prompts = [
                f'''The following scenario involves a conversation between a curious human and an AI assistant. The AI is tasked with identifying persuasion techniques in both an image and accompanying text provided by the human. The human has presented a list of possible techniques, and the AI is expected to analyze the content and select relevant techniques from the list. Here is the list of techniques for reference:
                Appeal to authority
                Appeal to fear/prejudice
                Black-and-white Fallacy/Dictatorship
                Causal Oversimplification
                Doubt
                Exaggeration/Minimisation
                Flag-waving
                Glittering generalities (Virtue)
                Loaded Language
                Misrepresentation of Someone's Position (Straw Man)
                Name calling/Labeling
                Obfuscation, Intentional vagueness, Confusion
                Presenting Irrelevant Data (Red Herring)
                Reductio ad hitlerum
                Repetition
                Slogans
                Smears
                Thought-terminating cliché
                Whataboutism
                Bandwagon
                Transfer
                Appeal to (Strong) Emotions
                Human: Image: <image>. Text: {dev_text[0]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Black-and-white Fallacy/Dictatorship", "Name calling/Labeling", "Smears"]
                Human: Image: <image>. Text: {dev_text[1]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Appeal to (Strong) Emotions", "Appeal to fear/prejudice", "Loaded Language", "Slogans"]
                Human: Image: <image>. Text: {dev_text[2]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: ["Causal Oversimplification"]
                Human: Image: <image>. Text: {dev_text[3]}.
                Human: Please select the techniques used in the image and the text from the list above.
                AI: '''
            ]
        else:
            raise ValueError("Invalid shot number!")
    else:
        raise ValueError("Invalid prompt number!")

    images = [Image.open(_) for _ in dev_images]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    print(sentence)
    generated_text.append(sentence)
    print("=====================================")

with open('results/no_grounding'+'_prompt'+str(args.prompt)+'_shot'+str(args.shot)+'.txt', 'w') as f:
    for i in range(len(generated_text)):
        f.write("Generating for meme:"+memes_idx[i]+'\n')
        f.write(generated_text[i]+'\n')
        f.write('=====================\n')