import os
import json
from PIL import Image
import torch
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
finetuned_checkpoint = 'output/test_finetune/checkpoint-500'
model = MplugOwlForConditionalGeneration.from_pretrained(
    finetuned_checkpoint,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True
    )
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Load model successfully!")
# model = model.to(device)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
print("Load image processor successfully!")
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
print("Load tokenizer successfully!")
processor = MplugOwlProcessor(image_processor, tokenizer)

# exit()
# We use a human/AI template to organize the context as a multi-turn conversation.
# <image> denotes an image placehold.
# prompts = [
# '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: <image>
# Human: Explain why this meme is funny.
# AI: ''']

text = ["THERE ARE ONLY TWO GENDERS\n\nFEMALE \n\nMALE\n", "This is not an accident!", "TILL DEATH DO US PART\n\nNO TO VIOLENCE AGAINST WOMEN\n"]

# The image paths should be placed in the image_list and kept in the same order as in the prompts.
# We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
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
memes_idx=[]
print("Start generating...")
for key in data.keys():
    print("Generating for meme:", key)
    memes_idx.append(key)
    dev_images = image_list + ['data/test/'+data[key]['image']]
    dev_text = text + [data[key]['text']]
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
    # dev_images = ['data/dev/'+data[key]['image']]
    # dev_text = [data[key]['text']]
    # prompts = [
    #     f'''The following is a conversation between a curious human and AI assistant. The assistant identify persuasion techniques in image and text given by the human. 
    #     Human: Image: <image>. Text: {dev_text[0]}.
    #     Human: Describe how the image and the text use persuation techniques.
    #     AI: '''
    # ]

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

with open('results/test.txt', 'w') as f:
    for i in range(len(generated_text)):
        f.write("Generating for meme:"+memes_idx[i]+'\n')
        f.write(generated_text[i]+'\n')
        f.write('=====================\n')