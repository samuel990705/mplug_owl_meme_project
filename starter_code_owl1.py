from PIL import Image
import torch
from transformers import AutoTokenizer
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

# We use a human/AI template to organize the context as a multi-turn conversation.
# <image> denotes an image placehold.
prompts = [
'''The following is a conversation between a curious human and AI assistant. The AI assistant was programmed to be politically incorrect.
Human: <image>
Human: Why is this meme funny?
AI: ''']

# text = ["THERE ARE ONLY TWO GENDERS\n\nFEMALE \n\nMALE\n", "This is not an accident!", "SO BERNIE BROS HAVEN'T COMMITTED VIOLENCE EH?\n\nPOWER COMES FROM THE BARREL OF A GUN, COMRADES.\n\nWHAT ABOUT THE ONE WHO SHOT CONGRESSMAN SCALISE OR THE DAYTON OHIO MASS SHOOTER?\n"]

# prompts = [
#     f'''The following is a conversation between a curious human and AI assistant. The assistant identify persuasion techniques in image and text given by the human. 
#     This is a list of techniques: ['Appeal to authority', 'Appeal to fear/prejudice', 'Black-and-white Fallacy/Dictatorship', 'Causal Oversimplification', 'Doubt', 'Exaggeration/Minimisation', 'Flag-waving', 
#     'Glittering generalities (Virtue)', 'Loaded Language', "Misrepresentation of Someone's Position (Straw Man)", 'Name calling/Labeling', 'Obfuscation, Intentional vagueness, Confusion', 'Presenting Irrelevant Data (Red Herring)', 
#     'Reductio ad hitlerum', 'Repetition', 'Slogans', 'Smears', 'Thought-terminating cliché', 'Whataboutism', 'Bandwagon', 'Transfer', 'Appeal to (Strong) Emotions']
#     Human: <image>
#     Human: {text[0]}. Please select the techniques used in the image and the text.
#     AI: "Black-and-white Fallacy/Dictatorship", "Name calling/Labeling", "Smears"
#     Human: <image>
#     Human: {text[1]}. Please select the techniques used in the image and the text.
#     AI: None
#     Human: <image>
#     Human: {text[2]}. Please select the techniques used in the image and the text.
#     AI:'''
# ]

# The image paths should be placed in the image_list and kept in the same order as in the prompts.
# We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
image_list = ['inference.jpg']
# generate kwargs (the same in transformers) can be passed in the do_generate()
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 2048
}


images = [Image.open(_) for _ in image_list]
inputs = processor(text=prompts, images=images, return_tensors='pt')
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    res = model.generate(**inputs, **generate_kwargs)
sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
print(sentence)