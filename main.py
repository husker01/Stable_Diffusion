import os
import torch
torch.cuda.empty_cache()
from torch import autocast
from diffusers import StableDiffusionPipeline
from prompt_engineering import art_styles
torch.cuda.memory_summary(device=None, abbreviated=False)


class ImageGenerator:
    def __init__(self):
        self.SDV5_MODEL_PATH = os.getenv("SDV5_MODEL_PATH")
        self.SAVE_PATH = os.path.join(os.environ["USERPROFILE"], 'Desktop', 'SD_OUTPUT')

        if not os.path.exists(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)

    @staticmethod
    def uniquify(path):
        filename, extension = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = filename + '(' + str(counter) + ')' + extension
            counter += 1
        return path

    def render_prompt(self):
        shorted_prompt = (prompt[:25] + '...') if len(prompt) > 25 else prompt
        shorted_prompt = shorted_prompt.replace(' ', '_')
        generation_path = os.path.join(self.SAVE_PATH, shorted_prompt.removesuffix('...'))

        if not os.path.exists(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)
        if not os.path.exists(generation_path):
            os.mkdir(generation_path)

        #call stable diffusion pipeline
        if device_type == 'cuda':
            if low_vram:
                pipe = StableDiffusionPipeline.from_pretrained(
                    self.SDV5_MODEL_PATH,
                    torch_dtype=torch.float16,
                    revision='fp16'
                )

            else:
                pipe = StableDiffusionPipeline.from_pretrained(self.SDV5_MODEL_PATH)
            pipe = pipe.to('cuda')
            if low_vram:
                pipe.enable_attention_slicing()
        elif device_type == 'cpu':
            pipe = StableDiffusionPipeline.from_pretrained(self.SDV5_MODEL_PATH)

        else:
            print('Invalid Device Type, use cpu or cuda')
            return




        for style_type, style_prompt in art_styles.items():
            prompt_stylized = f"{prompt}, {style_prompt}"
            print(f'Full prompt: \n{prompt_stylized}\n')
            print(f'Characters in prompt: {len(prompt_stylized)}, limit: 200')

            for i in range(num_of_image_per_prompt):
                if device_type == 'cuda':
                    with autocast('cuda'):
                        image = pipe(
                            prompt_stylized,
                            negative_prompt=negative_prompt,
                            height=height,
                            width=width
                        ).images[0]
                else:
                    image = pipe(prompt).images[0]


                image_path = self.uniquify(os.path.join(self.SAVE_PATH, generation_path, style_type + " - "  + shorted_prompt) + '.png')
                print(image_path)
                image.save(image_path)
        print('render finished')






if __name__ == "__main__":
    generator = ImageGenerator()
    height = 512
    width = 720
    num_inference_steps = 100
    device_type = 'cuda'
    low_vram = True
    num_of_image_per_prompt = 3
    prompt = 'In the ocean, the dolphin trying challenge the octopus'
    negative_prompt = ''
    generator.render_prompt()
