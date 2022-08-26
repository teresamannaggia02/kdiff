import PIL
import gradio as gr
import argparse, os, sys, glob
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import accelerate
import mimetypes
import gc
from basicsr.utils import imwrite
import cv2
from gfpgan import GFPGANer
from io import BytesIO
import random

mimetypes.init()
mimetypes.add_type('application/javascript', '.js')


import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def FACE_RESTORATION(image, bg_upsampling, upscale, mpth):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True)
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(mpth)
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=None
        )
        
    image=np.array(image)
    input_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    
    cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=False, only_center_face=False, paste_back=True)

    image = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    image = bg_upsampler.enhance(image, outscale=upscale)[0]
    del bg_upsampler
    del restorer
    del model
    return image

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_img_pil(img_pil):
    image = img_pil.convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    print(f"cropped image to size ({w}, {h})")
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_img(path):
    return load_img_pil(Image.open(path))
	
	
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

#from https://github.com/lstein/stable-diffusion/blob/main/ldm/simplet2i.py
def split_weighted_subprompts(text):
        """
        grabs all text up to the first occurrence of ':' 
        uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
        if ':' has no value defined, defaults to 1.0
        repeats until no text remaining
        """
        remaining = len(text)
        prompts = []
        weights = []
        while remaining > 0:
            if ":" in text:
                idx = text.index(":") # first occurrence from start
                # grab up to index as sub-prompt
                prompt = text[:idx]
                remaining -= idx
                # remove from main text
                text = text[idx+1:]
                # find value for weight 
                if " " in text:
                    idx = text.index(" ") # first occurence
                else: # no space, read to end
                    idx = len(text)
                if idx != 0:
                    try:
                        weight = float(text[:idx])
                    except: # couldn't treat as float
                        print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                        weight = 1.0
                else: # no value found
                    weight = 1.0
                # remove from main text
                remaining -= idx
                text = text[idx+1:]
                # append the sub-prompt and its weight
                prompts.append(prompt)
                weights.append(weight)
            else: # no : found
                if len(text) > 0: # there is still text though
                    # take remainder as weight 1
                    prompts.append(text)
                    weights.append(1.0)
                remaining = 0
        return prompts, weights



parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./configs/stable-diffusion/v1-inference.yaml", type=str)
parser.add_argument("--ckpt", default="./models/ldm/stable-diffusion-v1/model.ckpt", type=str)
parser.add_argument("--precision", default="autocast", type=str)
parser.add_argument("--outdir", default="./outputs/txt2img-samples", type=str)
parser.add_argument("--GFPGAN", default='./models/ldm/stable-diffusion-v1/GFPGANv1.3.pth', type=str)

args = parser.parse_args()


config = OmegaConf.load(args.config)
model = load_model_from_config(config, args.ckpt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.half().to(device)


def dream(prompt: str, init_img, ddim_steps: int, plms: bool, fixed_code: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scales: str, denoising_strength: float, seed: int, height: int, width: int, same_seed: bool, GFPGAN: bool, bg_upsampling: bool, upscale: int, outdir = args.outdir, precision = args.precision, GFPGANth = args.GFPGAN):
    torch.cuda.empty_cache()

    H = height
    W = width 
    C = 4
    f = 8
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    rng_seed = seed_everything(seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    if plms and init_img == None:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    prompt = prompt
    assert prompt is not None
    data = [batch_size * [prompt]]


    sample_path = os.path.join(outpath)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    seedit = 0

    if fixed_code and init_img == None:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)
        
    if init_img != None:
        image = init_img.convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        w, h = map(lambda x: x - x % 32, (W, H))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        print(f"cropped image to size ({w}, {h})")
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    if len(cfg_scales) > 1: cfg_scales = list(map(float, cfg_scales.split(' '))) 
    output_images = []
    precision_scope = autocast if precision == "autocast" else nullcontext
    ff = []
    message = ''
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache() 
        with precision_scope("cuda"):
            if init_img != None:
                init_image = 2.*image - 1.
                init_image = init_image.to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                x0 = init_latent
                
                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
                
                assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
                t_enc = int(denoising_strength * ddim_steps)
                print(f"target t_enc is {t_enc} steps")
            del sampler
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        output_images.append([])
                        
                        aaa = f'{rng_seed + seedit}{random.randint(8, 10000)}'
                        output_images[-1].append(aaa)
                        output_images[-1].append(prompts[0])

                        os.makedirs(f'{outpath}/{aaa}', exist_ok=True)
                        for n in trange(n_iter, desc="Sampling", disable=not accelerator.is_main_process):
                            
                            
                            with open(f'{outpath}/{aaa}/prompt.txt', 'w') as fff:
                                fff.write(prompts[0])
                            
                            if n_iter > 1: seedit += 1
                            for cfg in tqdm(cfg_scales, desc="cfg_scales", disable=not accelerator.is_main_process):
                                cfg_scale = cfg
                                uc = None
                                if cfg_scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                    
                                    
                                #from https://github.com/lstein/stable-diffusion/blob/main/ldm/simplet2i.py
                                subprompts,weights = split_weighted_subprompts(prompts[0])
                                
                                if len(subprompts) > 1:
                                    # i dont know if this is correct.. but it works
                                    c = torch.zeros_like(uc)
                                    # get total weight for normalizing
                                    totalWeight = sum(weights)
                                    # normalize each "sub prompt" and add it
                                    for i in range(0,len(subprompts)):
                                        weight = weights[i]
                                        weight = weight / totalWeight
                                        c = torch.add(c,model.get_learned_conditioning(subprompts[i]), alpha=weight)
                                        
                                        
                                else: # just standard 1 prompt
                                    c = model.get_learned_conditioning(prompts)
                                
                                torch.manual_seed(rng_seed + seedit) # changes manual seeding procedure
                                sigmas = model_wrap.get_sigmas(ddim_steps)
                                if init_img == None:
                                    shape = [C, H // f, W // f]
                                    x = torch.randn([n_samples, *shape], device=device) * sigmas[0] # for GPU draw
                                else:
                                    noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1] # for GPU draw
                                    x = x0 + noise
                                    sigmas = sigmas[ddim_steps - t_enc - 1:]
                                
                                model_wrap_cfg = CFGDenoiser(model_wrap)
                                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': cfg_scale}
                                samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                                x_samples_ddim = accelerator.gather(x_samples_ddim)
                                #del samples_ddim, model_wrap_cfg
                                if accelerator.is_main_process:
                                    for x_sample in x_samples_ddim:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        output_images[-1].append(Image.fromarray(x_sample.astype(np.uint8)))
                                        # aaa = output_images[-1][0]
                                        # message+= f'Запрос "{output_images[-1][1]}" находится в папке {outpath}/{aaa}/ \n'
                                        # pt = f'{outpath}/{aaa}/{cfg_scale} {random.randint(1, 200000)}.jpg'
                                        # gc.collect()
                                        # torch.cuda.empty_cache() 
                                        # if GFPGAN:
                                            # (Image.fromarray(FACE_RESTORATION(output_images[-1][-1], bg_upsampling, upscale, GFPGANth).astype(np.uint8))).save(pt, format = 'JPEG', mize = True)
                                        # else:
                                            # output_images[-1][-1].save(pt, format = 'JPEG', mize = True)
                                        # #del output_images[-1][-1]
                                        # ff.append(pt)
                                        base_count += 1
                                        if not same_seed: seedit += 1
                                    #del x_samples_ddim

                
                toc = time.time()
                gc.collect()
                torch.cuda.empty_cache() 

    for i in trange(len(output_images), desc="prompts"):
        z = 0
        aaa = output_images[i][0]
        message+= f'Запрос "{output_images[i][1]}" находится в папке {outpath}/{aaa}/ \n'
        for k in range(2, len(output_images[i])):
            if k%batch_size == 0:
                cfg=cfg_scales[z]
                if z+1> len(cfg_scales)-1:
                    z=0
                else:
                    z+=1
            pt = f'{outpath}/{aaa}/{cfg} {random.randint(1, 200000)}.jpg'
            if GFPGAN:
                (Image.fromarray(FACE_RESTORATION(output_images[i][k], bg_upsampling, upscale, GFPGANth).astype(np.uint8))).save(pt, format = 'JPEG', mize = True)
            else:
                output_images[i][k].save(pt, format = 'JPEG', mize = True)
            ff.append(pt)

    return ff, rng_seed, message

def dev_dream(prompt: str, init_img,use_img: bool, ddim_steps: int, plms: bool, fixed_code: bool, ddim_eta: float, n_iter: int, n_samples: int, cfg_scales: str, denoising_strength: float, seed: int, height: int, width: int, same_seed: bool, GFPGAN: bool, bg_upsampling: bool, upscale: int):
    prompts = list(map(str, prompt.split('|'))) 
    if not use_img:
        init_img=None
    f, rng_seed, message = [], [], ''
    for prompt in tqdm(prompts, desc="prompts"):
        ff, rng_seedf, messagef = dream(prompt, init_img, ddim_steps, plms, fixed_code, ddim_eta, n_iter, n_samples, cfg_scales, denoising_strength, seed, height, width, same_seed, GFPGAN, bg_upsampling, upscale)
        for i in ff:
            f.append(i)
        rng_seed=rng_seedf
        message+=messagef
    return f, rng_seed, message


 

dream_interface = gr.Interface(
    dream,
    inputs=[
        gr.Textbox(label='Query di testo. Supporta la ponderazione di parti della query con ":number" (è richiesto lo spazio dopo il numero). È supportata anche la query normale.',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1),        
        gr.Variable(value=None, visible=False),
        gr.Slider(minimum=1, maximum=500, step=1, label="Passi di diffusione, ideale - 100.", value=50),
        gr.Checkbox(label='Abilita PLMS', value=True),
        gr.Checkbox(label='Campionamento da un punto', value=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=50, step=1, label='Quante volte generare su richiesta (in sequenza)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Quante immagini alla volta (contemporaneamente). USA MOLTA MEMORIA', value=1),
        gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scale, separate da uno spazio o solo da una. Se più di una, campiona la stessa query con cfgs diversi. Numero richiesto con un punto, ad esempio 7.0 o 15.0', lines=1, value=9.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Percentuale di passaggi sopra per scorrere l''immagine. Può essere considerato "forza"', value=0.75, visible=False),
        gr.Number(label='SEED', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Высота", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Ширина", value=512),
        gr.Checkbox(label='Lo stesso seed ogni volta. Per generare lo stesso seed con la stessa query.', value=False),
        gr.Checkbox(label='GFPGAN, ripristina i volti, può eseguire l''upscaling. Tutte le impostazioni sottostanti.', value=True),
        gr.Checkbox(label='Miglioramento dello sfondo', value=True),
        gr.Slider(minimum=1, maximum=8, step=1, label="Upscaling. 1 significa non utilizzato.", value=2)
    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed'),
        gr.Textbox(label='Per scaricare la cartella con i risultati, apri i file sul lato sinistro della colab e scarica le cartelle specificate')
    ],
    title="Stable Diffusion 1.4 da testo a immagine",
    description="             crea un'immagine dal testo, anon, K-LMS è l'impostazione predefinita",
)

# prompt, init_img, ddim_steps, plms, ddim_eta, n_iter, n_samples, cfg_scale, denoising_strength, seed

img2img_interface = gr.Interface(
    dream,
    inputs=[
        gr.Textbox(label='Query di testo. Supporta l''assegnazione di peso a parti della query con ":number" (è richiesto lo spazio dopo il numero). È supportata anche la query normale.',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Image(value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", source="upload", interactive=True, type="pil"),
        gr.Slider(minimum=1, maximum=500, step=1, label="Passi di diffusione, ideale - 100.", value=100),
        gr.Checkbox(label='Abilita PLMS', value=True, visible=False),
        gr.Checkbox(label='Campionamento da un punto', value=False, visible=False),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=False),
        gr.Slider(minimum=1, maximum=50, step=1, label='Quante volte generare su richiesta (in sequenza)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Quante immagini alla volta (contemporaneamente). USA MOLTA MEMORIA', value=1),
        gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scale, separate da uno spazio o solo da una. Se più di una, campiona la stessa query con cfgs diversi. Numero richiesto con un punto, ad esempio 7.0 o 15.0', lines=1, value=9.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Percentuale di passaggi sopra per scorrere l''immagine. Può essere considerato "forza"', value=0.75),
        gr.Number(label='SEED', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Width", value=512),
        gr.Checkbox(label='Lo stesso seed ogni volta. Per generare lo stesso seed con la stessa query.', value=False),
        gr.Checkbox(label='GFPGAN, ripristina i volti, può eseguire l''upscaling. Tutte le impostazioni sottostanti.', value=True),
        gr.Checkbox(label='Miglioramento dello sfondo', value=True),
        gr.Slider(minimum=1, maximum=8, step=1, label="Upscaling. 1 significa non utilizzato.", value=2)

    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='Seed'),
        gr.Textbox(label='Per scaricare la cartella con i risultati, apri i file sul lato sinistro della colab e scarica le cartelle specificate')
    ],
    title="Stable Diffusion 1.4 Image-to-Image",
    description="genera un'immagine da un'immagine",
)

ctrbbl_interface = gr.Interface(
    dev_dream,
    inputs=[
        gr.Textbox(label='Query di testo delimitate da '|'. Supporta l''assegnazione di peso alle parti di query con ":number" (è richiesto lo spazio dopo il numero). È supportata anche la query normale.',  placeholder="A corgi wearing a top hat as an oil painting.", lines=1),
        gr.Image(value="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg", source="upload", interactive=True, type="pil"),
        gr.Checkbox(label='Usa img2img (disabilitato, significa che l''immagine viene ignorata)', value=False, visible=True),
        gr.Slider(minimum=1, maximum=500, step=1, label="Passi di diffusione, ideale - 100.", value=100),
        gr.Checkbox(label='Abilita PLMS', value=True, visible=True),
        gr.Checkbox(label='Campionamento da un punto', value=False, visible=True),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="DDIM ETA", value=0.0, visible=True),
        gr.Slider(minimum=1, maximum=50, step=1, label='Quante volte generare su richiesta (in sequenza)', value=1),
        gr.Slider(minimum=1, maximum=8, step=1, label='Quante immagini alla volta (contemporaneamente). USA MOLTA MEMORIA', value=1),
        gr.Textbox(placeholder="7.0", label='Classifier Free Guidance Scale, separate da uno spazio o solo da una. Se più di una, campiona la stessa query con cfgs diversi. Numero richiesto con un punto, ad esempio 7.0 o 15.0', lines=1, value=9.0),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Percentuale di passaggi sopra per scorrere l''immagine. Può essere considerato "forza"', value=0.75),
        gr.Number(label='SEED', value=-1),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Height", value=512),
        gr.Slider(minimum=64, maximum=2048, step=64, label="Resize Width", value=512),
        gr.Checkbox(label='Lo stesso seed ogni volta. Per generare lo stesso seed con la stessa query.', value=False),
        gr.Checkbox(label='GFPGAN, ripristina i volti, può eseguire l''upscaling. Tutte le impostazioni sottostanti.', value=True),
        gr.Checkbox(label='Miglioramento dello sfondo', value=True),
        gr.Slider(minimum=1, maximum=8, step=1, label="Upscaling. 1 significa non utilizzato.", value=2)

    ],
    outputs=[
        gr.Gallery(),
        gr.Number(label='SEED'),
        gr.Textbox(label='Per scaricare la cartella con i risultati, apri i file sul lato sinistro della colab e scarica le cartelle specificate')
    ],
    title="Stable Diffusion multiprompt",
    description="scheda sperimentale per trovare le impostazioni ideali",
)



demo = gr.TabbedInterface(interface_list=[dream_interface, img2img_interface, ctrbbl_interface], tab_names=["Dream", "Image Translation", "Dev inference"])

demo.launch(share=True)