from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import LlavaMistralForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    model_name_l = str(model_name).lower()
    model_path_s = str(model_path)
    model_base_s = str(model_base) if model_base is not None else ""
    arch_source = (model_base_s or model_path_s).lower()

    is_llava = ('llava' in model_name_l) or ('llava' in model_base_s.lower()) or ('llava' in model_path_s.lower())
    is_mistral = ('mistral' in model_name_l) or ('mistral' in model_base_s.lower()) or ('mistral' in arch_source)

    if is_llava:
        # Load LLaVA model (optionally with a LoRA adapter when model_base is provided).
        if is_mistral:
            base_for_load = model_base if model_base is not None else model_path
            tokenizer = AutoTokenizer.from_pretrained(base_for_load, use_fast=False)
            model = LlavaMistralForCausalLM.from_pretrained(
                base_for_load,
                low_cpu_mem_usage=False,
                use_flash_attention_2=False,
                **kwargs
            )
            if model_base is not None:
                from peft import PeftModel
                print(f"Loading LoRA adapter from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print("Merging adapter weights...")
                model = model.merge_and_unload()
                print("Convert to FP16...")
                model.to(torch.float16)
        else:
            raise ValueError(f"Unsupported LLaVA model family for model_name={model_name} / model_base={model_base}")
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if is_llava: # or 'mistral' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        model.model.mm_projector.to(device=device, dtype=torch.float16)
        model.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
