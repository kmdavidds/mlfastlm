import time
import json
import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)


###################################################################
# LOAD MODEL ONCE (REUSABLE BENCHMARK OBJECT)
###################################################################
class VLMRunner:
    def __init__(self, model_path, model_base=None, conv_mode="qwen_2"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for benchmarking.")

        device = "cuda"
        disable_torch_init()

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        self.conv_mode = conv_mode
        self.device = device

        # Set pad token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id


    ###################################################################
    # INFERENCE + MEASUREMENT
    ###################################################################
    def run(self, image_path, prompt, max_new_tokens=256):
        # Build conversation
        if self.model.config.mm_use_im_start_end:
            full_prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # Image
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)[0].to(self.device)
        image_tensor = image_tensor.unsqueeze(0).half()

        gen_kwargs = dict(
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            use_cache=True
        )

        results = {}

        # Reset GPU stats
        torch.cuda.reset_peak_memory_stats()

        ###################################################################
        # 1) Optional encoder timing
        ###################################################################
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if hasattr(self.model, "vision_encoder"):
                _ = self.model.vision_encoder(image_tensor)
            elif hasattr(self.model, "encode_image"):
                _ = self.model.encode_image(image_tensor)
            else:
                raise Exception("No image encoder exposed")

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            results["encoder_time_s"] = t1 - t0

        except:
            results["encoder_time_s"] = None

        ###################################################################
        # 2) TTFT — generate 1 token
        ###################################################################
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        first_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=1,
            **gen_kwargs
        )

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        results["ttft_s"] = t_end - t_start

        ###################################################################
        # 3) Full generation
        ###################################################################
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        full_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        results["full_gen_s"] = t3 - t2

        ###################################################################
        # 4) Decode output
        ###################################################################
        output = self.tokenizer.batch_decode(full_ids, skip_special_tokens=True)[0].strip()
        results["output"] = output

        ###################################################################
        # 5) Peak GPU memory
        ###################################################################
        results["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

        return results


###################################################################
# USAGE EXAMPLE
###################################################################
if __name__ == "__main__":
    runner = VLMRunner(
        model_path="./checkpoints/llava-fastvithd_0.5b_stage3",
        model_base=None,
        conv_mode="qwen_2"
    )

    metrics = runner.run(
        image_path="./images/1.png",
        prompt="Describe the image.",
        max_new_tokens=256
    )

    # Pretty print
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
