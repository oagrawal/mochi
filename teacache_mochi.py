import argparse
import os
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import MochiPipeline
from diffusers.models.transformers import MochiTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Rescale coefficients for delta TEMNI (rescaled relative L1), same as TeaCache logic
DELTA_TEMNI_COEFFICIENTS = [-3.51241319e+03, 8.11675948e+02, -6.09400215e+01, 2.42429681e+00, 3.05291719e-03]


def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p = self.config.patch_size

        post_patch_height = height // p
        post_patch_width = width // p

        temb, encoder_hidden_states = self.time_embed(
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_dtype=hidden_states.dtype,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.unflatten(0, (batch_size, -1)).flatten(1, 2)

        image_rotary_emb = self.rope(
            self.pos_frequencies,
            num_frames,
            post_patch_height,
            post_patch_width,
            device=hidden_states.device,
            dtype=torch.float32,
        )

        record_delta_temni = getattr(self, "delta_TEMNI", None) is not None and isinstance(
            getattr(self, "delta_TEMNI"), list
        )
        should_calc = True  # default when not using teacache

        if self.enable_teacache or record_delta_temni:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            modulated_inp, gate_msa, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, temb_)
            if record_delta_temni:
                delta_temni_list = self.delta_TEMNI
                if getattr(self, "previous_modulated_input", None) is not None:
                    raw_delta = (
                        (modulated_inp - self.previous_modulated_input).abs().mean()
                        / self.previous_modulated_input.abs().mean()
                    ).cpu().item()
                    rescale_func = np.poly1d(DELTA_TEMNI_COEFFICIENTS)
                    delta_temni_list.append(float(rescale_func(raw_delta)))
                else:
                    delta_temni_list.append(0.0)
                self.previous_modulated_input = modulated_inp.clone()
            if self.enable_teacache:
                if self.cnt == 0 or self.cnt == self.num_steps - 1:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                else:
                    rescale_func = np.poly1d(DELTA_TEMNI_COEFFICIENTS)
                    self.accumulated_rel_l1_distance += rescale_func(
                        (
                            (modulated_inp - self.previous_modulated_input).abs().mean()
                            / self.previous_modulated_input.abs().mean()
                        ).cpu().item()
                    )
                    # Unified low/high: same value = fixed threshold; different = adaptive (first N steps high, rest low)
                    thresh_low = getattr(self, "rel_l1_thresh_low", None)
                    thresh_high = getattr(self, "rel_l1_thresh_high", None)
                    if thresh_low is not None and thresh_high is not None:
                        adaptive_steps = getattr(self, "adaptive_high_steps", 33)
                        current_thresh = thresh_high if self.cnt <= adaptive_steps else thresh_low
                    else:
                        current_thresh = self.rel_l1_thresh
                    if self.accumulated_rel_l1_distance < current_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance = 0
                self.previous_modulated_input = modulated_inp
                self.cnt += 1
                if self.cnt == self.num_steps:
                    self.cnt = 0

        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for i, block in enumerate(self.transformer_blocks):
                  if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                              def custom_forward(*inputs):
                                    return module(*inputs)

                              return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                              create_custom_forward(block),
                              hidden_states,
                              encoder_hidden_states,
                              temb,
                              encoder_attention_mask,
                              image_rotary_emb,
                              **ckpt_kwargs,
                        )
                  else:
                        hidden_states, encoder_hidden_states = block(
                              hidden_states=hidden_states,
                              encoder_hidden_states=encoder_hidden_states,
                              temb=temb,
                              encoder_attention_mask=encoder_attention_mask,
                              image_rotary_emb=image_rotary_emb,
                        )
                hidden_states = self.norm_out(hidden_states, temb)
                self.previous_residual = hidden_states - ori_hidden_states                
        else:
            for i, block in enumerate(self.transformer_blocks):
                  if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module):
                              def custom_forward(*inputs):
                                    return module(*inputs)

                              return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                              create_custom_forward(block),
                              hidden_states,
                              encoder_hidden_states,
                              temb,
                              encoder_attention_mask,
                              image_rotary_emb,
                              **ckpt_kwargs,
                        )
                  else:
                        hidden_states, encoder_hidden_states = block(
                              hidden_states=hidden_states,
                              encoder_hidden_states=encoder_hidden_states,
                              temb=temb,
                              encoder_attention_mask=encoder_attention_mask,
                              image_rotary_emb=image_rotary_emb,
                        )
            hidden_states = self.norm_out(hidden_states, temb)

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, num_frames, post_patch_height, post_patch_width, p, p, -1)
        hidden_states = hidden_states.permute(0, 6, 1, 2, 4, 3, 5)
        output = hidden_states.reshape(batch_size, -1, num_frames, height, width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def _plot_delta_temni(transformer_class, save_file):
    """Plot and save delta TEMNI when recorded (no TeaCache). Same style as Wan2.1."""
    delta_temni = getattr(transformer_class, "delta_TEMNI", None)
    if not delta_temni or len(delta_temni) == 0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = os.path.dirname(os.path.abspath(save_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(save_file))[0]
    plt.figure(figsize=(10, 6))
    x = range(1, len(delta_temni) + 1)
    plt.plot(x, delta_temni, "g-", linewidth=2, marker="s", markersize=4)
    plt.xlabel("Forward step (cond/uncond pair per solver step)")
    plt.ylabel("Delta TEMNI (rescaled relative L1)")
    plt.title("Delta TEMNI over steps (no TeaCache)")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    step = max(1, len(delta_temni) // 20)
    ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    plot_path = os.path.join(out_dir, f"{base}_delta_TEMNI_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Delta TEMNI plot saved to: {plot_path}")
    txt_path = os.path.join(out_dir, f"{base}_delta_TEMNI.txt")
    with open(txt_path, "w") as f:
        for v in delta_temni:
            f.write(f"{v}\n")
    print(f"Delta TEMNI values saved to: {txt_path}")


MochiTransformer3DModel.forward = teacache_forward

DEFAULT_PROMPT = (
    "A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. "
    "The hand gently tosses the lemon up and catches it, showcasing its smooth texture. "
    "A beige string bag sits beside the bowl, adding a rustic touch to the scene. "
    "Additional lemons, one halved, are scattered around the base of the bowl. "
    "The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere."
)


def run_generation(
    prompt=DEFAULT_PROMPT,
    num_inference_steps=64,
    seed=0,
    out_dir="outputs",
    enable_teacache=True,
    rel_l1_thresh=0.09,
    rel_l1_thresh_low=None,
    rel_l1_thresh_high=None,
    adaptive_high_steps=33,
    record_delta_temni=False,
    save_file=None,
):
    # Unified threshold: use low/high if both provided (fixed when equal, adaptive when different); else single rel_l1_thresh
    if rel_l1_thresh_low is not None or rel_l1_thresh_high is not None:
        low = rel_l1_thresh_low if rel_l1_thresh_low is not None else rel_l1_thresh_high
        high = rel_l1_thresh_high if rel_l1_thresh_high is not None else rel_l1_thresh_low
        thresh_low, thresh_high = low, high
    else:
        thresh_low = thresh_high = rel_l1_thresh

    pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", force_zeros_for_empty_prompt=True)
    cls = pipe.transformer.__class__
    cls.enable_teacache = enable_teacache
    cls.cnt = 0
    cls.num_steps = num_inference_steps
    cls.rel_l1_thresh = rel_l1_thresh  # kept for backward compat when only this is used
    cls.rel_l1_thresh_low = thresh_low
    cls.rel_l1_thresh_high = thresh_high
    cls.adaptive_high_steps = adaptive_high_steps
    cls.accumulated_rel_l1_distance = 0
    cls.previous_modulated_input = None
    cls.previous_residual = None
    if record_delta_temni:
        cls.delta_TEMNI = []
    else:
        cls.delta_TEMNI = None

    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            pipe.encode_prompt(prompt=prompt)
        )

    with torch.autocast("cuda", torch.bfloat16):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            frames = pipe(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                guidance_scale=4.5,
                num_inference_steps=num_inference_steps,
                height=480,
                width=848,
                num_frames=163,
                generator=torch.Generator("cuda").manual_seed(seed),
                output_type="latent",
                return_dict=False,
            )[0]

    video_processor = VideoProcessor(vae_scale_factor=8)
    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(frames.device, frames.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std).view(1, 12, 1, 1, 1).to(frames.device, frames.dtype)
        )
        frames = frames * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        frames = frames / pipe.vae.config.scaling_factor

    with torch.no_grad():
        video = pipe.vae.decode(frames.to(pipe.vae.dtype), return_dict=False)[0]

    video = video_processor.postprocess_video(video)[0]
    os.makedirs(out_dir, exist_ok=True)
    if save_file:
        video_path = save_file if save_file.endswith(".mp4") else save_file + ".mp4"
    else:
        video_path = os.path.join(out_dir, "teacache_mochi__{}.mp4".format(prompt[:50].replace("/", "_")))
    video_path = os.path.abspath(os.path.normpath(video_path))
    video_dir = os.path.dirname(video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    export_to_video(video, video_path, fps=30)
    print(f"Video saved to: {video_path}")

    if record_delta_temni:
        _plot_delta_temni(cls, video_path)
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Mochi + TeaCache or delta TEMNI plot (no cache)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for video")
    parser.add_argument("--save_file", type=str, default=None, help="Output video path (and base name for delta TEMNI plot/txt). If not set, auto-generated in out_dir.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of inference steps")
    parser.add_argument("--delta_temni_plot", action="store_true", help="No caching; record and plot delta TEMNI over steps (like Wan2.1 baseline)")
    parser.add_argument("--teacache_thresh", type=float, default=0.09, help="Single TeaCache threshold (fixed mode) when --teacache_thresh_low/high not set")
    parser.add_argument("--teacache_thresh_low", type=float, default=None, help="Low threshold. With thresh_high: same value = fixed; different = adaptive (first N steps high, rest low)")
    parser.add_argument("--teacache_thresh_high", type=float, default=None, help="High threshold. With thresh_low: same value = fixed; different = adaptive")
    parser.add_argument("--teacache_adaptive_high_steps", type=int, default=33, help="When adaptive: number of steps (0-indexed) that use high threshold; rest use low. Default 33.")
    args = parser.parse_args()

    if args.save_file and not os.path.isabs(args.save_file) and not os.path.dirname(args.save_file):
        args.save_file = os.path.join(args.out_dir, args.save_file)

    run_generation(
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        seed=args.seed,
        out_dir=args.out_dir,
        enable_teacache=not args.delta_temni_plot,
        rel_l1_thresh=args.teacache_thresh,
        rel_l1_thresh_low=args.teacache_thresh_low,
        rel_l1_thresh_high=args.teacache_thresh_high,
        adaptive_high_steps=args.teacache_adaptive_high_steps,
        record_delta_temni=args.delta_temni_plot,
        save_file=args.save_file,
    )


if __name__ == "__main__":
    main()