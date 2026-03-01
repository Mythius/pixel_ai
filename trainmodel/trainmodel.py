#!/usr/bin/env python3
"""
Pixel Art LoRA Fine-tuning for Stable Diffusion 1.5
=====================================================
Fine-tunes a LoRA adapter so SD 1.5 generates pixel art.

Usage:
  python trainmodel.py preprocess            # Scale images to 512x512
  python trainmodel.py train                 # Fine-tune with LoRA
  python trainmodel.py generate "a red dragon"  # Generate pixel art

Install dependencies first:
  pip install torch diffusers transformers peft accelerate Pillow tqdm
  (On Colab, most are pre-installed — just: pip install diffusers peft accelerate)
"""

import json
import math
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ─── PATHS (relative to this script) ─────────────────────────────────────────

SCRIPT_DIR        = Path(__file__).parent
TRAINING_DATA_DIR = SCRIPT_DIR.parent / "training_data"
DESCRIPTIONS_FILE = SCRIPT_DIR.parent / "labeler" / "descriptions.json"
PROCESSED_DIR     = SCRIPT_DIR / "processed_dataset"
LORA_OUTPUT_DIR   = SCRIPT_DIR / "lora_output"

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────

BASE_MODEL    = "runwayml/stable-diffusion-v1-5"
RESOLUTION    = 512       # SD 1.5 native resolution
LORA_RANK     = 8         # LoRA rank — higher = more expressive but slower
LORA_ALPHA    = 16        # LoRA scaling (usually 2x rank)
LEARNING_RATE = 1e-4
TRAIN_STEPS   = 2000      # With 120 images, ~16 epochs at bs=4 — good starting point
BATCH_SIZE    = 1         # Increase to 2-4 if you have >8GB VRAM
GRAD_ACCUM    = 4         # Effective batch size = BATCH_SIZE * GRAD_ACCUM
SAVE_STEPS    = 500
SEED          = 42


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess():
    """
    Scale each pixel art image to 512x512 using NEAREST-NEIGHBOR interpolation.

    Pixel art must NOT use bilinear/bicubic scaling — that would blur the edges
    and destroy the pixel art aesthetic. NEAREST keeps hard edges intact.

    Images are first padded to a square (centered on white background) to
    preserve aspect ratio before scaling.
    """
    print("Loading descriptions...")
    with open(DESCRIPTIONS_FILE) as f:
        descriptions = json.load(f)

    PROCESSED_DIR.mkdir(exist_ok=True)
    metadata_path = PROCESSED_DIR / "metadata.jsonl"

    written = 0
    skipped = 0

    with open(metadata_path, "w") as meta_f:
        for filename, caption in descriptions.items():
            img_path = TRAINING_DATA_DIR / filename
            if not img_path.exists():
                print(f"  [skip] {filename} — not found in training_data/")
                skipped += 1
                continue

            try:
                img = Image.open(img_path).convert("RGBA")
            except Exception as e:
                print(f"  [skip] {filename} — could not open: {e}")
                skipped += 1
                continue

            # 1. Composite RGBA onto white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img_rgb = background

            # 2. Pad to square (center the image)
            w, h = img_rgb.size
            max_side = max(w, h)
            square = Image.new("RGB", (max_side, max_side), (255, 255, 255))
            paste_x = (max_side - w) // 2
            paste_y = (max_side - h) // 2
            square.paste(img_rgb, (paste_x, paste_y))

            # 3. Scale to RESOLUTION x RESOLUTION — NEAREST is critical here
            scaled = square.resize((RESOLUTION, RESOLUTION), Image.NEAREST)

            # 4. Save
            out_path = PROCESSED_DIR / filename
            scaled.save(out_path)

            # 5. Write metadata entry
            meta_f.write(json.dumps({"file_name": filename, "text": caption}) + "\n")
            written += 1

    print(f"\nDone! {written} images preprocessed → {PROCESSED_DIR}/")
    if skipped:
        print(f"Skipped {skipped} images (not found or unreadable)")
    print(f"Metadata written to {metadata_path}")
    print("\nNext step: python trainmodel.py train")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PixelArtDataset(torch.utils.data.Dataset):
    """
    Loads preprocessed 512x512 pixel art images and their captions.
    Returns (pixel_values, input_ids) for each training sample.
    """

    def __init__(self, dataset_dir: Path, tokenizer, resolution: int = 512):
        from torchvision import transforms

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.resolution = resolution

        self.items = []
        metadata_path = dataset_dir / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found at {metadata_path}. "
                "Run: python trainmodel.py preprocess"
            )

        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        # Normalize to [-1, 1] — what SD's VAE expects
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        img = Image.open(self.dataset_dir / item["file_name"]).convert("RGB")
        pixel_values = self.transform(img)

        tokens = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    """
    Fine-tune Stable Diffusion 1.5 with LoRA on pixel art images.

    Architecture:
      - VAE encodes images to latent space (frozen)
      - CLIP text encoder encodes captions (frozen)
      - UNet predicts noise — this is where LoRA adapters are added
      - Only the tiny LoRA matrices (~3MB) are trained; everything else frozen

    Loss: MSE between predicted noise and actual noise (standard diffusion loss)
    """
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.optimization import get_scheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install diffusers transformers peft accelerate")
        return

    torch.manual_seed(SEED)

    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        weight_dtype = torch.float16   # saves VRAM on GPU
        print("Using CUDA (GPU) ✓")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        weight_dtype = torch.float32   # MPS doesn't support float16 reliably
        print("Using Apple MPS (Apple Silicon) — training will be slow")
        print("Recommend: use Google Colab for faster training")
    else:
        device = torch.device("cpu")
        weight_dtype = torch.float32
        print("WARNING: Using CPU — this will be very slow. Use Colab instead.")

    # ── Load model components ─────────────────────────────────────────────────
    print(f"\nDownloading/loading {BASE_MODEL}...")
    print("(First run will download ~4GB — subsequent runs use cache)")

    tokenizer    = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    # Freeze VAE and text encoder — we only want to teach the UNet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae          = vae.to(device, dtype=weight_dtype)
    text_encoder = text_encoder.to(device, dtype=weight_dtype)
    unet         = unet.to(device)   # keep UNet in float32 for stable LoRA training

    # ── Apply LoRA to UNet attention layers ───────────────────────────────────
    #
    # LoRA adds small trainable matrices to the attention Q/K/V/Out projections.
    # Instead of updating 860M params, we train ~3M — much less prone to overfitting.
    #
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = PixelArtDataset(PROCESSED_DIR, tokenizer, RESOLUTION)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,   # 0 is safer on Mac/Windows
        drop_last=True,
    )
    print(f"\nDataset: {len(dataset)} images")
    print(f"Steps per epoch: {math.ceil(len(dataset) / args.batch_size)}")
    print(f"Effective batch size: {args.batch_size} × {GRAD_ACCUM} = {args.batch_size * GRAD_ACCUM}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=max(100, args.steps // 20),
        num_training_steps=args.steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    LORA_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nTraining for {args.steps} steps...")
    print("Checkpoints saved every", SAVE_STEPS, "steps to", LORA_OUTPUT_DIR)

    global_step = 0
    epoch = 0
    loss_log = []

    progress = tqdm(total=args.steps, desc="Training", unit="step")

    while global_step < args.steps:
        epoch += 1
        unet.train()
        accum_loss = 0.0
        accum_count = 0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids    = batch["input_ids"].to(device)

            # Encode images → latent space (8x compressed, 64x64 for 512x512 images)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(torch.float32)

            # Sample noise and random timestep for each image in batch
            noise     = torch.randn_like(latents)
            bsz       = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_sched.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()

            # Forward diffusion: add noise at the sampled timestep
            noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

            # Encode text captions
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0].to(torch.float32)

            # Predict the noise with the UNet (with LoRA adapters)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample

            # Standard diffusion loss: MSE between predicted and actual noise
            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float())
            loss = loss / GRAD_ACCUM

            loss.backward()
            accum_loss += loss.item()
            accum_count += 1

            # Gradient accumulation step
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, unet.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accum_loss / accum_count
                loss_log.append(avg_loss)
                accum_loss = 0.0
                accum_count = 0

                progress.update(1)
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    epoch=epoch,
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    ckpt_dir = LORA_OUTPUT_DIR / f"checkpoint-{global_step}"
                    unet.save_pretrained(ckpt_dir)
                    # Also save loss log
                    with open(LORA_OUTPUT_DIR / "loss_log.json", "w") as f:
                        json.dump(loss_log, f)
                    print(f"\n  [checkpoint saved at step {global_step}]")

                if global_step >= args.steps:
                    break

    progress.close()

    # Save final LoRA weights
    final_dir = LORA_OUTPUT_DIR / "final"
    unet.save_pretrained(str(final_dir))
    with open(LORA_OUTPUT_DIR / "loss_log.json", "w") as f:
        json.dump(loss_log, f)

    print(f"\nTraining complete!")
    print(f"LoRA weights saved to: {final_dir}")
    print(f"\nNext step: python trainmodel.py generate \"pixel image: a green dragon\"")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def generate(args):
    """
    Generate pixel art from a text prompt using the trained LoRA.

    Tips for good prompts:
      - Start with "pixel image:" to trigger the pixel art style
      - Include the size like "(16x16)" or "(32x32)" to match training data format
      - Example: "pixel image: (16x16) a small blue dragon breathing fire"
    """
    try:
        from diffusers import StableDiffusionPipeline
        from peft import PeftModel
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    lora_path = LORA_OUTPUT_DIR / "final"
    if not lora_path.exists():
        # Check for latest checkpoint
        checkpoints = sorted(LORA_OUTPUT_DIR.glob("checkpoint-*"))
        if checkpoints:
            lora_path = checkpoints[-1]
            print(f"Using latest checkpoint: {lora_path}")
        else:
            print(f"No trained model found. Run: python trainmodel.py train")
            return

    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading pipeline on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Load LoRA weights into the UNet
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()  # saves VRAM

    prompt = args.prompt
    negative_prompt = args.negative or "blurry, smooth, photorealistic, 3d, realistic"

    print(f"\nPrompt:   {prompt}")
    print(f"Negative: {negative_prompt}")
    print(f"Generating {args.num} image(s)...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for i in range(args.num):
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator(device=device).manual_seed(SEED + i),
        )
        image = result.images[0]

        if args.num == 1:
            out_path = out_dir / "generated.png"
        else:
            out_path = out_dir / f"generated_{i+1}.png"

        image.save(out_path)
        print(f"  Saved: {out_path}")

    print("\nDone!")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pixel Art LoRA Fine-tuning for Stable Diffusion 1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trainmodel.py preprocess
  python trainmodel.py train --steps 2000 --lr 1e-4
  python trainmodel.py generate "pixel image: (16x16) a small red mushroom"
  python trainmodel.py generate "pixel image: a golden sword" --num 4 --cfg 8.0
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── preprocess ────────────────────────────────────────────────────────────
    sub_pre = subparsers.add_parser(
        "preprocess",
        help="Scale training images to 512x512 and build metadata.jsonl",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    sub_train = subparsers.add_parser(
        "train",
        help="Fine-tune SD 1.5 with LoRA on the preprocessed dataset",
    )
    sub_train.add_argument(
        "--steps", type=int, default=TRAIN_STEPS,
        help=f"Total training steps (default: {TRAIN_STEPS})",
    )
    sub_train.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    sub_train.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
        help=f"Batch size per step (default: {BATCH_SIZE})",
    )
    sub_train.add_argument(
        "--rank", type=int, default=LORA_RANK,
        help=f"LoRA rank — higher = more expressive (default: {LORA_RANK})",
    )

    # ── generate ──────────────────────────────────────────────────────────────
    sub_gen = subparsers.add_parser(
        "generate",
        help="Generate pixel art from a text prompt using the trained LoRA",
    )
    sub_gen.add_argument(
        "prompt", type=str,
        help='Text prompt, e.g. "pixel image: (16x16) a small blue dragon"',
    )
    sub_gen.add_argument(
        "--negative", type=str, default=None,
        help="Negative prompt (default: blurry, smooth, photorealistic, 3d)",
    )
    sub_gen.add_argument(
        "--num", type=int, default=1,
        help="Number of images to generate (default: 1)",
    )
    sub_gen.add_argument(
        "--steps", type=int, default=50,
        help="Denoising steps (default: 50, more = higher quality)",
    )
    sub_gen.add_argument(
        "--cfg", type=float, default=7.5,
        help="Classifier-free guidance scale (default: 7.5, higher = more prompt-adherent)",
    )
    sub_gen.add_argument(
        "--output-dir", type=str, default=str(SCRIPT_DIR / "output"), dest="output_dir",
        help="Directory to save generated images",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess()
    elif args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
