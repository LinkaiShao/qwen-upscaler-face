# Qwen Face Upscaler — Development Changelog

## Phase 1: Base LoRA Training (`qwen_upscaler_face/`)

Single-step residual upscaler: predicts v = HR - LR at t=0, then HR_pred = LR + v_pred.

**Architecture**
- Qwen2VL transformer with LoRA (rank 16, alpha 16) on Q/K/V/Out projections
- Frozen VAE (AutoencoderKLQwenImage) for latent encode/decode
- Frozen Qwen2.5-VL text encoder for prompt embeddings
- 2x2 latent patchification for transformer input

**Losses**
- Flow loss: face-weighted MSE on latent residuals (face regions 5x weight)
- Face identity loss: cosine similarity via frozen InceptionResNetV1 (vggface2)
- Anchor loss: L2 regularization on v_pred to prevent large deviations
- Total: `flow + 0.5*id + 0.1*anchor`

**Training**
- 3 epochs, batch_size=1, grad_accum=4 (effective batch 4)
- LR 1e-4, AdamW, 10% linear warmup + cosine decay
- bf16 mixed precision, gradient checkpointing
- Best checkpoint: step-6500 (20260224 run)

**Lighting lock**
- Anchors ambient light of predicted image to LR before face identity comparison
- Y-channel blend (0.80) + gaussian blur (radius 7.0) in YCbCr space
- Prevents identity loss from chasing lighting changes

---

## Phase 2: PatchGAN Discriminator Training (`patch_gan/`)

Standalone discriminator pre-trained on cached predictions from step-6500.

**Precompute**
- Ran frozen step-6500 LoRA on all 11,647 train+val samples
- Saved pred/hr/lr as PNG pixel cache (1024x768)

**Discriminator architecture**
- NLayerDiscriminator (pix2pix-style), 33.2M params
- Input: cat([LR, HR_or_pred]) = (B, 6, 1024, 768)
- Output: (B, 1, 126, 94) patch logits, 70x70 receptive field
- InstanceNorm (works with batch_size=1), LeakyReLU
- Gaussian weight init (std=0.02)

**Training**
- Clothing-only: loss masked by VITON-HD parse labels {5,6,7,9,12}
- BCE with label smoothing (real=0.9, fake=0.0)
- 5 epochs, batch_size=4, lr=2e-4, Adam(0.5, 0.999)
- Output: `patch_gan_disc_clothing/disc_final.pt`

**Metrics (v2)**
- p_real = mean(sigmoid(real_logits)) weighted by clothing mask
- p_fake = mean(sigmoid(fake_logits)) weighted by clothing mask
- gap_p = p_real - p_fake (AUC-like separation proxy)
- Replaced hard-threshold accuracy which oscillated wildly with batch_size=1

---

## Phase 3: GAN-Augmented LoRA Training (`qwen_upscaler_face_gan/`)

Fine-tunes step-6500 LoRA with adversarial loss from pre-trained discriminator.

**Two-phase training**
- Phase 1 (steps 0-1000): D frozen, G trains against fixed adversarial signal
- Phase 2 (step 1000+): D unfrozen with slow updates (lr=2e-5 vs G lr=1e-4)
- D updates every 2 G steps to prevent D from outpacing G

**New GAN loss**
- Shared VAE decode: single decode feeds both face identity and GAN losses
- Disc input: cat([LR_pixels, pred_pixels]) -> patch logits
- G wants disc to output high logits (real label=1.0)
- Masked BCE weighted by clothing parse map coverage per patch
- lambda_gan = 0.005 (small adversarial term)
- Total: `flow + 0.5*id + 0.1*anchor + 0.005*gan`

**Gradient masking fix**
- Problem: GAN gradients flow through VAE decode, affecting all pixels including face
- Solution: detach non-clothing pixels from GAN computation graph
  ```python
  masked_pred = pred * clothing_mask + pred.detach() * (1 - clothing_mask)
  ```
- Only clothing regions receive GAN gradient signal

**Contiguous region mining**
- Clothing mask uses connected component analysis (scipy.ndimage.label)
- Keeps only largest contiguous region per parse label
- Discards stray regions < 500 pixels
- Prevents noisy mislabeled patches from influencing GAN loss

**Training config**
- 2 epochs (~4658 steps), batch_size=1, grad_accum=4
- Starts from step-6500 LoRA + pre-trained clothing disc
- 5% warmup + cosine decay
- Crash recovery with emergency checkpointing

---

## Key Lessons

1. **Lighting lock causes face artifacts** when used during inference (not during training). Canonical pipeline (make_package.py) does NOT use it. Always generate without `--lock_lighting`.

2. **GAN gradient leakage**: Without masking, adversarial gradients through VAE decode distort face/background. Detaching non-clothing pixels is essential.

3. **Hard-threshold D accuracy is misleading**: With batch_size=1 and a large logit map, a slight shift swings accuracy from 0% to 100%. Mean sigmoid probability (p_real, p_fake, gap_p) is far more stable.

4. **D overfitting in phase 2**: Post-trained D called even HR GT fake (39.65%). D learning rate or update frequency needs careful tuning. Updating D every 2 G steps helps.

5. **Pre-trained D should be calibrated**: The v2 disc gives moderate probabilities (HR GT: 0.634, pre-GAN pred: 0.400) with clear separation, rather than extreme overconfident outputs.

6. **Virtual environment**: Must use `ootd` venv at `/home/link/venvs/ootd/` (has facenet_pytorch). NOT tf-gpu.
