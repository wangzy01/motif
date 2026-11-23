# Motif

Motif is a research codebase for anime text-to-image generation. Its central question is how to reuse a pretrained language model’s knowledge of characters, attributes, and composition when generating an illustration. Motif keeps the language backbone of Qwen3-VL-2B-Instruct frozen, copies its transformer blocks into a trainable image branch, and teaches that branch to turn Gaussian noise into an image in FLUX.2 VAE latent space.

One evaluation prompt asks for a full-body illustration of two idols in school uniforms, one holding a microphone. Producing a plausible idol portrait satisfies only part of the request. The subject count, framing, clothing, and microphone give the experiments concrete details to check. [Evaluation prompts](experiments/eval/prompts/anime_evalset_nl13_plain_headcrop.jsonl)

## Two ways to condition the generator

The baseline is a 28-layer **Mixture-of-Transformers (MoT)**. The frozen text branch has about 1.72B parameters. Its trainable image counterpart starts from the corresponding Qwen layer weights and has about 1.42B parameters, excluding the vocabulary embedding.

| Design | What an image token can read |
| --- | --- |
| Joint attention baseline | At each layer, an image token attends to every valid prompt token and every image token. Each modality has its own attention projections, normalization, and MLP. Text attention stays causal and independent of the image, so the sampler can cache text keys and values across denoising steps. |
| Stage-256 text adapter | The frozen branch first encodes the whole prompt. A learned gate blends its final hidden state with a weighted sum of its last 12 hidden states. A two-layer causal adapter transforms that sequence into text features that every image layer can attend to. At initialization the gate selects the final hidden state alone. |

The same backbone also supports experiments with low-rank updates to the text attention projections, long skip connections in the image branch, and removal of noise-level conditioning. [Transformer implementation](networks/arch_mot.py) · [Stage-256 recipe](configs/stage256_20260718_causalmlf_offline_exact_v1.yaml)

## What the model learns

The frozen FLUX.2 VAE maps an RGB image to a normalized grid of 128-channel latents at 1/16 of the input height and width. Each grid cell becomes one image token, so a 256 × 256 training image contributes 256 tokens. [VAE implementation](networks/vae.py)

For a clean latent $x_0$, Gaussian noise $\epsilon$, and noise level $\sigma$, the rectified-flow objective follows a straight line between data and noise.

$$
x_\sigma = (1-\sigma)x_0 + \sigma\epsilon, \qquad
v_{\mathrm{target}} = \epsilon - x_0.
$$

The generator predicts this velocity from the noisy grid, the prompt, and the noise level, minimizing mean squared error. At inference, `generate_samples` starts at $\sigma=1$ and takes Euler steps down to $\sigma=0$. Classifier-free guidance combines the conditional and empty-prompt velocity predictions before each step. The VAE decodes the final grid to RGB. [Training objective and sampler](networks/model_mot.py)

## Training and evaluation

The data pipeline draws from five domains spanning anime, MONET-real, GPIC, synthetic images, and TextAtlas. The multi-aspect-ratio recipe groups images by shape and fixes domain counts in an offline batch plan. This gives architecture comparisons a reproducible mix of training examples across distributed workers. Full training requires external model weights, image data, and the catalogs and plans referenced by the recipe. Sampling trained images also requires a matching Motif checkpoint. [Offline batch planner](docs/OFFLINE_ARB_PLANNER_DESIGN.md) · [Training entry point](train.py)

The anime evaluation measures how often an image tagger recovers requested attributes and character identities. CLIP scores measure overall image–text alignment. DPG-Bench probes a different failure mode by breaking a prompt into yes/no questions about the generated scene. Together these tests let researchers compare what each conditioning design preserves. [Anime scoring](experiments/eval/run_prompt_following.py) · [DPG-Bench protocol](experiments/eval/dpg/README.md)

To inspect a single training step, start with the tiny CPU configuration. It uses a two-layer model with random weights and synthetic latents. With PyTorch and the repository dependencies installed, run the following on a compute worker with access to the Qwen tokenizer. [Dependencies](env/requirements.txt) · [Tiny configuration](configs/debug_tiny_cpu.yaml)

```bash
python train.py --config configs/debug_tiny_cpu.yaml --max-steps 1 \
  model.tokenizer_path=Qwen/Qwen3-VL-2B-Instruct trainer.save_final=false
```
