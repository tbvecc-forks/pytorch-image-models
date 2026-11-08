# Learning plan: timm (`pytorch-image-models`)

A study plan for timm's Vision Transformer stack, written for someone who has already worked through [vit-pytorch](https://github.com/tbvecc-forks/vit-pytorch). That means the [study plan in its ANALYSIS.md](https://github.com/tbvecc-forks/vit-pytorch/blob/main/ANALYSIS.md#12-study-plan), the CIFAR-10 baseline, the pre-norm vs post-norm ablation, and the attention-map and residual-stream analyses.

**What timm is:**
- the PyTorch *production* library for vision models: about 1,380 registered model architectures and about 1,875 pretrained weights on the Hugging Face Hub;
- one configurable `VisionTransformer` class that covers most ViT papers;
- a full training script (`train.py`) and a large test suite.

**Where vit-pytorch fits:** vit-pytorch shows *how a paper's idea looks in code*. timm shows how those ideas become a **reusable, tested, weight-compatible library**.

Each stage has **Read** (files to study, with line numbers at the time of writing), **Do** (hands-on work on one GPU) and a **Checkpoint** (what you should be able to explain). Papers for each stage are listed at the end.

---

## Stage 0: setup (½ session)

1. Install timm into the existing vit-pytorch environment, so your CIFAR-10 tooling keeps working:
   ```bash
   ~/vit-pytorch/.venv/bin/pip install -e ~/pytorch-image-models
   ~/vit-pytorch/.venv/bin/python -c "import timm; print(timm.__version__, len(timm.list_models()), len(timm.list_pretrained()))"
   ```
2. Run the tests for one tiny model, to see how the suite is organised:
   ```bash
   cd ~/pytorch-image-models && ~/vit-pytorch/.venv/bin/python -m pytest tests/test_models.py -k "test_vit" -q
   ```
3. Skim [`AGENTS.md`](AGENTS.md) for the project's coding conventions, and [`hfdocs/source/models.mdx`](hfdocs/source/models.mdx) for the model index.

> ✅ **Checkpoint:** how many models and pretrained tags does your install list, and how many of them are `vit_*`?

## Stage 1: the model API, registry and pretrained configs (1 session)

**Read**
- [`timm/models/_registry.py:114`](timm/models/_registry.py) `register_model`: how a function becomes a model name, and how `default_cfgs` become `arch.tag` entries.
- [`timm/models/_factory.py:162`](timm/models/_factory.py) `create_model`: the `hf-hub:` and `local-dir:` prefixes, `pretrained_cfg_overlay`, `checkpoint_path`.
- [`timm/models/_builder.py:427`](timm/models/_builder.py) `build_model_with_cfg`: config resolution and `features_only` wrapping.
- [`timm/models/vision_transformer.py:2047`](timm/models/vision_transformer.py) `default_cfgs`, and the entry point `vit_base_patch16_224` at `:3493`.

**Do**
1. Call `timm.list_pretrained('vit_base_patch16_224*')`. Decode three tags, for example `augreg2_in21k_ft_in1k`: what each part says about the data, recipe and fine-tuning.
2. Call `timm.create_model('test_vit', pretrained=False)`. Print `model.pretrained_cfg` and `model.default_cfg`, then compare `num_classes=10` with `num_classes=0`.
3. Compare `forward_features(x)` with `forward_head(..., pre_logits=True)` and `forward_head(...)`. Where does pooling happen?

> ✅ **Checkpoint:** explain the path from `timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True)` to a model loaded with weights, naming each function involved.

## Stage 2: `VisionTransformer` in depth, against vit-pytorch (2 sessions)

**Read**: [`timm/models/vision_transformer.py`](timm/models/vision_transformer.py)
- `VisionTransformer` at `:749`. Read every constructor argument: `global_pool`, `class_token`, `reg_tokens`, `no_embed_class`, `pre_norm`, `fc_norm`, `qk_norm`, `init_values`, `drop_path_rate`, `dynamic_img_size`, `weight_init`, `block_fn`.
- `Block` at `:166` (pre-norm) and `ResPostBlock` at `:255` (post-norm, used by `vit_base_patch16_rpn_224` at `:4028`).
- `_pos_embed` at `:1088`: prefix tokens, and "concat then add" vs "add then concat" (`no_embed_class`).
- `init_weights_vit_jax` at `:1373`, and the default timm initialization.
- Shared layers:
  - [`timm/layers/attention.py:43`](timm/layers/attention.py) `Attention`: fused `scaled_dot_product_attention` vs the explicit softmax path;
  - [`timm/layers/patch_embed.py:26`](timm/layers/patch_embed.py) `PatchEmbed`: a strided conv;
  - [`timm/layers/layer_scale.py:5`](timm/layers/layer_scale.py) `LayerScale`;
  - [`timm/layers/drop.py:178`](timm/layers/drop.py) `DropPath`;
  - [`timm/layers/attention_pool.py:13`](timm/layers/attention_pool.py) `AttentionPoolLatent` (the `'map'` head).

**Do**
1. **Rebuild your CIFAR-10 model in timm:** `VisionTransformer(img_size=32, patch_size=4, embed_dim=256, depth=6, num_heads=8, mlp_ratio=2, num_classes=10)`. Compare its parameter count with `make_vit()` from vit-pytorch's [`train_cifar10.py`](https://github.com/tbvecc-forks/vit-pytorch/blob/main/train_cifar10.py) and explain every difference. Hints: q/k/v biases, and vit-pytorch's two patch LayerNorms.
2. **Port your trained weights.** Load `runs/models/pre.pt` (from vit-pytorch) into the timm model:
   - reshape the patch `Linear` weight into the `Conv2d` kernel;
   - concatenate `to_qkv`;
   - deal with the patch LayerNorms, either by setting them to identity or with `pre_norm` and `embed_norm_layer`.

   Check that the logits match to about 1e-5. This exercise is the best way to make every structural difference concrete.
3. **Repeat your pre-norm vs post-norm ablation in timm,** building the post-norm model in Python with `block_fn=ResPostBlock`. `--model-kwargs` on the command line only takes literal values, not classes. Do you reproduce post-norm's advantage?

> ✅ **Checkpoint:** list every architectural difference between timm's `VisionTransformer` and vit-pytorch's `vit.ViT`: patch embedding, initialization, biases, position handling, pooling options, norms.

## Stage 3: pretrained weights, conversion and resolution changes (1–2 sessions)

**Read**
- [`timm/models/_builder.py:158`](timm/models/_builder.py) `load_pretrained`: `filter_fn`, `in_chans` adaptation, classifier reset.
- [`timm/models/vision_transformer.py:1931`](timm/models/vision_transformer.py) `checkpoint_filter_fn`: converting CLIP, DINOv2 and other formats, and resampling patch and position embeddings.
- `_load_weights` at `:1472`: loading **JAX `.npz`** checkpoints from `big_vision` and `vision_transformer`, including the HWIO→OIHW and `[in,out]`→`[out,in]` transposes.
- [`timm/layers/pos_embed.py:20`](timm/layers/pos_embed.py) `resample_abs_pos_embed`: the proper fix for vit-pytorch's `pos_embedding[:seq]` slicing.

**Do**
1. Create `vit_small_patch16_224.augreg_in21k_ft_in1k` with `img_size=384`. Watch the position embedding go from 14×14 to 24×24, and compare with what `[:seq]` would have done.
2. **Position-embedding similarity plot** (ViT paper, Fig. 7): reshape `pos_embed[0, 1:]` to 14×14×D and plot the cosine similarity of each position to all others. Compare with your sin-cos analysis from vit-pytorch.
3. **Fine-tune on CIFAR-10** (upsampled to 224) from `augreg_in21k`. Compare with your from-scratch 80–82%, and note how few epochs it takes.

> ✅ **Checkpoint:** explain what `checkpoint_filter_fn` must do to load a 224-px JAX checkpoint into a 384-px timm model with 10 classes.

## Stage 4: inspecting pretrained models (1 session)

**Read**
- [`timm/models/vision_transformer.py:1127`](timm/models/vision_transformer.py) `forward_intermediates`: returns the residual stream at every block, **un-normalized by default**.
- [`timm/utils/attention_extract.py:9`](timm/utils/attention_extract.py) `AttentionExtract`.
- [`timm/layers/config.py`](timm/layers/config.py) `TIMM_FUSED_ATTN`: attention maps only exist when fused attention is **off**.
- [`hfdocs/source/feature_extraction.mdx`](hfdocs/source/feature_extraction.mdx).

**Do**
1. Rerun your **residual-stream analysis** (`inspect_residual_stream.py` in the vit-pytorch fork) on pretrained `vit_small_patch16_224.augreg_in21k_ft_in1k`, using `forward_intermediates`. Does the norm grow with depth like your pre-norm model's?
2. Compare with a register model, `vit_base_patch16_reg4_gap_256`: find the **high-norm "artifact" tokens** in the model without registers, and check whether registers absorb them (*Vision Transformers Need Registers*).
3. Rerun your **cls attention heat-maps** on a pretrained model with `TIMM_FUSED_ATTN=0`. Compare their entropy and center bias with your CIFAR-10 models.

> ✅ **Checkpoint:** why does attention extraction need fused attention off, and why does `forward_intermediates` default to the un-normalized stream?

## Stage 5: training infrastructure (2 sessions)

**Read**
- [`train.py`](train.py): skim the argument groups, then follow one training step.
- Optimizer:
  - [`timm/optim/_optim_factory.py:1242`](timm/optim/_optim_factory.py) `create_optimizer_v2`;
  - [`timm/optim/_param_groups.py:19`](timm/optim/_param_groups.py) `param_groups_weight_decay`: 1-D parameters plus `no_weight_decay()`;
  - `param_groups_layer_decay` at `:113`: layer-wise learning-rate decay through `group_matcher`.
- [`timm/scheduler/`](timm/scheduler/): cosine with warmup, WSD.
- Data:
  - [`timm/data/transforms_factory.py:387`](timm/data/transforms_factory.py) `create_transform`;
  - [`timm/data/auto_augment.py:762`](timm/data/auto_augment.py) `rand_augment_transform`;
  - [`timm/data/mixup.py:96`](timm/data/mixup.py) `Mixup`.
- [`timm/utils/model_ema.py:178`](timm/utils/model_ema.py) `ModelEmaV3`.
- [`hfdocs/source/hparams.mdx`](hfdocs/source/hparams.mdx): the named recipes (A1/A2/A3 from *ResNet strikes back*).

**Do**
1. **Weight-decay groups:** call `create_optimizer_v2(model, 'adamw', lr=1e-3, weight_decay=0.05)` and print each group's parameter names. Do they match the groups you built by hand in `train_cifar10.py`?
2. **Train on CIFAR-10 with `train.py`:**
   ```bash
   python train.py --dataset torch/cifar10 --dataset-download --data-dir ~/data --model test_vit --img-size 32 \
       --model-kwargs patch_size=4 --num-classes 10 --epochs 30 --batch-size 256 --lr 1e-3 --opt adamw --weight-decay 0.05 --warmup-epochs 2
   ```
3. **Ablate one recipe component per run**, holding the rest fixed: `--mixup 0.8 --cutmix 1.0`, `--aa rand-m9-mstd0.5`, `--model-ema`, `--drop-path 0.1`, `--layer-decay 0.75` (fine-tuning only). Use your vit-pytorch `plot_ablation.py` to compare the curves.

> ✅ **Checkpoint:** explain layer-wise learning-rate decay and why it helps fine-tuning but not training from scratch.

## Stage 6: the ViT family and contributing (1–2 sessions)

**Read**
- [`timm/models/deit.py:28`](timm/models/deit.py) `VisionTransformerDistilled`: extends `VisionTransformer` by subclassing, not copy-paste.
- [`timm/models/eva.py`](timm/models/eva.py): RoPE and SwiGLU in a production model.
- [`timm/models/naflexvit.py:1167`](timm/models/naflexvit.py) `NaFlexVit`: NaViT-style variable resolution, compared with vit-pytorch's `na_vit.py`.
- [`timm/models/beit.py`](timm/models/beit.py), [`timm/models/cait.py`](timm/models/cait.py): families that keep their own blocks.
- [`tests/test_models.py:221`](tests/test_models.py) `test_model_backward` (every parameter must get a gradient), and `test_model_forward_intermediates` at `:786`.

**Do**
1. Register your own model with `@register_model`, for example a post-norm ViT or a ViT with value residuals ported from vit-pytorch. Make it pass the forward, backward and default-config tests with `pytest -k <name>`.
2. Compare vit-pytorch's `cait.py` with timm's. What did timm need to add to make it weight-compatible and testable?

> ✅ **Checkpoint:** when would you choose timm, and when vit-pytorch, for (a) a quick research idea, (b) a fine-tuned production model, (c) reproducing a paper's numbers?

---

## Reading list

★ = core reading for the plan. Every link was checked against arXiv. "cited in" names the timm file that references the paper.

### Foundations and training recipes (Stages 1, 5)

| | Paper | Link | In timm |
|---|---|---|---|
| ★ | *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT) | [arXiv 2010.11929](https://arxiv.org/abs/2010.11929) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| ★ | *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers* (AugReg: the `augreg` tags) | [arXiv 2106.10270](https://arxiv.org/abs/2106.10270) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| ★ | *ResNet strikes back: An improved training procedure in timm* | [arXiv 2110.00476](https://arxiv.org/abs/2110.00476) | [`hparams.mdx`](hfdocs/source/hparams.mdx) |
| ★ | *Training data-efficient image transformers & distillation through attention* (DeiT) | [arXiv 2012.12877](https://arxiv.org/abs/2012.12877) | [`deit.py`](timm/models/deit.py) |
| | *DeiT III: Revenge of the ViT* | [arXiv 2204.07118](https://arxiv.org/abs/2204.07118) | [`deit.py`](timm/models/deit.py), [`auto_augment.py`](timm/data/auto_augment.py) |
| | *ImageNet-21K Pretraining for the Masses* | [arXiv 2104.10972](https://arxiv.org/abs/2104.10972) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Decoupled Weight Decay Regularization* (AdamW) | [arXiv 1711.05101](https://arxiv.org/abs/1711.05101) | [`timm/optim/`](timm/optim/) |
| | *SGDR: Stochastic Gradient Descent with Warm Restarts* (cosine schedule) | [arXiv 1608.03983](https://arxiv.org/abs/1608.03983) | [`cosine_lr.py`](timm/scheduler/cosine_lr.py) |
| | *Deep Networks with Stochastic Depth* (DropPath) | [arXiv 1603.09382](https://arxiv.org/abs/1603.09382) | [`drop.py`](timm/layers/drop.py) |
| | *mixup: Beyond Empirical Risk Minimization* | [arXiv 1710.09412](https://arxiv.org/abs/1710.09412) | [`mixup.py`](timm/data/mixup.py) |
| | *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features* | [arXiv 1905.04899](https://arxiv.org/abs/1905.04899) | [`mixup.py`](timm/data/mixup.py) |
| | *RandAugment: Practical automated data augmentation with a reduced search space* | [arXiv 1909.13719](https://arxiv.org/abs/1909.13719) | [`auto_augment.py`](timm/data/auto_augment.py) |
| | *AutoAugment* / *AugMix* | [arXiv 1805.09501](https://arxiv.org/abs/1805.09501) / [arXiv 1912.02781](https://arxiv.org/abs/1912.02781) | [`auto_augment.py`](timm/data/auto_augment.py) |
| | *Random Erasing Data Augmentation* | [arXiv 1708.04896](https://arxiv.org/abs/1708.04896) | [`random_erasing.py`](timm/data/random_erasing.py) |

### Architecture options in `VisionTransformer` (Stage 2)

| | Paper | Link | In timm |
|---|---|---|---|
| ★ | *Going deeper with Image Transformers* (CaiT: LayerScale) | [arXiv 2103.17239](https://arxiv.org/abs/2103.17239) | [`layer_scale.py`](timm/layers/layer_scale.py), [`cait.py`](timm/models/cait.py) |
| ★ | *Vision Transformers Need Registers* | [arXiv 2309.16588](https://arxiv.org/abs/2309.16588) | `reg_tokens` in [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Scaling Vision Transformers to 22 Billion Parameters* (qk-norm, parallel blocks) | [arXiv 2302.05442](https://arxiv.org/abs/2302.05442) | `qk_norm` in [`attention.py`](timm/layers/attention.py) |
| | *Three things everyone should know about Vision Transformers* (parallel blocks) | [arXiv 2203.09795](https://arxiv.org/abs/2203.09795) | `ParallelThingsBlock` in [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Talking-Heads Attention* | [arXiv 2003.02436](https://arxiv.org/abs/2003.02436) | [`cait.py`](timm/models/cait.py) |
| | *Differential Transformer* | [arXiv 2410.05258](https://arxiv.org/abs/2410.05258) | `DiffParallelScalingBlock` in [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Rotary Position Embedding for Vision Transformer* | [arXiv 2403.13298](https://arxiv.org/abs/2403.13298) | [`eva.py`](timm/models/eva.py) |
| | *Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution* | [arXiv 2307.06304](https://arxiv.org/abs/2307.06304) | [`naflexvit.py`](timm/models/naflexvit.py) |
| | *FlexiViT: One Model for All Patch Sizes* (patch-embedding resampling) | [arXiv 2212.08013](https://arxiv.org/abs/2212.08013) | [`vision_transformer.py`](timm/models/vision_transformer.py), [`naflexvit.py`](timm/models/naflexvit.py) |
| | *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design* (SoViT-400m) | [arXiv 2305.13035](https://arxiv.org/abs/2305.13035) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *On Layer Normalization in the Transformer Architecture* (pre- vs post-LN) | [arXiv 2002.04745](https://arxiv.org/abs/2002.04745) | `Block` vs `ResPostBlock` |

### Pretrained weight families (Stages 3–4, 6)

| | Paper | Link | In timm |
|---|---|---|---|
| ★ | *Learning Transferable Visual Models From Natural Language Supervision* (CLIP) | [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| ★ | *Sigmoid Loss for Language Image Pre-Training* (SigLIP: the `'map'` head) | [arXiv 2303.15343](https://arxiv.org/abs/2303.15343) | [`attention_pool.py`](timm/layers/attention_pool.py) |
| | *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* | [arXiv 2502.14786](https://arxiv.org/abs/2502.14786) | [`vision_transformer.py`](timm/models/vision_transformer.py), [`naflexvit.py`](timm/models/naflexvit.py) |
| ★ | *DINOv2: Learning Robust Visual Features without Supervision* | [arXiv 2304.07193](https://arxiv.org/abs/2304.07193) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Emerging Properties in Self-Supervised Vision Transformers* (DINO) | [arXiv 2104.14294](https://arxiv.org/abs/2104.14294) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Masked Autoencoders Are Scalable Vision Learners* (MAE) | [arXiv 2111.06377](https://arxiv.org/abs/2111.06377) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA) | [arXiv 2301.08243](https://arxiv.org/abs/2301.08243) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *BEiT: BERT Pre-Training of Image Transformers* / *BEiT v2* | [arXiv 2106.08254](https://arxiv.org/abs/2106.08254) / [arXiv 2208.06366](https://arxiv.org/abs/2208.06366) | [`beit.py`](timm/models/beit.py) |
| | *EVA: Exploring the Limits of Masked Visual Representation Learning at Scale* / *EVA-02* | [arXiv 2211.07636](https://arxiv.org/abs/2211.07636) / [arXiv 2303.11331](https://arxiv.org/abs/2303.11331) | [`eva.py`](timm/models/eva.py) |
| | *Multimodal Autoregressive Pre-training of Large Vision Encoders* (AIMv2) | [arXiv 2411.14402](https://arxiv.org/abs/2411.14402) | [`vision_transformer.py`](timm/models/vision_transformer.py) |
| | *Swin Transformer V2: Scaling Up Capacity and Resolution* (log-spaced relative position) | [arXiv 2111.09883](https://arxiv.org/abs/2111.09883) | [`vision_transformer_relpos.py`](timm/models/vision_transformer_relpos.py) |
| | *Scaling Vision Transformers* | [arXiv 2106.04560](https://arxiv.org/abs/2106.04560) | [`vision_transformer.py`](timm/models/vision_transformer.py) |

---

## Gotchas collected while analyzing

- **Attention maps need `TIMM_FUSED_ATTN=0`** (or `attn.fused_attn = False`). `scaled_dot_product_attention` never materializes the softmax matrix.
- **`forward_intermediates` returns the un-normalized residual stream** unless you pass `norm=True`.
- **`--model-kwargs` only accepts literal values,** so architecture classes like `block_fn` must be set in Python.
- **`no_weight_decay()` is declared by the model.** That's how 2-D and 3-D parameters such as `pos_embed` escape weight decay, since the 1-D rule wouldn't catch them.
