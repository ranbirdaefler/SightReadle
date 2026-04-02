# SCMPA: Score-Conditioned Music Performance Assessment

A system that assesses music performance quality by comparing learned embeddings from a music foundation model (MERT). Takes a musical score (MusicXML/MIDI) and an audio recording, outputs quality scores for rhythm, pitch, completeness, and flow.

## Core Idea

Synthetic score-conditioned supervision induces an embedding space where cosine similarity is a reliable proxy for human-perceived performance quality — without any human annotations in training.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# System dependency (for MIDI-to-audio synthesis)
# Ubuntu: sudo apt-get install fluidsynth fluid-soundfont-gm
# Windows: install FluidSynth from https://github.com/FluidSynth/fluidsynth/releases

# Run feasibility test FIRST
python scripts/00_feasibility_test.py
```

## Project Structure

```
scmpa/
├── configs/           Hyperparameter configs (YAML)
├── src/data/          Score parsing, synthesis, degradation, augmentation, dataset
├── src/model/         MERT backbone, scoring head, loss functions, full model
├── src/evaluation/    Proxy metrics, correlation analysis, layer probing
├── src/utils/         Audio utilities, visualization
├── scripts/           Numbered execution scripts (run in order)
├── data/              Raw scores, augmentation assets, synthetic + real data
├── results/           Figures, ablation results, evaluation outputs
└── tests/             Unit tests
```

## Execution Order

```
Phase 0: python scripts/00_feasibility_test.py   (CHECK RESULTS before continuing)
Phase 1: python scripts/01_download_data.py
         python scripts/02_generate_synthetic.py
Phase 2: (model implementation — no script)
Phase 3: python scripts/05_train.py --config configs/default.yaml
         python scripts/07_ablation.py
Phase 4: python scripts/06_evaluate.py
```
