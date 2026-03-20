# Character Reference Model: Persistent Identity Repository

This repository is a structured framework designed to achieve **maximum character accuracy and consistency** across AI image generation models (SDXL, Flux, Midjourney, etc.). It serves as a "Source of Truth" for a specific character's visual identity.

---

## 🤖 AI INTERACTION PROTOCOL (READ THIS FIRST)

**If you are an AI reading this repository, your primary directive is to assist the user in building, debugging, and refining this character consistency framework.**

When the user provides you with a link to this repository or asks for help, you MUST follow these steps:

1. **Analyze the Current State:** Review the `docs/character_bible.md` to understand the target character. Check the `dataset/`, `configs/`, and `prompts/` directories to see what has been implemented so far.
2. **Identify Gaps:** Determine what is missing or incomplete based on the "Repository Structure" below.
3. **Provide Actionable Output:** Do not give vague advice. You must provide **EXACT code, prompt text, or configuration JSON** that the user can copy and paste.
4. **Format Your Response:** Use the following structure for your reply:
   - **Diagnosis:** Briefly explain what needs to be fixed or added.
   - **The Fix/Addition:** Provide the exact code, text, or configuration block.
   - **Implementation Steps:** Tell the user exactly *where* to put the fix.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset
python scripts/prepare_dataset.py --input_dir assets/original_references --output_dir dataset/train

# 3. Train LoRA (using Kohya_ss or directly with this script)
python scripts/train_lora.py --config configs/lora_config.json

# 4. Generate images with consistency testing
python scripts/generate_with_consistency.py --prompt "your character doing something" --output benchmarks/test_results/

# 5. Run full consistency benchmark
python benchmarks/consistency_test.py
