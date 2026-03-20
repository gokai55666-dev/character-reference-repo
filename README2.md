# Install dependencies
pip install -r requirements.txt

# Prepare your dataset
python scripts/prepare_dataset.py --input assets/original_references --output dataset/train

# Train LoRA
python scripts/train_lora.py --config configs/lora_config.json

# Test consistency
python benchmarks/consistency_test.py --model models/character_lora_v1.safetensors

If you want to run everything in one go (after adding images):

```bash
# Navigate to repo
cd /path/to/character-reference-repo

# Install dependencies (once)
pip install -r requirements.txt

# Prepare dataset
python scripts/prepare_dataset.py --input assets/original_references --output dataset/train

# Edit config with your character name
nano configs/lora_config.json
# Change "model_name": "your_character_lora"

# Train (this takes time)
python scripts/train_lora.py --config configs/lora_config.json

# Find your model file
ls models/

# Test consistency (replace filename with your actual model)
python benchmarks/consistency_test.py --model models/your_character_lora_step_1000.safetensors

# View report
open benchmarks/test_results/consistency_report.html
```

---

💡 Important Notes

1. GPU Required: Training requires a GPU with at least 8GB VRAM. If you don't have one, use Google Colab.
2. Python Environment: Make sure you're using Python 3.10+:
   ```bash
   python --version  # Should be 3.10 or higher
   ```
3. Virtual Environment (Recommended):
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   source venv/bin/activate  # On Mac/Linux
   # or
   venv\Scripts\activate  # On Windows
   
   # Then install requirements
   pip install -r requirements.txt
   ```
4. If You Get Errors:
   · Make sure you're in the correct directory
   · Check that all files exist (use ls to verify)
   · Ensure Python packages installed correctly
   · Check GPU availability: nvidia-smi (should show your GPU)

---

📝 Example Output When Running

When you run the consistency test, you'll see:

```
==================================================
Starting Consistency Test Suite
==================================================

Step 1: Generating test images...
Generating consistency test grid for model: models/your_character_lora.safetensors
Character features: {'eye_color': 'emerald green', 'hair_color': 'chestnut', ...}
Total test cases: 100
  Generating case 1/10: {'lighting': 'bright daylight', 'style': 'photorealistic', ...}
  ...

Step 2: Analyzing consistency...
  Analyzing 10 images...

Step 3: Generating report...
Report generated: benchmarks/test_results/consistency_report.html

==================================================
Test Summary
==================================================
Total test cases: 100
Images analyzed: 10
Overall score: 85.3%

✅ Excellent consistency! Character identity is well preserved.
```

Then open the HTML report to see detailed results with charts and metrics!
