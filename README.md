# GenBIB-ML

Use the collection-specific TabDDPM models from Paper 1 to generate detector hits.

1. Clone this repository and enter the project directory.

   ```bash
   git clone https://github.com/ShiyuP1/GenBIB-ML.git
   cd GenBIB-ML
   ```

2. Create the environment 

   ```bash
   conda create -n genbib python=3.9.7 -y
   conda activate genbib
   python -m pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   python -m pip install -r /oscar/data/mleblan6/mucoll/speng44/bib_gen_model/ddpm_outputs/tabddpm/local_phi/paper1-inference/diffusion/tabddpm_official/requirements.txt
   python -m pip install matplotlib==3.5.0
   ```

3. Choose the model directory for the collection you want to sample.

   The models are stored in `/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/ddpm_outputs/tabddpm/local_phi`.

   The collection names are `VBC`, `VEC`, `ITBC`, `ITEC`, `OTBC`, and `OTEC`.

4. Save the conditions as an integer NumPy array with shape `(N, 5)`. The columns are `system_id, side, layer, module, sensor`.

   Collection IDs are `1=VBC`, `2=VEC`, `3=ITBC`, `4=ITEC`, `5=OTBC`, and `6=OTEC`.

   ```python
   import numpy as np

   conditions = np.array([
       [3, 0, 1, 12, 0],
       [3, 0, 1, 13, 0],
   ], dtype=np.int64)

   np.save("conditions.npy", conditions)
   ```

   Each row must be present in the selected model's `y_lookup.npy`. Repeating a row requests another sample with the same condition. Set `CONDITIONS_FILE = None` to test the first 16 entries in `y_lookup.npy`.

5. Edit the variables at the top of `sample.py`. Replace `MODEL_DIRECTORY` with the directory selected in step 3.

   ```python
   COLLECTION = "ITBC"
   PAPER1_CODE_ROOT = Path("/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/ddpm_outputs/tabddpm/local_phi/paper1-inference")
   MODEL_DIR = Path("/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/ddpm_outputs/tabddpm/local_phi/MODEL_DIRECTORY")
   CONDITIONS_FILE = Path("conditions.npy")
   OUTPUT_FILE = Path("generated_samples.npy")
   ```

6. Run the sampling script.

   ```bash
   conda activate genbib
   python sample.py
   ```

   The output columns are `logE, time, r, phi, z, side, layer, module, sensor`.

