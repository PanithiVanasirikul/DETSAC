# DETSAC: Query-Guided Class-Aware Multi-Model Fitting

We proposed DETSAC, a multi-model fitting approach that uses object queries to communicate with each other and guide each points that will be sampled to the correct classes before sampling with the ability to determine if each object query is assigned to a class. This leads to a more computationally efficient sampling of geometric models.

## Installation
Get the code:
```
git clone --recurse-submodules https://github.com/PanithiVanasirikul/DETSAC.git
cd DETSAC
git submodule update --init --recursive
```

Set up the Python environment using [Anaconda](https://www.anaconda.com/): 
```
conda env create -f environment.yml
source activate detsac
```

## Datasets

Please follow the instructions in the [PARSAC](https://github.com/fkluger/parsac/) repository to download and prepare the datasets.




## Training
### Vanishing Points
- SU3
    ```sh
    python detsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset su3 --problem vp --instances 8 --hypsamples 64 --data_path datasets/su3 --checkpoint_dir ./tmp/checkpoints --no_refine --network_layers 3 --ckpt_mode all --augment
    ```
- YUD
    ```sh
    python detsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset yud --problem vp --instances 8 --hypsamples 64 --data_path datasets/yud_plus/data --checkpoint_dir ./tmp/checkpoints --no_refine --network_layers 3 --ckpt_mode all --augment
    ```
- NYU-VP
    ```sh
    python detsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset nyuvp --problem vp --instances 8 --hypsamples 64 --data_path datasets/nyu_vp/data --checkpoint_dir ./tmp/checkpoints --no_refine --ckpt_mode all --augment
    ```
- YUD+
    ```sh
    python detsac.py --hypotheses 32 --batch 64 --samplecount 8 --inlier_threshold 0.0001 --dataset yudplus --problem vp --instances 8 --hypsamples 64 --data_path datasets/yud_plus/data --checkpoint_dir ./tmp/checkpoints --no_refine --network_layers 3 --ckpt_mode all --augment
    ```

### Fundamental Matrices
- HOPE-F
    ```sh
    python detsac.py --hypotheses 32 --batch 32 --samplecount 16 --inlier_threshold 0.004 --assignment_threshold 0.02 --dataset hope --problem fundamental --instances 4 --hypsamples 128 --epochs 3000 --lr_steps 2500 --data_path datasets/hope --checkpoint_dir ./tmp/checkpoints --encoder_layers 3 --decoder_layers 3 --ckpt_mode all --augment
    ```
- Adelaide
    ```sh
    python detsac.py --hypotheses 128 --batch 8 --samplecount 16 --inlier_threshold 0.01 --assignment_threshold 0.02 --dataset adelaide --problem fundamental --instances 4 --hypsamples 128 --epochs 500 --lr_steps 350 --data_path datasets/adelaide --checkpoint_dir ./tmp/checkpoints --encoder_layers 3 --decoder_layers 3 --ckpt_mode all --augment
    ```

### Homographies
- SMH
    ```sh
    python detsac.py --hypotheses 64 --batch 4 --samplecount 8 --inlier_threshold 1e-6 --assignment_threshold 4e-6 --dataset smh --problem homography --instances 24 --hypsamples 64 --epochs 500 --lr_steps 350 --data_path datasets/smh --checkpoint_dir ./tmp/checkpoints --network_layers 3 --ckpt_mode all --augment
    ```
- Adelaide
    ```sh
    python detsac.py --hypotheses 512 --batch 4 --samplecount 8 --inlier_threshold 1e-4 --assignment_threshold 4e-3 --dataset adelaide --problem homography --instances 24 --hypsamples 512 --epochs 500 --lr_steps 350 --data_path datasets/adelaide --checkpoint_dir ./tmp/checkpoints --network_layers 3 --ckpt_mode all --augment
    ```



## Evaluation
### Vanishing points
- SU3
    ```sh
    python detsac.py --eval --dataset su3 --data_path datasets/su3 --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32
    ```
- YUD
    ```sh
    python detsac.py --eval --dataset yud --data_path datasets/yud_plus/data --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32
    ```
- NYU-VP
    ```sh
    python detsac.py --eval --dataset nyuvp --data_path datasets/nyu_vp/data --problem vp --load weights/main_results/vp_nyu --inlier_threshold 0.0001 --instances 8 --hypotheses 32
    ```
- YUD+
    ```sh
    python detsac.py --eval --dataset yudplus --data_path datasets/yud_plus/data --problem vp --load weights/main_results/vp_su3 --inlier_threshold 0.0001 --instances 8 --hypotheses 32
    ```

### Fundamental Matrices
- HOPE-F
    ```sh
    python detsac.py --eval --load /mnt/ssd2/se3_to_image/related_repos/parsac/tmp/checkpoints/stellar-yogurt-456/ --dataset hope --data_path datasets/hope --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128
    ```
- Adelaide
    ```sh
    python detsac.py --eval --load ./weights/main_results/fundamental --dataset adelaide --data_path ./datasets/adelaide --problem fundamental --inlier_threshold 0.01 --assignment_threshold 0.02 --instances 4 --hypotheses 128
    ```

### Homographies
- SMH
    ```sh
    python detsac.py --eval --load weights/main_results/homography --dataset smh --data_path datasets/smh --problem homography --inlier_threshold 1e-6 --assignment_threshold 4e-6 --instances 24 --hypotheses 512
    ```
- Adelaide
    ```sh
    python detsac.py --eval --load weights/main_results/homography --dataset adelaide --data_path datasets/adelaide --problem homography --inlier_threshold 1e-4 --assignment_threshold 4e-3 --instances 24 --hypotheses 512
    ```

## Special Thanks

We would like to thank [PARSAC](https://github.com/fkluger/parsac/) for the code base and dataset preparation steps used in this project.

