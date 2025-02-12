# NGP_Lenticular_Reconstruction

### A fun little demo of a lenticular effect using NeRF.

By manipulating the dataloader and rays differently during training and testing in the [NGP repo](https://github.com/HrushikeshBudhale/NGP_from_scratch), we can reconstruct these interesting looking lenticular effects.

Inspired from repo [gaussian-painters](https://github.com/ReshotAI/gaussian-painters).

## Results

![Avengers reconstruction](results/recon_output_avengers.gif)

![DC weather reconstruction](results/recon_output_DC_weather.gif)

## Setup

1. Clone the repo

    ```bash
    git clone https://github.com/HrushikeshBudhale/NGP_lenticular_recon.git
    ```

2. Install supported version of pytorch

3. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

4. Add any 4 images to new folder (with roughtly same resolution) and update the `conf.yaml` file with that folder's path.

5. Run the script

    ```bash
    python train_nerf.py
    python test_nerf.py
    ```

6. It will save the resulting gif in current working directory.

Experiment with `ImageMagicDataloader` to get the desired effect for your images.
