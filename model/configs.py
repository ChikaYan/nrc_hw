import gin
import dataclasses
import argparse

@gin.configurable
@dataclasses.dataclass
class NrcConfig():
    '''
    NRC neural network configurations
    '''
    # Number of rays used to train NRC
    training_samples: int = 4096 * 4 * 4
    # TODO: not sure what this does
    reflectance_factorisation: bool = False
    # Train all rays for first frame
    first_frame_training: bool = True
    # 
    activate_nrc: bool = True
    # Learning rate for network optimizer
    optim_lr: float = 5e-3
    # Number of training steps per frame
    steps_per_frame: int = 4


@gin.configurable
@dataclasses.dataclass
class RenderConfig():
    '''
    Rendering configurations
    '''
    # 
    short_depth: int = 2
    # 
    long_depth: int = 4
    # Path to rendering scene
    scene_path: str = None
    # Sec 3.4 area threshold for path termination 
    path_termination_c: float = 0.01
    # Visualise primary vertex with NRC
    visualise_primary: bool = False
    # only return nrc predicted rendering
    output_only_nrc: bool = False
    # Log dir
    log_dir: str = None
    # Sample (ray) per pixel
    spp: int = 1



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default=None)

    return parser.parse_args()
