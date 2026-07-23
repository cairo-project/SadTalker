"""Installable library API for SadTalker.

Provides a simple ``load_models`` / ``generate`` interface on top of the
original ``inference.py`` pipeline (CropAndExtract -> Audio2Coeff ->
AnimateFromCoeff), plus a ``main`` CLI entry point.

Example
-------
    from generate_sadtalker import load_models, generate
    models = load_models(checkpoint_dir="checkpoints", device="cuda")
    out = generate(models, "face.png", "speech.wav", "out.mp4")
"""

import os
import sys
import shutil
from argparse import ArgumentParser
from time import strftime

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


# Directory of this module; used to locate the bundled ``src/config`` folder
# regardless of the current working directory (unlike inference.py which relies
# on ``os.path.split(sys.argv[0])[0]``).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_device(device):
    """Return a valid torch device string, falling back to CPU when needed."""
    try:
        import torch
    except ImportError:
        return device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        return "cpu"
    return device


def load_models(checkpoint_dir="checkpoints", device="cuda", size=256,
                preprocess="crop", old_version=False, **_):
    """Load the three SadTalker model stages.

    Parameters
    ----------
    checkpoint_dir : str
        Path to the directory containing SadTalker checkpoints.
    device : str
        Torch device, e.g. ``"cuda"`` or ``"cpu"``. Falls back to CPU if CUDA
        is unavailable.
    size : int
        Face render image size (256 or 512).
    preprocess : str
        Preprocessing mode; determines which mapping checkpoint / facerender
        config is used ('crop', 'extcrop', 'resize', 'full', 'extfull').
    old_version : bool
        Use the ``.pth`` checkpoints instead of the ``.safetensors`` version.

    Returns
    -------
    dict
        Dictionary with keys ``preprocess_model``, ``audio_to_coeff``,
        ``animate_from_coeff``, ``sadtalker_paths``, ``device``, ``size`` and
        ``preprocess`` for use with :func:`generate`.
    """
    device = _resolve_device(device)

    config_dir = os.path.join(_MODULE_DIR, "src", "config")
    sadtalker_paths = init_path(checkpoint_dir, config_dir, size, old_version, preprocess)

    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    return {
        "preprocess_model": preprocess_model,
        "audio_to_coeff": audio_to_coeff,
        "animate_from_coeff": animate_from_coeff,
        "sadtalker_paths": sadtalker_paths,
        "device": device,
        "size": size,
        "preprocess": preprocess,
    }


def generate(models, source_image, driven_audio, output_path,
             result_dir="/tmp/sadtalker", pose_style=0, batch_size=2,
             expression_scale=1.0, enhancer=None, background_enhancer=None,
             still=False, preprocess="crop", size=256,
             ref_eyeblink=None, ref_pose=None,
             input_yaw=None, input_pitch=None, input_roll=None,
             verbose=False, **_):
    """Run the full SadTalker pipeline and produce a talking-face video.

    Parameters
    ----------
    models : dict
        The dictionary returned by :func:`load_models`.
    source_image : str
        Path to the source portrait image.
    driven_audio : str
        Path to the driving audio (wav).
    output_path : str
        Destination path for the final ``.mp4`` video.
    result_dir : str
        Working directory for intermediate outputs.
    pose_style : int
        Pose style index in [0, 46).
    batch_size : int
        Facerender batch size.
    expression_scale : float
        Expression intensity scale.
    enhancer : str or None
        Face enhancer, e.g. ``"gfpgan"`` or ``"RestoreFormer"``.
    background_enhancer : str or None
        Background enhancer, e.g. ``"realesrgan"``.
    still : bool
        Reduce head motion (useful for full-body / full preprocess).
    preprocess : str
        Preprocessing mode. Should match the mode used in load_models.
    size : int
        Face render image size.

    Returns
    -------
    str
        Path to the generated video (``output_path``).
    """
    preprocess_model = models["preprocess_model"]
    audio_to_coeff = models["audio_to_coeff"]
    animate_from_coeff = models["animate_from_coeff"]
    device = models["device"]
    # Prefer the values models were loaded with when available.
    preprocess = models.get("preprocess", preprocess)
    size = models.get("size", size)

    save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)
    print("3DMM Extraction for source image")
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        source_image, first_frame_dir, preprocess,
        source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        raise RuntimeError("Can't get the coeffs of the input image: %s" % source_image)

    # optional reference video for eye blinking
    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print("3DMM Extraction for the reference video providing eye blinking")
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    # optional reference video for pose
    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing pose")
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path = None

    # audio2coeff
    batch = get_data(first_coeff_path, driven_audio, device,
                     ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # coeff2video
    data = get_facerender_data(
        coeff_path, crop_pic_path, first_coeff_path, driven_audio,
        batch_size, input_yaw, input_pitch, input_roll,
        expression_scale=expression_scale, still_mode=still,
        preprocess=preprocess, size=size)

    result = animate_from_coeff.generate(
        data, save_dir, source_image, crop_info,
        enhancer=enhancer, background_enhancer=background_enhancer,
        preprocess=preprocess, img_size=size)

    # move final result to the requested output path
    output_path = os.path.abspath(output_path)
    out_parent = os.path.dirname(output_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    shutil.move(result, output_path)
    print("The generated video is named:", output_path)

    if not verbose:
        shutil.rmtree(save_dir, ignore_errors=True)

    return output_path


def main():
    """CLI entry point mirroring inference.py options."""
    parser = ArgumentParser(description="SadTalker talking-face generation")
    parser.add_argument("--driven_audio", required=True, help="path to driven audio")
    parser.add_argument("--source_image", required=True, help="path to source image")
    parser.add_argument("--output_path", default="./sadtalker_result.mp4",
                        help="path to the output .mp4 video")
    parser.add_argument("--ref_eyeblink", default=None,
                        help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None,
                        help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default="./checkpoints",
                        help="path to checkpoints directory")
    parser.add_argument("--result_dir", default="/tmp/sadtalker",
                        help="path for intermediate outputs")
    parser.add_argument("--pose_style", type=int, default=0,
                        help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,
                        help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.0,
                        help="the expression scale of the facerender")
    parser.add_argument("--input_yaw", nargs="+", type=int, default=None,
                        help="the input yaw degree of the user")
    parser.add_argument("--input_pitch", nargs="+", type=int, default=None,
                        help="the input pitch degree of the user")
    parser.add_argument("--input_roll", nargs="+", type=int, default=None,
                        help="the input roll degree of the user")
    parser.add_argument("--enhancer", type=str, default=None,
                        help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument("--background_enhancer", type=str, default=None,
                        help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--still", action="store_true",
                        help="can crop back to the original videos for the full body animation")
    parser.add_argument("--preprocess", default="crop",
                        choices=["crop", "extcrop", "resize", "full", "extfull"],
                        help="how to preprocess the images")
    parser.add_argument("--verbose", action="store_true",
                        help="saving the intermediate output or not")
    parser.add_argument("--old_version", action="store_true",
                        help="use the pth other than safetensor version")

    args = parser.parse_args()

    device = "cpu"
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

    models = load_models(checkpoint_dir=args.checkpoint_dir, device=device,
                         size=args.size, preprocess=args.preprocess,
                         old_version=args.old_version)

    out = generate(
        models,
        source_image=args.source_image,
        driven_audio=args.driven_audio,
        output_path=args.output_path,
        result_dir=args.result_dir,
        pose_style=args.pose_style,
        batch_size=args.batch_size,
        expression_scale=args.expression_scale,
        enhancer=args.enhancer,
        background_enhancer=args.background_enhancer,
        still=args.still,
        preprocess=args.preprocess,
        size=args.size,
        ref_eyeblink=args.ref_eyeblink,
        ref_pose=args.ref_pose,
        input_yaw=args.input_yaw,
        input_pitch=args.input_pitch,
        input_roll=args.input_roll,
        verbose=args.verbose,
    )
    print(out)
    return out


if __name__ == "__main__":
    main()
