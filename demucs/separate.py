import argparse
# import sys
from pathlib import Path

# from dora.log import fatal
import torch as th

from .api import Separator, save_audio#, list_models

# from .apply import BagOfModels
# from .htdemucs import HTDemucs
from .pretrained import add_model_flags#, ModelLoadingError


def get_parser():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='*', type=Path, default=[], help='Path to tracks')
    add_model_flags(parser)
    parser.add_argument("--list-models", action="store_true", help="List available models "
                        "from current repo and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        help="Only separate audio into {STEM} and no_{STEM}. ")
    parser.add_argument("--other-method", dest="other_method", choices=["none", "add", "minus"],
                        default="add", help='Decide how to get "no_{STEM}". "none" will not save '
                        '"no_{STEM}". "add" will add all the other stems. "minus" will use the '
                        "original track minus the selected stem.")
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--int24", action="store_true",
                             help="Save wav output as 24 bits wav.")
    depth_group.add_argument("--float32", action="store_true",
                             help="Save wav output as float32 (2x bigger).")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp", "none"],
                        help="Strategy for avoiding clipping: rescaling entire signal "
                             "if necessary  (rescale) or hard clipping (clamp).")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--flac", action="store_true",
                              help="Convert the output wavs to flac.")
    format_group.add_argument("--mp3", action="store_true",
                              help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate",
                        default=320,
                        type=int,
                        help="Bitrate of converted mp3.")
    parser.add_argument("--mp3-preset", choices=range(2, 8), type=int, default=2,
                        help="Encoder preset of MP3, 2 for highest quality, 7 for "
                        "fastest speed. Default is 2")
    parser.add_argument("-j", "--jobs",
                        default=0,
                        type=int,
                        help="Number of jobs. This can increase memory usage but will "
                             "be much faster when multiple cores are available.")

    return parser


def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)
    # print(args)
    separator = Separator(model=args.name,
                            repo=args.repo,
                            device=args.device,
                            # shifts=args.shifts,
                            # split=args.split,
                            overlap=args.overlap,
                            # progress=True,
                            # jobs=args.jobs,
                            # segment=args.segment
                            )
   
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    # print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
      
        _, res = separator.separate_audio_file(track)

        ext = "mp3"
        # ext = "wav"
        
        kwargs = {
            "samplerate": separator._samplerate,
            "bitrate": args.mp3_bitrate,
            "preset": args.mp3_preset,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }

        for name, source in res.items():
            stem = out / args.filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem=name,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)


if __name__ == "__main__":
    main()
