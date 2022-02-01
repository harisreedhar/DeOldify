import os
import sys
import cv2
import uuid
import argparse
import mimetypes
import numpy as np
from pathlib import Path
import subprocess, platform

from deoldify import device
from deoldify.visualize import *
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")


class ColorizeLocal(ModelImageVisualizer):
    def colorize_single(self, frame, render_factor, watermark=True, post_process=True):
        orig_image = Image.fromarray(frame).convert('RGB')
        filtered_image = self.filter.filter(orig_image, orig_image, render_factor=render_factor, post_process=post_process)
        if watermark:
            filtered_image = get_watermarked(filtered_image)
        restored = cv2.cvtColor(np.asarray(filtered_image), cv2.COLOR_BGR2RGB)
        return restored

    def colorize_image(self, args):
        input_path = args.input
        name, extension = os.path.splitext(input_path)
        base_name = Path(input_path).stem
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        colored_image = self.colorize_single(image, args.factor,
            watermark=args.watermark, post_process=args.post_process)
        fileName = os.path.join(args.output, base_name  + f"_colorized.{extension}")
        cv2.imwrite(fileName, colored_image)

    def colorize_directory(self, args):
        input_path = args.input
        files = os.listdir(input_path)
        length = len(files)
        for i, f in enumerate(files):
            sys.stdout.write(f"Colorizing file {i+1} of {length}\r")
            sys.stdout.flush()
            file_path = os.path.join(input_path, f)
            if not os.path.isfile(file_path):
                continue
            if mimetypes.guess_type(file_path)[0].startswith('image'):
                name, extension = os.path.splitext(file_path)
                base_name = Path(file_path).stem
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                colored_image = self.colorize_single(image, args.factor,
                    watermark=args.watermark, post_process=args.post_process)
                out_file_name = os.path.join(args.output, base_name  + f".{extension}")
                cv2.imwrite(out_file_name, colored_image)

    def colorize_video(self, args):
        os.makedirs(args.output, exist_ok=True)
        input_video = args.input
        render_factor = args.factor
        base_name = Path(input_video).stem
        out_no_audio = os.path.join(args.output, base_name + "_colorized_noaudio.mp4")
        out_with_audio = os.path.join(args.output, base_name + "_colorized.mp4")

        # write video
        video_writer = None
        frame_count = 0
        cap = cv2.VideoCapture(input_video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = cv2.VideoWriter_fourcc(*'mp4v')

        while True:
            _, frame = cap.read(cv2.IMREAD_COLOR)
            if frame is None:
                break
            else:
                frame_count += 1
                sys.stdout.write(f"Colorizing frame {frame_count} of {length}\r")
                sys.stdout.flush()

                restored = self.colorize_single(frame, render_factor,
                    watermark=args.watermark, post_process=args.post_process)

                if video_writer == None:
                    frame_size = (restored.shape[1], restored.shape[0])
                    video_writer = cv2.VideoWriter(out_no_audio, codec, fps, frame_size, True)

                video_writer.write(restored)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        # transfer audio
        print("\nTransferring audio")
        tempAudio = os.path.join(args.output, f"temp_{uuid.uuid4().hex}.mp3" )
        command1 = f"ffmpeg -i {input_video} -map 0:a {tempAudio}"
        command2 = f'ffmpeg -y -i {tempAudio} -i {out_no_audio} -strict -2 -q:v 1 {out_with_audio}'
        subprocess.call(" && ".join([command1, command2]), shell=platform.system() != 'Windows')
        os.remove(tempAudio)
        os.remove(out_no_audio)
        print(f'Results are in the [{args.output}] folder.')

def getModel(args):
    root_folder = Path("./")
    if args.model == "stable":
        learn = gen_inference_wide(root_folder=root_folder, weights_name='ColorizeStable_gen')
    if args.model == "artistic":
        learn = gen_inference_deep(root_folder=root_folder, weights_name='ColorizeArtistic_gen')
    if args.model == "video":
        learn = gen_inference_wide(root_folder=root_folder, weights_name='ColorizeVideo_gen')
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=args.factor)
    vis = ColorizeLocal(filtr, results_dir="result")
    return vis

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs/test.mp4', help='Input directory or file(image or video)')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--factor', type=int, default=21, help='Render factor (default 21)')
    parser.add_argument('--model', type=str, default='stable', help="Model weight type(artistic, stable, video)")
    parser.add_argument('--no_watermark', dest='watermark', action='store_false', help="Disable watermark")
    parser.add_argument('--no_postprocess', dest='post_process', action='store_false', help="Disable post-process")
    parser.set_defaults(watermark=True, post_process=True)
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        print("Output should be a directory")

    model = getModel(args)

    if os.path.isfile(args.input):
        guess_type = mimetypes.guess_type(args.input)[0]
        if guess_type.startswith('video'):
            model.colorize_video(args)
        elif guess_type.startswith('image'):
            model.colorize_image(args)
        else:
            print("Invalid input")
    else:
        if os.path.isdir(args.input):
            model.colorize_directory(args)
        else:
            print("Invalid input")
