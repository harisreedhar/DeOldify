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
    def colorizeSingle(self, frame, render_factor):
        orig_image = Image.fromarray(frame).convert('RGB')
        filtered_image = self.filter.filter(orig_image, orig_image, render_factor=render_factor,post_process=False)
        restored = cv2.cvtColor(np.asarray(filtered_image), cv2.COLOR_BGR2RGB)
        return restored

    def colorizeImage(self, args):
        inputPath = args.input
        name, extension = os.path.splitext(inputPath)
        baseName = Path(inputPath).stem
        image = cv2.imread(inputPath, cv2.IMREAD_COLOR)
        coloredImage = self.colorizeSingle(image, args.render_factor)
        fileName = os.path.join(args.output, baseName  + f"_colored.{extension}")
        cv2.imwrite(fileName, coloredImage)

    def colorizeDirectory(self, args):
        inputPath = args.input
        files = os.listdir(inputPath)
        length = len(files)
        for i, filename in enumerate(files):
            sys.stdout.write(f"Colorizing file {i+1} of {length}\r")
            sys.stdout.flush()
            filePath = os.path.join(inputPath, filename)
            if not os.path.isfile(filePath):
                continue
            name, extension = os.path.splitext(filePath)
            baseName = Path(filePath).stem
            image = cv2.imread(filePath, cv2.IMREAD_COLOR)
            coloredImage = self.colorizeSingle(image, args.render_factor)
            outFileName = os.path.join(args.output, baseName  + f".{extension}")
            cv2.imwrite(outFileName, coloredImage)

    def colorizeVideo(self, args):
        os.makedirs(args.output, exist_ok=True)
        inputVideo = args.input
        render_factor = args.render_factor
        baseName = Path(inputVideo).stem
        outNoAudio = os.path.join(args.output, baseName + "_colored_noaudio.mp4")
        outWithAudio = os.path.join(args.output, baseName + "_colored.mp4")

        # write video
        videoWriter = None
        frameCount = 0
        cap = cv2.VideoCapture(inputVideo)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        codec = cv2.VideoWriter_fourcc(*'mp4v')

        while True:
            _, frame = cap.read(cv2.IMREAD_COLOR)
            if frame is None:
                break
            else:
                frameCount += 1
                sys.stdout.write(f"Colorizing frame {frameCount} of {length}\r")
                sys.stdout.flush()

                restored = self.colorizeSingle(frame, render_factor)

                if videoWriter == None:
                    frameSize = (restored.shape[1], restored.shape[0])
                    videoWriter = cv2.VideoWriter(outNoAudio, codec, fps, frameSize, True)

                videoWriter.write(restored)

        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()

        # transfer audio
        print("\nTransferring audio")
        tempAudio = os.path.join(args.output, f"temp_{uuid.uuid4().hex}.mp3" )
        command1 = f"ffmpeg -i {inputVideo} -map 0:a {tempAudio}"
        command2 = f'ffmpeg -y -i {tempAudio} -i {outNoAudio} -strict -2 -q:v 1 {outWithAudio}'
        subprocess.call(" && ".join([command1, command2]), shell=platform.system() != 'Windows')
        os.remove(tempAudio)
        os.remove(outNoAudio)
        print(f'Results are in the [{args.output}] folder.')

def getModel(args):
    root_folder = Path("./")
    if args.model == "stable":
        learn = gen_inference_wide(root_folder=root_folder, weights_name='ColorizeStable_gen')
    if args.model == "artistic":
        learn = gen_inference_deep(root_folder=root_folder, weights_name='ColorizeArtistic_gen')
    if args.model == "video":
        learn = gen_inference_wide(root_folder=root_folder, weights_name='ColorizeVideo_gen')
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=args.render_factor)
    vis = ColorizeLocal(filtr, results_dir="result_images")
    return vis

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs/test.mp4', help='Image, Directory or Video')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--render_factor', type=int, default=21, help='default value is 21')
    parser.add_argument('--model', type=str, default='stable', help="'artistic', 'stable', 'video'")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        print("Output should be a directory")

    model = getModel(args)

    if os.path.isfile(args.input):
        guessType = mimetypes.guess_type(args.input)[0]
        if guessType.startswith('video'):
            model.colorizeVideo(args)
        elif guessType.startswith('image'):
            model.colorizeImage(args)
        else:
            print("Invalid input")
    else:
        if os.path.isdir(args.input):
            model.colorizeDirectory(args)
        else:
            print("Invalid input")
