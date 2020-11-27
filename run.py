from stressdetection.StressDetection import Stress
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', default=0,
                    help='path to the video to be processed or webcam id (0 for example). Default value: 0.')

args = parser.parse_args()

stress = Stress()
stress.run(args.input)