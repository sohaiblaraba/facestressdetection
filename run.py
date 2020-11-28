from stressdetection.StressDetection import Stress
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', default=0,
                    help='Path to the video to be processed or webcam id (0 for example). Default value: 0.')
parser.add_argument('--rectangle', default=False, action=argparse.BooleanOptionalAction,
					help='Show the rectangle around the detected face. Default: False')
parser.add_argument('--landmarks', default=False, action=argparse.BooleanOptionalAction,
					help='Show the landmarks of the detected face. Default: False')
parser.add_argument('--forehead', default=False, action=argparse.BooleanOptionalAction,
					help='Show the forehead in oridinal color. Default: False')
parser.add_argument('--forehead_outline', default=False, action=argparse.BooleanOptionalAction,
					help='Draw a rectangle around the detected forehead. Default: True')
parser.add_argument('--fps', default=False, action=argparse.BooleanOptionalAction,
					help='Show the framerate. Default: False')

args = parser.parse_args()

stress = Stress()
stress.display_rectangle(args.rectangle)
stress.display_landmarks(args.landmarks)
stress.display_forehead(args.forehead)
stress.display_forehead_outline(args.forehead_outline)
stress.display_fps(args.fps)

stress.run(args.input)