import cv2
import av
import numpy as np

class SupplyWriter:
    def __init__(self, intput_video, output_video, opt_thres, rgb_input=True):
        reader = cv2.VideoCapture(intput_video)
        self.fps = reader.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reader.release()
        
        self.output_video = output_video
        self.rgb_input = rgb_input
        self.opt_thres = opt_thres

    def run(self, images, scores, boxes):
        # Open the output container and add a video stream
        container = av.open(self.output_video, mode='w')
        # Use libx264 for robust browser compatibility
        stream = container.add_stream('libx264', rate=int(self.fps))
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = 'yuv420p'

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 5
        font_scale = 3

        for image, score, box in zip(images, scores, boxes):
            # Ensure image is mutable
            if not image.flags.writeable:
                image = image.copy()
                
            if self.rgb_input:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            if box is not None:
                label = "fake" if score > self.opt_thres else "real"
                x1, y1, x2, y2 = box
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                color = (255, 255, 0) if label == "real" else (0, 255, 255) # BGR
                cv2.putText(
                    image, label, (x, y + h + 68),
                    font_face, font_scale, color, thickness, 2
                )
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 10)
            
            # Convert back to RGB for PyAV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame = av.VideoFrame.from_ndarray(image_rgb, format='rgb24')
            
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush the encoder
        for packet in stream.encode():
            container.mux(packet)
            
        container.close()
