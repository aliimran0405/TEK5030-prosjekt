import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox
import numpy as np
from sklearn.cluster import KMeans as KMeans

class PlayerTracker():

    def __init__(self, model):
        self.model = model
        self.class_list = [0]

    def hsv_to_color_name(self, hsv):
        h, s, v = hsv
        # Svart
        if v < 50:
            return "Svart"
        # Hvit
        if s < 30 and v > 180:
            return "Hvit"
        # Rød (inkl. oransje og rødtoner)
        if h <= 25 or h >= 160:
            return "Rød"
        # Gul
        elif 26 <= h <= 34:
            return "Gul"
        # Grønn
        elif 35 <= h <= 85:
            return "Grønn"
        # Blå (inkl. lilla)
        elif 86 <= h <= 145:
            return "Blå"


    def process_frame(self, frame):


        results = self.model.track(frame, classes=self.class_list, tracker='botsort.yaml',  persist=True)[0]
        annotator = Annotator(frame, line_width=2)


        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.tolist()
            confs = results.boxes.conf.cpu().numpy()
            track_ids = results.boxes.id.int().tolist()
            clss = results.boxes.cls.cpu().tolist()
            masks = results.masks

        # Prepare mask
        img = LetterBox(masks.shape[1:])(
            image=annotator.result())
        im_gpu = (torch.as_tensor
                (img, dtype=torch.float16,
                    device=masks.data.device)
                .permute(2, 0, 1).flip(0)
                .contiguous() /255)

        if masks is not None and track_ids is not None:
            mask_data = masks.data.cpu().numpy()
            players_colors = {}

            for mask, player_id in zip(mask_data, track_ids):
                resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary_mask = resized_mask > 0.5
                # Pixels for this player [B, G, R]
                player_pixel = frame[binary_mask]
                player_pixel = player_pixel[(player_pixel[:,1] < 150)]

                # Kmean clustering for å finne farge istedenfor å bruke mean
                if player_pixel.shape[0] > 10:
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(player_pixel)
                    unique, counts = np.unique(kmeans.labels_, return_counts=True)
                    dominant_index = unique[np.argmax(counts)]
                    dominant_color = kmeans.cluster_centers_[dominant_index].astype(np.uint8)
                    hsv_pixel = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)
                    h, s, v = hsv_pixel[0][0]
                    color_name = self.hsv_to_color_name((h, s, v))
                else:
                    color_name = 'ukjent'


                # Legger kun til farge på en spiller en gang
                if player_id not in players_colors:
                    players_colors[player_id] = color_name

                # Plot masks, en farge på alle - bytte dette ulik farge på de to lagene?
                annotator.masks(masks.data, colors=[((4, 42, 255))],
                                im_gpu=im_gpu)

        for b, t, c in zip(boxes, track_ids, clss):
            if c == 0:
                name = 'player'
            elif c == 32:
                name = 'ball'

            l = f"{name}, id:{t}"
            annotator.box_label(b, color=(221, 0, 186), label= l)
        return frame


