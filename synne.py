import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.augment import LetterBox
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
from skimage.color import rgb2hsv

class PlayerTracker():

    def __init__(self, model):
        self.model = model
        self.class_list = [0]


    def remove_outliers(self, x : np.ndarray, normalizer : Normalizer, stand_scaler: StandardScaler, scale: bool = True):
        x_norm = normalizer.transform(x)
        z_scores = np.abs((x_norm - x_norm.mean(axis=0)) / x_norm.std(axis=0))
        outlier_mask = (z_scores < 3).all(axis=1)
        x_no_outliers = x_norm[outlier_mask]
        x_no_outliers_scaled = stand_scaler.transform(x_no_outliers)
        return x_no_outliers_scaled


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
        
    def bgr_to_name(self, bgr):
        print(bgr)
        return ""

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
            players_teams = {}
            bgr_values = []
            hsv_values = []

            for mask, player_id in zip(mask_data, track_ids):
                resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                binary_mask = resized_mask > 0.5

                # Pixels for this player [B, G, R]
                player_pixelBGR = frame[binary_mask]
                player_pixelHSV = rgb2hsv(player_pixelBGR[::-1]) # Get hsv from BGR ([::-1] reshapes from (B,G,R) to (R,G,B))

                hsv_values.append(np.mean(player_pixelHSV, axis=0).tolist())

                mean_color = np.mean(player_pixelBGR, axis=0) # Get mean BGR values for 'player_id' [mean_B, mean_G, mean_R], then transform to grayscale
                bgr_values.append(mean_color)


                # Plot masks, en farge på alle - bytte dette ulik farge på de to lagene?
                annotator.masks(masks.data, colors=[((4, 42, 255))],
                                im_gpu=im_gpu)
            

            """
            bgr_values = np.vstack(bgr_values)
            normalizer_bgr = Normalizer().fit(bgr_values)
            standard_scaler_bgr = StandardScaler().fit(bgr_values)
            scaled_norm_bgr = standard_scaler_bgr.transform(normalizer_bgr.transform(bgr_values))
            bgr_no_outliers = self.remove_outliers(bgr_values, normalizer_bgr, standard_scaler_bgr)
            kmeans_bgr = KMeans(n_clusters=2, random_state=0).fit(bgr_no_outliers) # Kmeans with mean bgr values as features
            """

            
            hsv_values = np.vstack(hsv_values)
            normalizer_hsv = Normalizer().fit(hsv_values)
            standard_scaler_hsv = StandardScaler().fit(hsv_values)
            scaled_hsv = standard_scaler_hsv.transform(hsv_values)
            weighted_scaled_hsv = scaled_hsv * np.array([2, 1.2, 0.5]) # Hsv with more weight on Hue
            print(weighted_scaled_hsv.shape)
            hsv_no_outliers = self.remove_outliers(hsv_values, normalizer_hsv, standard_scaler_hsv)
            kmeans_hsv = KMeans(n_clusters=2, random_state=0).fit(weighted_scaled_hsv) # Kmeans with mean hsv + std(hsv) values as features
            


            #labels = kmeans_hsv.predict(scaled_norm_hsv) # Kmeans with both bgr and hsv as features
            #labels = kmeans_bgr.predict(scaled_norm_bgr) # Kmeans with mean BGR as values

            # Run through labels for each player id and add to players_teams dict to accsess later.
            for (idx, player_label) in enumerate(kmeans_hsv.labels_):
                players_teams[track_ids[idx]] = player_label


        for b, t, c in zip(boxes, track_ids, clss):
            if c == 0:
                name = 'player'
            elif c == 32:
                name = 'ball'

            #l = f"{name}, id:{t}, team: {players_teams[t]}"
            l = f"team: {players_teams[t]}"
            annotator.box_label(b, color=(221, 0, 186), label= l)
        return frame


