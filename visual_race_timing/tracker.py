import numpy as np
from collections import deque
from ultralytics.trackers.basetrack import BaseTrack, TrackState
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT
from ultralytics.trackers.utils import matching
from ultralytics.utils.plotting import Annotator, colors
import torchreid
import torch
import cv2
import copy


class RaceTracker(BOTSORT):
    def __init__(
            self,
            model_weights,
            args,
            frame_rate=30
    ):
        super().__init__(args, frame_rate)
        self.half = False
        self.device = "cuda"
        self.unknowns = {}
        if args.with_reid:
            self.feature_extractor = torchreid.utils.FeatureExtractor(model_name='osnet_ain_x1_0',
                                                                      model_path=model_weights,
                                                                      device='cuda')

    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        xyxys = results.xyxy
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        xyxys = xyxys[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        if len(xyxys > 0):
            crops = self.get_crops(xyxys, img)
            features = self.feature_extractor(crops)
            features = [x.cpu().numpy() for x in features]
        else:
            features = []

        detections = [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features)]
        '''
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        '''
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        emb_dists = matching.embedding_distance(strack_pool, detections)

        # Step 3: Deal with the unmatched
        track_ids = [t.track_id for t in strack_pool]
        for inew in u_detection:
            annotator = Annotator(
                copy.deepcopy(img),
            )
            annotator.box_label(detections[inew].xyxy, color=colors(0))
            cv2.imshow("Race Tracking", annotator.result())
            cv2.waitKey(1)
            if len(strack_pool) == 0:
                user_input = input("New ID")
                while True:
                    try:
                        new_id = int(user_input, 16)
                        break
                    except ValueError:
                        user_input = input("New ID")
                track = detections[inew]
                track.activate(self.kalman_filter, self.frame_id)
                track.track_id = new_id
                activated_stracks.append(track)
                continue

            matched_tracks = [match[0] for match in matches]
            # Mask embedding dist for matched tracks
            emb_dists[matched_tracks] = 1
            iclosest = np.argmin(emb_dists[:, inew])
            det = detections[inew]
            track = strack_pool[iclosest]
            print(f"re-ID as {hex(track.track_id)[2:]}?")
            user_input = input("Y(es)/U(nknown)/Other ID")
            user_input = user_input.strip()
            if 'Y' in user_input[0] or 'y' in user_input[0]:
                # track.re_activate(det, self.frame_id, new_id=False)
                # refind_stracks.append(track)
                # matched_tracks.append(iclosest)
                matches.append([iclosest, inew])
            elif 'U' in user_input[0] or 'u' in user_input[0]:
                name = user_input[1:].strip()
                det.activate(self.kalman_filter, self.frame_id)
                det.track_id = 0xF00 + len(self.unknowns)
                activated_stracks.append(det)
                self.unknowns[det.track_id] = name

            else:
                while True:
                    try:
                        other_id = int(user_input, 16)
                        break
                    except ValueError:
                        user_input = input("Y/Other ID")

                if other_id in track_ids:
                    itracked = track_ids.index(other_id)
                    matched_tracks = [match[0] for match in matches]
                    if itracked in matched_tracks:
                        wrong_match_idx = matched_tracks.index(itracked)
                        matches[wrong_match_idx][1] = inew
                        # TODO
                    else:
                        matches.append([itracked, inew])
                else:
                    # First time a person is tracked
                    det.activate(self.kalman_filter, self.frame_id)
                    det.track_id = other_id
                    activated_stracks.append(det)

        # Step 4: Update tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for it in range(len(r_tracked_stracks)):
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 5: Update state

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        # self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        # self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # self.removed_stracks.extend(removed_stracks)
        # if len(self.removed_stracks) > 1000:
        #     self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        return np.asarray([x.result for x in self.tracked_stracks], dtype=np.float32)

    def get_crops(self, xyxys, img):
        crops = []
        h, w = img.shape[:2]
        resize_dims = (128, 256)
        interpolation_method = cv2.INTER_LINEAR
        mean_array = np.array([0.485, 0.456, 0.406])
        std_array = np.array([0.229, 0.224, 0.225])
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype('int')
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            # resize
            crop = cv2.resize(
                crop,
                resize_dims,  # from (x, y) to (128, 256) | (w, h)
                interpolation=interpolation_method,
            )

            # (cv2) BGR 2 (PIL) RGB. The ReID models have been trained with this channel order
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        # List of torch tensor crops to unified torch tensor
        crops = torch.stack(crops, dim=0)

        # Normalize the batch
        crops = crops / 255.0

        # Standardize the batch
        crops = (crops - mean_array) / std_array

        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(dtype=torch.half if self.half else torch.float, device=self.device)

        return crops
