import numpy as np
from scipy.spatial.distance import cdist
from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack, TrackState
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT
from ultralytics.trackers.utils import matching
from ultralytics.utils import ops
from ultralytics.utils.ops import xywh2ltwh
from ultralytics.utils.plotting import Annotator, colors
import torchreid
import torch
import cv2
import copy

from visual_race_timing.prompts import ask_for_id


def get_crops(xyxys, img, half=False, device='cuda'):
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
    crops = crops.to(dtype=torch.half if half else torch.float, device=device)

    return crops


class RaceTracker(BOTSORT):
    def __init__(
            self,
            model_weights,
            args,
            frame_rate=30,
            participants=None,
            device="0"
    ):
        super().__init__(args, frame_rate)
        self.half = False
        self.device = device
        self.unknowns = {}
        self.participants = participants
        self.display_delegate = lambda x: None
        if args.with_reid:
            self.feature_extractor = torchreid.utils.FeatureExtractor(model_name='osnet_ain_x1_0',
                                                                      model_path=model_weights,
                                                                      device=self.device)
        # HACK: Monkey patch so we get hex print outs to match bibs
        BOTrack.__str__ =  lambda self: f"OT_{format(self.track_id, '02x')}_({self.start_frame}-{self.end_frame})"
        # cv2.namedWindow("Decision")
        self.frame_id = None

    def update(self, results, img=None, frame_id=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        if self.frame_id is None and frame_id is not None:
            self.frame_id = frame_id
        elif frame_id is not None:
            delta = frame_id - self.frame_id
            if delta > 1:
                # We've skipped some frames, so we need null updates
                for i in range(delta - 1):
                    self.update(Boxes(np.zeros((0, 7)), img.shape[:2]), img)
            return self.update(results, img)
        else:
            self.frame_id += 1
        activated_stracks = [] # Active and detected within this update
        refind_stracks = []
        lost_stracks = [] # Temporarily unmatched, but could come back momentarily
        removed_stracks = [] # Assumed gone. Requires human intervention to re-activate

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        xyxys = results.xyxy
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores > self.args.track_high_thresh

        dets = bboxes[remain_inds]
        xyxys = xyxys[remain_inds]

        if self.feature_extractor and len(xyxys > 0):
            crops = get_crops(xyxys, img)
            features = self.feature_extractor(crops)
            features = [x.cpu().numpy() for x in features]
        else:
            features = []

        # NOTE: BOTrack constructor's label for first arg is wrong. It's treated as xywh, not tlwh
        detections = [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(dets, scores, cls, features)]

        # Step 1: First association, with high score detection boxes
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        # Predict updates for tracked or recently-tracked objects
        self.multi_predict(strack_pool)
        # We'll consider all the tracks, but later we'll disqualify the removed ones from some matchings
        strack_pool = self.joint_stracks(strack_pool, self.removed_stracks)
        removed_track_i = [i for i, t in enumerate(strack_pool) if t.state == TrackState.Removed]
        # Calculate track-to-detection cost matrix (lower cost => better match)
        dists = self.get_dists(strack_pool, detections)
        dists[removed_track_i, :] = np.inf
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        emb_dists = matching.embedding_distance(strack_pool, detections)


        # Step 2: Deal with the unmatched
        track_ids = [t.track_id for t in strack_pool]
        for _, inew in enumerate(u_detection):
            annotator = Annotator(
                copy.deepcopy(img),
            )
            annotator.box_label(detections[inew].xyxy, label=f"{scores[inew]:.2f} {_ + 1}/{len(u_detection)}", color=colors(0))
            self.display_delegate(annotator.result())
            if len(strack_pool) == 0:
                # No one to match with, new track
                user_input = ask_for_id([(bib.lower(), (name,)) for bib, name in self.participants.items()], allow_other=True)
                if user_input[0] == 'U':
                    name = user_input[1:].strip()
                    det = detections[inew]
                    det.activate(self.kalman_filter, self.frame_id)
                    det.track_id = 0xF00 + len(self.unknowns)
                    activated_stracks.append(det)
                    self.unknowns[det.track_id] = name
                    continue
                elif user_input.strip().lower() == 'skip':
                    continue
                else:
                    det = detections[inew]
                    det.activate(self.kalman_filter, self.frame_id)
                    det.track_id = int(user_input, 16)
                    activated_stracks.append(det)
                continue

            matched_tracks = [match[0] for match in matches]
            # Mask embedding dist for matched tracks
            emb_dists[matched_tracks] = 1
            # Get (at most) the 5 closest embeddings
            emb_dist_ranking = np.argsort(emb_dists[:, inew])
            match_emb_dists = emb_dists[emb_dist_ranking, inew]
            det = detections[inew]
            closest_tracks = [strack_pool[i] for i in emb_dist_ranking[:5]]
            dists = list(match_emb_dists[:5])
            bibs = [format(track.track_id, '02x').lower() for track in closest_tracks]
            bibs.extend([bib for bib in self.participants.keys() if bib not in bibs])
            bibs.extend([format(bib, '02x').lower() for bib in self.unknowns.keys() if bib not in bibs])
            names = [self.participants.get(bib, '') for bib in bibs]
            names = [name if name else self.unknowns.get(bib, '') for name, bib in zip(names, bibs)]
            dists.extend([1 for _ in range(len(names)- len(dists))])
            user_input = ask_for_id([(bib, (name, f"{dist:.2f}")) for bib, name, dist in zip(bibs, names, dists)], show_default=False, allow_other=True)
            if user_input[0].lower() == 'u':
                name = user_input[1:].strip()
                det.activate(self.kalman_filter, self.frame_id)
                det.track_id = 0xF00 + len(self.unknowns)
                activated_stracks.append(det)
                self.unknowns[det.track_id] = name
            elif user_input.strip().lower() == 'skip':
                continue
            else:
                bib_id = int(user_input, 16)
                if bib_id in track_ids:
                    itracked = track_ids.index(bib_id)
                    # We may have matched this track earliier so we need to update the match
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
                    det.track_id = bib_id
                    activated_stracks.append(det)

        # Step 3: Update tracks based on matched pairs
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        # Step 4: Update state
        # Promote to lost if not matched
        unmatched_tracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for it in range(len(unmatched_tracks)):
            track = unmatched_tracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # Promote to removed if lost for too long
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks = self.sub_stracks(self.removed_stracks, self.tracked_stracks)
        self.removed_stracks.extend(removed_stracks)

        return np.asarray([x.result for x in self.tracked_stracks], dtype=np.float32)

    def get_dists(self, tracks, detections):
        """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        # Just multiplies together detection confidence and IoU
        dists = matching.fuse_score(dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            # emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists


def custom_track_str(track):
    # We need this to be a function so we can easily pickle the tracker
    return f"OT_{format(track.track_id, '02X')}_({track.start_frame}-{track.end_frame})"


def noop(*args, **kwargs):
    # We need this to be a function so we can easily pickle the tracker
    return None

class PartiallySupervisedTracker(BOTSORT):
    def __init__(
            self,
            model_weights,
            args,
            frame_rate=30,
            device="0"
    ):
        super().__init__(args, frame_rate)
        self.half = False
        self.device = device
        self.display_delegate = noop
        if args.with_reid:
            self.feature_extractor = torchreid.utils.FeatureExtractor(model_name='osnet_ain_x1_0',
                                                                      model_path=model_weights,
                                                                      device=self.device)
        # HACK: Monkey patch so we get hex print outs to match bibs
        BOTrack.__str__ = custom_track_str
        # cv2.namedWindow("Decision")

    def update(self, results, img=None):
        """Updates object tracker with new detections and returns tracked object bounding boxes."""
        self.frame_id += 1
        activated_stracks = []  # Active and detected within this update
        refind_stracks = []
        lost_stracks = []  # Temporarily unmatched, but could come back momentarily
        removed_stracks = []  # Assumed gone. Requires human intervention to re-activate

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        xyxys = results.xyxy
        if results.id is not None:
            box_ids = results.id
        else:
            box_ids = np.full((len(bboxes)), -1)
        # Add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        remain_inds = scores > self.args.track_high_thresh

        dets = bboxes[remain_inds]
        xyxys = xyxys[remain_inds]

        if self.feature_extractor and len(xyxys > 0):
            crops = get_crops(xyxys, img)
            features = self.feature_extractor(crops)
            features = [x.cpu().numpy() for x in features]
        else:
            features = []

        # NOTE: BOTrack constructor's label for first arg is wrong. It's treated as xywh, not tlwh
        detections = [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(dets, scores, cls, features)]

        # Step 1: First association, with high score detection boxes
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        # Predict updates for tracked or recently-tracked objects
        self.multi_predict(strack_pool)
        # We'll consider all the tracks, but later we'll disqualify the removed ones from some matchings
        strack_pool = self.joint_stracks(strack_pool, self.removed_stracks)
        removed_track_i = [i for i, t in enumerate(strack_pool) if t.state == TrackState.Removed]
        # Calculate track-to-detection cost matrix (lower cost => better match)
        dists = self.get_dists(strack_pool, detections)
        dists[removed_track_i, :] = np.inf
        for known_id_i in range(len(box_ids)):
            if box_ids[known_id_i] == -1:
                continue
            # If we have a track with this ID, force the match
            for track_i, track in enumerate(strack_pool):
                if track.track_id == box_ids[known_id_i]:
                    dists[:, known_id_i] = np.inf
                    dists[track_i, known_id_i] = 0
                    break
            # Note that if there's no track with a known ID, this annotation may still match and donate it's ID!

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        emb_dists = matching.embedding_distance(strack_pool, detections)

        # Step 2: Deal with the unmatched
        track_ids = [t.track_id for t in strack_pool]
        for _, inew in enumerate(u_detection):
            # First time we're seeing this ID, it didn't match with anything. Make a new track
            if box_ids[inew] != -1:
                det = detections[inew]
                det.activate(self.kalman_filter, self.frame_id)

                det.track_id = box_ids[inew]
                activated_stracks.append(det)
                continue

            # FIXME: Should make a guess at the matching based on reid model too, but should figure out
            #   how to share the representation with the outside too
            matched_tracks = [match[0] for match in matches]
            # Mask embedding dist for matched tracks
            emb_dists[matched_tracks] = 1
            # Get (at most) the 5 closest embeddings
            emb_dist_ranking = np.argsort(emb_dists[:, inew])
            match_emb_dists = emb_dists[emb_dist_ranking, inew]
            det = detections[inew]
            closest_tracks = [strack_pool[i] for i in emb_dist_ranking[:5]]
            dists = list(match_emb_dists[:5])

            # First time a person is tracked
            det.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(det)
            det.track_id += 0xF00  # Blow this ID out of the race namespace so it doesn't accidentally get treated as a real bib ID

        # Step 3: Update tracks based on matched pairs
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if box_ids[idet] != -1:
                # This track got matched to a detection with a known ID. Adopt the ID
                track.track_id = box_ids[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 4: Update state
        # Promote to lost if not matched
        unmatched_tracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for it in range(len(unmatched_tracks)):
            track = unmatched_tracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # Promote to removed if lost for too long
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks = self.sub_stracks(self.removed_stracks, self.tracked_stracks)
        self.removed_stracks.extend(removed_stracks)

        return np.asarray([x.result for x in self.tracked_stracks], dtype=np.float32)

    def guess_id(self, frame, box):
        crops = get_crops(box[:, :4], frame)
        query_features = self.feature_extractor(crops).cpu().numpy()[0]
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        strack_pool = self.joint_stracks(strack_pool, self.removed_stracks)
        candidate_features_by_id = {track.track_id: track.smooth_feat for track in strack_pool}
        candidate_features = np.asarray(list(candidate_features_by_id.values()), dtype=np.float32)
        cost_matrix = np.zeros((len(candidate_features), 1), dtype=np.float32)
        if cost_matrix.size == 0:
            return [], cost_matrix
        det_features = np.asarray([query_features], dtype=np.float32)
        cost_matrix = np.maximum(0.0, cdist(candidate_features, det_features, 'cosine'))  # Normalized features
        return list(candidate_features_by_id.keys()), cost_matrix

    def update_participant_features(self, frame, box, track_id: int):
        # One box at a time!
        box = np.atleast_2d(box)
        crops = get_crops(box[:, :4], frame)
        query_features = self.feature_extractor(crops).cpu().numpy()[0]
        strack_pool = self.joint_stracks(self.tracked_stracks, self.lost_stracks)
        strack_pool = self.joint_stracks(strack_pool, self.removed_stracks)
        for track in strack_pool:
            if track.track_id == track_id:
                track.update_features(query_features)
                return
        reformated_box = box.copy()
        reformated_box[:, :4] = ops.xywh2xyxy(reformated_box[:, :4])
        # There's no track yet, make one
        fresh_track = BOTrack(reformated_box.squeeze()[:5], 1, 0, query_features)
        fresh_track.track_id = track_id
        self.removed_stracks.append(fresh_track)

    def get_dists(self, tracks, detections):
        """Get distances between tracks and detections using IoU and (optionally) ReID embeddings."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        # Just multiplies together detection confidence and IoU
        dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists