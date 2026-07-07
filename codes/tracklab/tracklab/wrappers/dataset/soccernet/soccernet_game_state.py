import json
import logging
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rich.prompt import Confirm
from SoccerNet.Downloader import SoccerNetDownloader

from tracklab.datastruct import TrackingDataset, TrackingSet
from tracklab.utils import xywh_to_ltwh
from tracklab.utils.progress import progress


log = logging.getLogger(__name__)


DEFAULT_SPLITS = ("train", "valid", "test", "challenge")


class SoccerNetGameState(TrackingDataset):
    def __init__(
        self,
        dataset_path: str,
        nvid: int = -1,
        vids_dict: dict | None = None,
        eval_set: str | None = None,
        start_vid: int | None = None,
        end_vid: int | None = None,
        *args,
        **kwargs,
    ):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            download_dataset(self.dataset_path)
        assert self.dataset_path.exists(), (
            f"'{self.dataset_path}' directory does not exist. Please check the path or download the dataset "
            "following the instructions here: https://github.com/SoccerNet/sn-gamestate"
        )

        vids_dict = vids_dict or {}
        splits = _configured_splits(self.dataset_path, vids_dict, eval_set)
        sets = {}
        for split in splits:
            split_path = self.dataset_path / split
            if not split_path.exists():
                log.warning(
                    "Warning: The '%s' set does not exist in the SoccerNetGS dataset at '%s'.",
                    split,
                    self.dataset_path,
                )
                continue
            vids_filter = vids_dict.get(split, [])
            sets[split] = load_set(split_path, nvid, vids_filter, start_vid, end_vid)

        assert sets, f"No SoccerNetGS splits found under '{self.dataset_path}'. Tried: {splits}"

        # Subsampling is already handled by load_set so custom split filters can be applied before loading frames.
        super().__init__(dataset_path, sets, nvid=-1, vids_dict=None, *args, **kwargs)

    def process_trackeval_results(self, results, dataset_config, eval_config):
        combined_results = results["SUMMARIES"]["cls_comb_det_av"]
        combined_results["GS-HOTA"] = combined_results.pop("HOTA")
        combined_results["GS-HOTA"] = {
            k.replace("HOTA", "GS-HOTA"): v for k, v in combined_results["GS-HOTA"].items()
        }
        log.info(
            "SoccerNet Game State Reconstruction performance GS-HOTA = %s%%",
            combined_results["GS-HOTA"]["GS-HOTA"],
        )
        return combined_results

    def save_for_eval(
        self,
        detections: pd.DataFrame,
        image_metadatas: pd.DataFrame,
        video_metadatas: pd.DataFrame,
        save_folder: str,
        bbox_column_for_eval="bbox_ltwh",
        save_classes=False,
        is_ground_truth=False,
        save_zip=True,
    ):
        if is_ground_truth:
            return
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)

        detections = self.soccernet_encoding(detections.copy(), supercategory="object")
        camera_metadata = self.soccernet_encoding(image_metadatas.copy(), supercategory="camera")
        pitch_metadata = self.soccernet_encoding(image_metadatas.copy(), supercategory="pitch")
        predictions = pd.concat([detections, camera_metadata, pitch_metadata], ignore_index=True)
        zf_save_path = save_path.parents[1] / f"{save_path.parent.name}.zip"

        for video_id, video in video_metadatas.iterrows():
            file_path = save_path / f"{video['name']}.json"
            video_predictions_df = predictions[predictions["video_id"] == str(video_id)].copy()
            if video_predictions_df.empty:
                continue

            video_predictions_df.sort_values(by="id", inplace=True)
            video_predictions = [
                {k: int(v) if k == "track_id" else v for k, v in row.items() if np.all(pd.notna(v))}
                for row in video_predictions_df.to_dict(orient="records")
            ]
            with file_path.open("w") as fp:
                json.dump({"predictions": video_predictions}, fp, indent=2)
            if save_zip:
                with zipfile.ZipFile(zf_save_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(file_path, arcname=f"{save_path.name}/{file_path.name}")

    @staticmethod
    def soccernet_encoding(dataframe: pd.DataFrame, supercategory):
        dataframe["supercategory"] = supercategory
        dataframe = dataframe.replace({np.nan: None})
        if supercategory == "object":
            dataframe.dropna(subset=["track_id", "bbox_ltwh", "bbox_pitch"], how="any", inplace=True)
            dataframe = dataframe.rename(columns={"bbox_ltwh": "bbox_image", "jersey_number": "jersey"})
            dataframe["attributes"] = [
                {"role": row.get("role"), "jersey": row.get("jersey"), "team": row.get("team")}
                for _, row in dataframe.iterrows()
            ]
            dataframe["id"] = dataframe.index
            dataframe = dataframe[
                dataframe.columns.intersection(
                    [
                        "id",
                        "image_id",
                        "video_id",
                        "track_id",
                        "supercategory",
                        "category_id",
                        "attributes",
                        "bbox_image",
                        "bbox_pitch",
                    ]
                )
            ]
            dataframe["bbox_image"] = dataframe["bbox_image"].apply(transform_bbox_image)
        elif supercategory == "camera":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 6
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "01")
            dataframe = dataframe[
                dataframe.columns.intersection(
                    ["id", "image_id", "video_id", "supercategory", "category_id", "parameters", "relative_mean_reproj", "accuracy@5"]
                )
            ]
        elif supercategory == "pitch":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 5
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "00")
            dataframe = dataframe[
                dataframe.columns.intersection(["id", "image_id", "video_id", "supercategory", "category_id", "lines"])
            ]

        dataframe["video_id"] = dataframe["video_id"].apply(str)
        dataframe["image_id"] = dataframe["image_id"].apply(str)
        dataframe["id"] = dataframe["id"].apply(str)
        return dataframe.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def _configured_splits(dataset_path: Path, vids_dict: dict, eval_set: str | None):
    configured = []
    if eval_set:
        configured.append(eval_set)
        configured.extend(split for split, videos in vids_dict.items() if videos)
    else:
        configured.extend(vids_dict.keys())
    if not configured:
        configured.extend(split for split in DEFAULT_SPLITS if (dataset_path / split).exists())

    seen = set()
    splits = []
    for split in configured:
        if split not in seen:
            splits.append(split)
            seen.add(split)
    return splits or list(DEFAULT_SPLITS)


def _video_number(video_name: str):
    try:
        return int(video_name.split("-")[-1])
    except ValueError:
        return None


def _filter_video_range(video_list, start_vid, end_vid):
    if start_vid is None and end_vid is None:
        return video_list

    filtered = []
    start = int(start_vid) if start_vid is not None else None
    end = int(end_vid) if end_vid is not None else None
    for video in video_list:
        number = _video_number(video)
        if number is None:
            continue
        candidates = (number, 10000 + number)
        in_range = any(
            (start is None or candidate >= start) and (end is None or candidate <= end)
            for candidate in candidates
        )
        if not in_range:
            continue
        filtered.append(video)

    if not filtered:
        log.warning(
            "dataset.start_vid/end_vid selected no videos from %s. Ignoring that range.",
            video_list[:5],
        )
        return video_list
    return filtered


def transform_bbox_image(row):
    row = row.astype(float)
    return {"x": row[0], "y": row[1], "w": row[2], "h": row[3]}


def extract_category(attributes):
    role = attributes["role"]
    if role in ("goalkeeper", "player"):
        team = attributes["team"]
        jersey = attributes["jersey"]
        jersey_number = int(jersey) if jersey is not None and str(jersey).isdigit() else None
        return f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    if role in ("referee", "ball", "other"):
        return role
    raise AssertionError(f"Unknown SoccerNet role: {role}")


def dict_to_df_detections(annotation_dict, categories_list):
    df = pd.DataFrame.from_dict(annotation_dict)
    annotations_pitch_camera = df.loc[df["supercategory"] != "object"]
    df = df.loc[df["supercategory"] == "object"].copy()

    df["bbox_ltwh"] = df.apply(
        lambda row: xywh_to_ltwh(
            [row["bbox_image"]["x_center"], row["bbox_image"]["y_center"], row["bbox_image"]["w"], row["bbox_image"]["h"]]
        ),
        axis=1,
    )
    df["team"] = df.apply(lambda row: row["attributes"]["team"], axis=1)
    df["team_cluster"] = (df["team"] == "left").astype(float)
    df["role"] = df.apply(lambda row: row["attributes"]["role"], axis=1)
    df["jersey_number"] = df.apply(lambda row: row["attributes"]["jersey"], axis=1)
    df["position"] = None
    df["category"] = df.apply(lambda row: extract_category(row["attributes"]), axis=1)
    df["track_id"] = df["track_id"].astype(int)

    columns = [
        "id",
        "image_id",
        "track_id",
        "bbox_ltwh",
        "bbox_pitch",
        "team_cluster",
        "team",
        "role",
        "jersey_number",
        "position",
        "category",
    ]
    df = df[columns]
    video_level_categories = list(df["category"].unique())
    return df, annotations_pitch_camera, video_level_categories


def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def video_dir_to_dfs(args):
    dataset_path = args["dataset_path"]
    video_folder = args["video_folder"]
    split = args["split"]
    split_id = args["split_id"]
    video_folder_path = os.path.join(dataset_path, video_folder)
    if not os.path.isdir(video_folder_path):
        return None

    annotation_pitch_camera_df = None
    detections_df = None
    video_level_categories = []

    if not (Path(video_folder_path) / "Labels-GameState.json").exists():
        img_folder_path = os.path.join(video_folder_path, "img1")
        video_id = str(video_folder.split("-")[-1])
        nframes = len(os.listdir(img_folder_path))
        video_metadata = {"id": video_id, "name": video_folder}
        img_metadata_df = pd.DataFrame(
            {
                "frame": [i for i in range(0, nframes)],
                "id": [f"{split_id}{video_id}{i:06d}" for i in range(1, nframes + 1)],
                "video_id": video_id,
                "file_path": [os.path.join(img_folder_path, f"{i:06d}.jpg") for i in range(1, nframes + 1)],
                "nframes": nframes,
            }
        )
    else:
        gamestate_path = os.path.join(video_folder_path, "Labels-GameState.json")
        gamestate_data = read_json_file(gamestate_path)

        info_data = gamestate_data["info"]
        images_data = gamestate_data["images"]
        annotations_data = gamestate_data["annotations"]
        categories_data = gamestate_data["categories"]
        video_id = info_data.get("id", str(video_folder.split("-")[-1]))

        detections_df, annotation_pitch_camera_df, video_level_categories = dict_to_df_detections(
            annotations_data, categories_data
        )
        detections_df["video_id"] = video_id
        detections_df["person_id"] = detections_df["track_id"].astype(str) + detections_df["video_id"].astype(str)
        detections_df["visibility"] = 1

        nframes = int(info_data.get("seq_length", 0))
        video_metadata = {
            "id": video_id,
            "name": info_data.get("name", ""),
            "nframes": nframes,
            "frame_rate": int(info_data.get("frame_rate", 0)),
            "seq_length": nframes,
            "im_width": int(images_data[0].get("width", 0)),
            "im_height": int(images_data[0].get("height", 0)),
            "game_id": int(info_data.get("game_id", 0)),
            "action_position": int(info_data.get("action_position", 0)),
            "action_class": info_data.get("action_class", ""),
            "visibility": info_data.get("visibility", ""),
            "clip_start": int(info_data.get("clip_start", 0)),
            "game_time_start": info_data.get("game_time_start", " - ").split(" - ")[1],
            "game_time_stop": info_data.get("game_time_stop", " - ").split(" - ")[1],
            "clip_stop": int(info_data.get("clip_stop", 0)),
            "num_tracklets": int(info_data.get("num_tracklets", 0)),
            "half_period_start": int(info_data.get("game_time_start", "0 - ").split(" - ")[0]),
            "half_period_stop": int(info_data.get("game_time_stop", "0 - ").split(" - ")[0]),
        }
        img_folder_path = os.path.join(video_folder_path, info_data.get("im_dir", "img1"))
        img_metadata_df = pd.DataFrame(
            {
                "frame": [i for i in range(0, nframes)],
                "id": [i["image_id"] for i in images_data],
                "video_id": video_id,
                "file_path": [os.path.join(img_folder_path, i["file_name"]) for i in images_data],
                "is_labeled": [i["is_labeled"] for i in images_data],
                "nframes": nframes,
            }
        )
        annotation_pitch_camera_df["video_id"] = video_id

    return {
        "video_metadata": video_metadata,
        "image_metadata": img_metadata_df,
        "detections": detections_df,
        "annotations_pitch_camera": annotation_pitch_camera_df,
        "video_level_categories": video_level_categories,
    }


def load_set(dataset_path, nvid=-1, vids_filter_set=None, start_vid=None, end_vid=None):
    video_metadatas_list = []
    image_metadata_list = []
    annotations_pitch_camera_list = []
    detections_list = []
    categories_list = []
    split = os.path.basename(dataset_path)
    video_list = sorted(os.listdir(dataset_path))

    if vids_filter_set:
        missing_videos = set(vids_filter_set) - set(video_list)
        if missing_videos:
            log.warning("Warning: videos from dataset.vids_dict missing in %s set: %s", split, missing_videos)
        video_list = [video for video in video_list if video in vids_filter_set]

    video_list = _filter_video_range(video_list, start_vid, end_vid)

    if nvid > 0:
        video_list = video_list[:nvid]

    assert video_list, (
        f"After applying filtering, no videos left in the '{split}' set, "
        "please fix dataset.vids_dict or dataset.start_vid/end_vid."
    )

    split_id = {"train": 1, "valid": 2, "test": 3, "challenge": 4, "franz": 5, "sn500": 6}.get(split, 9)
    args = [
        {"dataset_path": dataset_path, "video_folder": video_folder, "split": split, "split_id": split_id}
        for video_folder in video_list
    ]

    with Pool() as pool:
        for result in progress(
            pool.imap_unordered(video_dir_to_dfs, args),
            total=len(args),
            desc=f"Loading SoccerNetGS '{split}' set videos",
        ):
            if result is not None:
                video_metadatas_list.append(result["video_metadata"])
                image_metadata_list.append(result["image_metadata"])
                detections_list.append(result["detections"])
                annotations_pitch_camera_list.append(result["annotations_pitch_camera"])
                categories_list += result["video_level_categories"]

    if len(categories_list) == 0:
        video_metadata = pd.DataFrame(video_metadatas_list)
        image_metadata = pd.concat(image_metadata_list, ignore_index=True)
        detections = None
        image_metadata.set_index("id", drop=False, inplace=True)
        image_gt = image_metadata.copy()
        video_metadata.set_index("id", drop=False, inplace=True)
    else:
        categories_list = [
            {"id": i + 1, "name": category, "supercategory": "person"}
            for i, category in enumerate(sorted(set(categories_list)))
        ]

        for video_metadata in video_metadatas_list:
            video_metadata["categories"] = categories_list

        video_metadata = pd.DataFrame(video_metadatas_list)
        image_metadata = pd.concat(image_metadata_list, ignore_index=True)
        detections = pd.concat(detections_list, ignore_index=True)

        detections["person_id"] = pd.factorize(detections["person_id"])[0]
        detections = detections.sort_values(by=["video_id", "image_id", "track_id"], ascending=[True, True, True])
        detections["id"] = (
            detections["video_id"].astype(str)
            + "_"
            + detections["image_id"].astype(str)
            + "_"
            + detections["track_id"].astype(str)
        )

        pitch_camera = pd.concat(annotations_pitch_camera_list, ignore_index=True)
        pitch_gt = pitch_camera[["image_id", "video_id", "lines"]][pitch_camera.supercategory == "pitch"].set_index(
            "image_id", drop=True
        )
        image_gt = image_metadata.copy().set_index("id", drop=False)
        image_gt["lines"] = pitch_gt["lines"]

        category_to_id = {category["name"]: category["id"] for category in categories_list}
        detections["category_id"] = detections["category"].apply(lambda x: category_to_id[x])

        detections.set_index("id", drop=False, inplace=True)
        image_metadata.set_index("id", drop=False, inplace=True)
        video_metadata.set_index("id", drop=False, inplace=True)

        video_metadata_columns = [
            "name",
            "nframes",
            "frame_rate",
            "seq_length",
            "im_width",
            "im_height",
            "game_id",
            "action_position",
            "action_class",
            "visibility",
            "clip_start",
            "game_time_start",
            "clip_stop",
            "game_time_stop",
            "num_tracklets",
            "half_period_start",
            "half_period_stop",
            "categories",
        ]
        video_metadata_columns.extend(set(video_metadata.columns) - set(video_metadata_columns))
        video_metadata = video_metadata[video_metadata_columns]

        image_metadata_columns = ["video_id", "frame", "file_path", "is_labeled", "nframes"]
        image_metadata_columns.extend(set(image_metadata.columns) - set(image_metadata_columns))
        image_metadata = image_metadata[image_metadata_columns]

        detections_column_ordered = ["image_id", "video_id", "track_id", "person_id", "bbox_ltwh", "visibility"]
        detections_column_ordered.extend(set(detections.columns) - set(detections_column_ordered))
        detections = detections[detections_column_ordered]
        detections["bbox_conf"] = 1

    return TrackingSet(video_metadata, image_metadata, detections, image_gt)


def download_dataset(dataset_path, splits=DEFAULT_SPLITS):
    my_soccernet_downloader = SoccerNetDownloader(LocalDirectory=str(dataset_path))
    download = Confirm.ask(f"Do you want to download the datasets automatically ? [i]({'/'.join(splits)})[/i]")
    if download:
        my_soccernet_downloader.downloadDataTask(task="gamestate-2025", split=splits)
        for split in splits:
            log.info("Unzipping %s split...", split)
            with zipfile.ZipFile(dataset_path / "gamestate-2025" / f"{split}.zip", "r") as zf:
                zf.extractall(dataset_path / split)
