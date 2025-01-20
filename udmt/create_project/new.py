"""
UDMT (https://cabooster.github.io/UDMT/)
Author: Yixin Li
https://github.com/cabooster/UDMT
Licensed under Non-Profit Open Software License 3.0
"""


import os
import shutil
import warnings
from pathlib import Path
from udmt import DEBUG
from udmt.utils.auxfun_videos import VideoReader


def create_new_project(
    project,
    videos,
    working_directory=None,
    copy_videos=False,
    videotype=""
):

    from datetime import datetime as dt
    from udmt.utils import auxiliaryfunctions

    months_3letter = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    date = dt.today()
    month = months_3letter[date.month]
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    if working_directory is None:
        working_directory = "."
    wd = Path(working_directory).resolve()
    project_name = "{pn}-{date}".format(pn=project, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return os.path.join(str(project_path), "config.yaml")
    video_path = project_path / "videos"
    results_path = project_path / "tracking-results"
    training_datasets_path = project_path / "training-datasets"
    model_path = project_path / "models"
    tmp_path = project_path / "tmp"
    for p in [video_path, model_path, training_datasets_path, results_path,tmp_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # Add all videos in the folder. Multiple folders can be passed in a list, similar to the video files. Folders and video files can also be passed!
    vids = []
    for i in videos:
        # Check if it is a folder
        if os.path.isdir(i):
            vids_in_dir = [
                os.path.join(i, vp) for vp in os.listdir(i) if vp.endswith(videotype)
            ]
            vids = vids + vids_in_dir
            if len(vids_in_dir) == 0:
                print("No videos found in", i)
                print(
                    "Perhaps change the videotype, which is currently set to:",
                    videotype,
                )
            else:
                videos = vids
                print(
                    len(vids_in_dir),
                    " videos from the directory",
                    i,
                    "were added to the project.",
                )
        else:
            if os.path.isfile(i):
                vids = vids + [i]
            videos = vids

    videos = [Path(vp) for vp in videos]
    dirs = [results_path / Path(i.stem) for i in videos]
    all_dirs = [results_path / Path(i.stem) for i in videos] + [training_datasets_path / Path(i.stem) for i in videos] + [tmp_path / Path(i.stem) for i in videos]
    for p in all_dirs:
        """
        Creates directory under data
        """
        p.mkdir(parents=True, exist_ok=True)

    destinations = [video_path.joinpath(vp.name) for vp in videos]
    if copy_videos:
        print("Copying the videos")
        for src, dst in zip(videos, destinations):
            shutil.copy(
                os.fspath(src), os.fspath(dst)
            )  # https://www.python.org/dev/peps/pep-0519/
    # else:
    #     # creates the symlinks of the video and puts it in the videos directory.
    #     print("Attempting to create a symbolic link of the video ...")
    #     for src, dst in zip(videos, destinations):
    #         if dst.exists() and not DEBUG:
    #             raise FileExistsError("Video {} exists already!".format(dst))
    #         try:
    #             src = str(src)
    #             dst = str(dst)
    #             os.symlink(src, dst)
    #             print("Created the symlink of {} to {}".format(src, dst))
    #         except OSError:
    #             try:
    #                 import subprocess
    #
    #                 subprocess.check_call("mklink %s %s" % (dst, src), shell=True)
    #             except (OSError, subprocess.CalledProcessError):
    #                 print(
    #                     "Symlink creation impossible (exFat architecture?): "
    #                     "copying the video instead."
    #                 )
    #                 shutil.copy(os.fspath(src), os.fspath(dst))
    #                 print("{} copied to {}".format(src, dst))
    #         videos = destinations

    if copy_videos:
        videos = destinations  # in this case the *new* location should be added to the config file

    # adds the video list to the config.yaml file
    video_sets = {}
    for video in videos:
        print(video)
        try:
            # For windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path = os.path.realpath(video)]
            rel_video_path = str(Path.resolve(Path(video)))
        except:
            rel_video_path = os.readlink(str(video))

        try:
            vid = VideoReader(rel_video_path)
            video_sets[rel_video_path] = {"crop": ", ".join(map(str, vid.get_bbox()))}
        except IOError:
            warnings.warn("Cannot open the video file! Skipping to the next one...")
            os.remove(video)  # Removing the video or link from the project

    if not len(video_sets):
        # Silently sweep the files that were already written.
        shutil.rmtree(project_path, ignore_errors=True)
        warnings.warn(
            "No valid videos were found. The project was not created... "
            "Verify the video files and re-create the project."
        )
        return "nothingcreated"

    # Set values to config file:
    # if multianimal:  # parameters specific to multianimal project
    cfg_file, ruamelFile = auxiliaryfunctions.create_config_template()
    # cfg_file["individuals"] = ["individual1", "individual2", "individual3"]

    # common parameters:
    cfg_file["Task"] = project
    cfg_file["video_sets"] = video_sets
    cfg_file["project_path"] = str(project_path)
    cfg_file["date"] = d
    # cfg_file["epoch"] = 20
    # cfg_file[
    #     "batch_size"
    # ] = 8  # batch size during training (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242


    projconfigfile = os.path.join(str(project_path), "config.yaml")
    # Write dictionary to yaml  config file
    auxiliaryfunctions.write_config(projconfigfile, cfg_file)

    print('Generated "{}"'.format(project_path / "config.yaml"))
    print(
        "A new project with name %s is created at %s and a configurable file (config.yaml) is stored there."
        % (project_name, str(wd))
    )
    return projconfigfile
