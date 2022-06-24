import imageio
import os
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def make_gif(plot_dir, client_name: str, gif_name="inducing", duration=3):
    """Build GIF of inducing points across training"""

    export_name = f"{plot_dir}/{gif_name}_{client_name}.gif"

    _train_dir = os.path.join(plot_dir, f"training/{client_name}")
    filenames = next(os.walk(_train_dir), (None, None, []))[2]
    filenames.sort(key=natural_keys)

    if len(filenames) == 0:
        return

    frame_duration = duration / len(filenames)  # 3 sec total dur

    images = list(map(lambda filename: imageio.imread(os.path.join(_train_dir, filename)), filenames))

    imageio.mimsave(export_name, images, format="GIF", duration=frame_duration)
