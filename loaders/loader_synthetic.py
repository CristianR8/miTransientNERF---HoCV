import collections
import json
import os

import imageio.v2 as imageio
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import scipy
import zipfile
from .utils import Rays
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.dataset_utils import read_h5
from tqdm import tqdm


# MITransient scenes produced for this project store all cameras in one HDF5
# file, unlike the official release where every camera lives in a separate
# file.  Keep the official loader below intact and select this code path only
# when the HDF5 file has these three datasets.
_MITSUBA_KEYS = {"images", "poses", "transients"}
_MITSUBA_MAX_CACHE = {}


def _find_mitsuba_scene(root_fp: str, subject_id: str):
    """Return a consolidated MITransient scene path, or ``None``.

    ``data_root_fp`` may point directly to ``scene_0.h5`` or to the directory
    containing it.  The latter also accepts ``<subject_id>.h5``.
    """
    candidates = [root_fp]
    if os.path.isdir(root_fp):
        candidates.extend([
            os.path.join(root_fp, f"{subject_id}.h5"),
            os.path.join(root_fp, "scene_0.h5"),
        ])
    for path in candidates:
        if not path.endswith(".h5") or not os.path.isfile(path):
            continue
        with h5py.File(path, "r") as data:
            if _MITSUBA_KEYS.issubset(data.keys()):
                return path
    return None


def _mitsuba_view_split(num_cameras: int, num_views: int):
    """Create nested 2/3/5-view sparse-training splits.

    The authors train separate 2-, 3-, and 5-view experiments and evaluate on
    held-out cameras.  Fibonacci-camera indices are evenly spaced here, making
    the splits deterministic and nested: 2={0,24}, 3={0,12,24}, and
    5={0,6,12,18,24} for this 25-camera dataset.
    """
    if num_views not in (2, 3, 5):
        raise ValueError("MITransient sparse-view experiments support num_views=2, 3, or 5.")
    if num_views > num_cameras:
        raise ValueError(f"Requested {num_views} views, but the scene has {num_cameras} cameras.")
    train_ids = np.rint(np.linspace(0, num_cameras - 1, num_views)).astype(np.int64)
    test_ids = np.setdiff1d(np.arange(num_cameras, dtype=np.int64), train_ids)
    return train_ids, test_ids


def _mitsuba_transient_max(path: str):
    """Find the global transient maximum one HDF5 chunk/view at a time."""
    if path in _MITSUBA_MAX_CACHE:
        return _MITSUBA_MAX_CACHE[path]

    max_value = 0.0
    with h5py.File(path, "r") as data:
        transients = data["transients"]
        # MITransient files are chunked as one complete view. Reading 32-bin
        # slices would therefore decompress the same ~775 MiB chunk repeatedly.
        for view_id in range(transients.shape[0]):
            max_value = max(max_value, float(np.max(transients[view_id])))
    if not np.isfinite(max_value) or max_value <= 0:
        raise ValueError(f"Invalid transient maximum in {path}: {max_value}")
    _MITSUBA_MAX_CACHE[path] = max_value
    return max_value

def _load_renderings(root_fp: str, subject_id: str, split: str, have_images=True, img_shape=(256, 256)):
    """Load images from disk."""
    # if not root_fp.startswith("/"):
    #     # allow relative path. e.g., "./data/nerf_synthetic/"
    #     root_fp = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "..",
    #         "..",
    #         root_fp,
    #     )

    data_dir = root_fp
    with open(
            os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    if have_images:
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            number = int(frame["file_path"].split("_")[-1])
            fname = os.path.join(data_dir, f"{number:03d}" + ".png")

            # fname = os.path.join(data_dir, frame["file_path"] + ".png")
            rgba = imageio.imread(fname)
            camtoworlds.append(frame["transform_matrix"])
            images.append(rgba)

        images = np.stack(images, axis=0)
        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = images.shape[1:3]
    else:
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            camtoworlds.append(frame["transform_matrix"])

        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = img_shape

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


def _load_renderings_transient(root_fp: str, subject_id: str, split: str, num_views= None, have_images=True, img_shape=(256, 256), gamma=False):
    """Load images from disk."""
    # if not root_fp.startswith("/"):
    #     # allow relative path. e.g., "./data/nerf_synthetic/"
    #     root_fp = os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "..",
    #         "..",
    #         root_fp,
    #     )

    data_dir = root_fp
    if split == "train": tname = f"train_v{num_views}"
    else: tname = split

    with open(
            os.path.join(data_dir, "transforms_{}.json".format(tname)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    if have_images:
        for i in tqdm(range(len(meta["frames"]))):
            frame = meta["frames"][i]
            number = int(frame["file_path"].split("_")[-1])

            try:
                files_dir = os.path.join(data_dir, split)
                fname = os.path.join(files_dir, f"{split}_{number:03d}" + ".h5")
                rgba = read_h5(fname)
            except:     
                try:
                    files_dir = os.path.join(data_dir, "test")
                    fname = os.path.join(files_dir, f"test_{number:03d}" + ".h5")
                    rgba = read_h5(fname)
                except:         
                    try:
                        files_dir = os.path.join(data_dir, "test")
                        fname = os.path.join(files_dir, f"test_{number:03d}" + ".h5")
                        archive = zipfile.ZipFile(f"{fname}.zip")
                        file = archive.open(f"test_{number:03d}" + ".h5")
                        rgba = read_h5(file)
                        file.close()
                    except:
                        pass



            rgba = rgba[..., :3]

            if gamma:
                print("using gamma")
                rgba_sum = rgba.sum(-2)
                rgba_sum_normalized = rgba_sum/rgba_sum.max()
                rgba_sum_norm_gamma = rgba_sum_normalized**(1/2.2)
                rgba = (rgba*rgba_sum_norm_gamma[..., None, :])/(rgba_sum[..., None, :]+1e-10)

            camtoworlds.append(frame["transform_matrix"])
            rgba = torch.clip(torch.Tensor(rgba), 0, None)
            images.append(torch.Tensor(rgba))


        images = torch.stack(images, axis=0)

        if split == "test":
            quotient = images.shape[1]//img_shape[0]
            times_downsample = int(np.log2(quotient))
        
            for i in range(times_downsample):
                images = (images[:, 1::2, ::2] + images[:, ::2, ::2] + images[:, 1::2, 1::2] + images[:, ::2, 1::2])/4


        if not gamma:
            #np.save(os.path.join(data_dir, "max.npy"), torch.max(images).numpy())
            max = torch.max(images)
            images /= torch.max(images)

        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = images.shape[1:3]
    else:
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            camtoworlds.append(frame["transform_matrix"])

        camtoworlds = np.stack(camtoworlds, axis=0)

        h, w = img_shape

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal, max



class SubjectLoaderTransient(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]

    # WIDTH, HEIGHT = 64, 64
    NEAR, FAR = 0, 6
    OPENGL_CAMERA = True

    def __init__(
            self,
            subject_id: str,
            root_fp: str,
            split: str,
            color_bkgd_aug: str = "black",
            num_rays: int = None,
            near: float = None,
            far: float = None,
            batch_over_images: bool = True,
            have_images=True,
            img_shape=(256, 256),
            n_bins=10000, 
            testing=False, 
            rfilter_sigma=0.3, 
            scene=None, 
            sample_as_per_distribution = True, 
            gamma=False,
            num_views = None
    ):
        super().__init__()
        self.testing = testing 
        # assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.sample_as_per_distribution = sample_as_per_distribution

        self.HEIGHT, self.WIDTH = img_shape
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (
                split in ["train", "trainval"]
        )
        self.rep = 0
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.have_images = have_images
        self.rfilter_sigma = rfilter_sigma
        self.n_bins = n_bins
        self._mitsuba_path = _find_mitsuba_scene(root_fp, subject_id)
        self._is_mitsuba_scene = self._mitsuba_path is not None

        if self._is_mitsuba_scene:
            if split == "trainval":
                raise ValueError("MITransient scenes use the explicit train/test sparse-view split, not trainval.")
            if num_views is None:
                raise ValueError(
                    "Pass num_views (2, 3, or 5) for a consolidated MITransient scene so test views "
                    "are held out consistently with training."
                )
            with h5py.File(self._mitsuba_path, "r") as data:
                transients = data["transients"]
                images = data["images"]
                poses = data["poses"]
                if transients.ndim != 5 or transients.shape[-1] != 3:
                    raise ValueError("Expected transients with shape [views, height, width, bins, 3].")
                if images.shape[:3] != transients.shape[:3] or poses.shape[0] != transients.shape[0]:
                    raise ValueError("MITransient images, poses, and transients do not describe the same cameras.")
                if tuple(transients.shape[1:3]) != (self.HEIGHT, self.WIDTH):
                    raise ValueError(
                        f"Configured image shape {(self.HEIGHT, self.WIDTH)} does not match "
                        f"the HDF5 data {tuple(transients.shape[1:3])}."
                    )
                if transients.shape[3] != self.n_bins:
                    raise ValueError(
                        f"Configured n_bins={self.n_bins}, but {self._mitsuba_path} contains "
                        f"{transients.shape[3]} bins. Set n_bins to that value."
                    )
                transient_shape = tuple(transients.shape)
                poses = np.asarray(poses)

            train_ids, test_ids = _mitsuba_view_split(poses.shape[0], num_views)
            self.view_ids = train_ids if split == "train" else test_ids
            # Some exporters retain a singleton channel after each 4x4 pose.
            self.camtoworlds = poses[self.view_ids]
            if self.camtoworlds.ndim == 4 and self.camtoworlds.shape[-1] == 1:
                self.camtoworlds = self.camtoworlds[..., 0]
            if self.camtoworlds.shape[-2:] != (4, 4):
                raise ValueError("Expected camera-to-world poses with shape [views, 4, 4] (or [views, 4, 4, 1]).")

            # A 60-degree vertical FOV at 256 pixels gives this focal length.
            # It is derived from the supplied Mitsuba sensor XML, not guessed
            # from the pose matrices.
            self.focal = 0.5 * self.HEIGHT / np.tan(np.deg2rad(60.0) / 2.0)
            self.max = torch.tensor(_mitsuba_transient_max(self._mitsuba_path), dtype=torch.float32)
            self._mitsuba_train_cache = None
            if self.training:
                # The source file stores each entire view in one compressed
                # chunk. Cache only the sparse training views in host RAM so a
                # random ray does not decompress a 775 MiB chunk on every read.
                cache_shape = (len(self.view_ids),) + transient_shape[1:]
                self._mitsuba_train_cache = np.empty(cache_shape, dtype=np.float32)
                with h5py.File(self._mitsuba_path, "r") as data:
                    for local_view, source_view in enumerate(self.view_ids):
                        self._mitsuba_train_cache[local_view] = data["transients"][source_view]
            # Preserve the existing public attribute. Pixels are fetched from
            # disk on demand so the 20+ GB HDF5 volume is never loaded at once.
            self.images = torch.empty(0, dtype=torch.float32)

        elif split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings_transient(
                root_fp, subject_id, "train", gamma=gamma
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings_transient(
                root_fp, subject_id, "val", gamma=gamma
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
            self.images = torch.from_numpy(self.images).to(torch.float32)

            # ste for transient
            self.images = torch.reshape(self.images, (-1, self.HEIGHT, self.WIDTH, self.n_bins*3))
            # assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

        elif have_images:
            self.images, self.camtoworlds, self.focal, self.max = _load_renderings_transient(
                root_fp, subject_id, split, gamma=gamma, img_shape=img_shape, num_views=num_views
            )
            self.images = self.images.to(torch.float32)
            assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        else:
            _, self.camtoworlds, self.focal = _load_renderings(
                root_fp, subject_id, split, have_images=have_images, img_shape=img_shape, num_views=num_views
            )

        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)

    def _load_mitsuba_pixels(self, image_id, x, y):
        """Read only the selected transient rays from a consolidated scene."""
        image_id = image_id.detach().cpu().numpy().astype(np.int64)
        x = x.detach().cpu().numpy().astype(np.int64)
        y = y.detach().cpu().numpy().astype(np.int64)
        if self._mitsuba_train_cache is not None:
            output = self._mitsuba_train_cache[image_id, y, x]
            return torch.from_numpy(np.asarray(output)) / self.max

        output = np.empty((len(image_id), self.n_bins, 3), dtype=np.float32)
        # Repeated samples (from spatial-filter replication) need only one HDF5
        # read. h5py point indexing is restrictive, so use scalar reads for the
        # small set of unique training rays.
        locations = {}
        for position, key in enumerate(zip(image_id, x, y)):
            locations.setdefault(key, []).append(position)
        with h5py.File(self._mitsuba_path, "r") as data:
            transients = data["transients"]
            for (local_view, pixel_x, pixel_y), positions in locations.items():
                output[positions] = transients[self.view_ids[local_view], pixel_y, pixel_x, :, :]
        return torch.from_numpy(output) / self.max

    def _load_mitsuba_image(self, image_id):
        """Read one full held-out view for the authors' TensorBoard/eval path."""
        with h5py.File(self._mitsuba_path, "r") as data:
            image = np.asarray(data["transients"][self.view_ids[image_id]], dtype=np.float32)
        return (torch.from_numpy(image) / self.max).to(self.camtoworlds.device)

    def __len__(self):
        return len(self.camtoworlds)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        # pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
        if rgba is not None:
            pixels = rgba.to(self.camtoworlds.device)
        else:
            pixels = rgba

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.camtoworlds.device)

        # pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index, rep=None, num_rays=None):
        """Fetch the data (it maybe cached for multiple batches)."""
        if num_rays==None:
            num_rays = self.num_rays
        if rep==None:
            rep = self.rep

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.camtoworlds) if self._is_mitsuba_scene else len(self.images),
                    size=(num_rays,),
                    device=self.camtoworlds.device if self._is_mitsuba_scene else self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.images.device
            )
            x = x.repeat(rep)
            y = y.repeat(rep)
            image_id = image_id.repeat(rep)


            if self._is_mitsuba_scene:
                rgba = self._load_mitsuba_pixels(image_id, x, y)
            else:
                rgba = self.images[image_id, y, x]  # (num_rays, 4)

        elif self.testing:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device="cpu"),
                torch.arange(self.HEIGHT, device="cpu"),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            x = x.repeat(rep)
            y = y.repeat(rep)
            # image_id = image_id.repeat(rep)
            if self._is_mitsuba_scene:
                rgba = self._load_mitsuba_image(index)[y, x]
            else:
                try:
                    rgba = self.images[image_id, y, x]  # (num_rays, 4)
                except:
                    rgba = None

        elif self.have_images:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.camtoworlds.device),
                torch.arange(self.HEIGHT, device=self.camtoworlds.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
            if self._is_mitsuba_scene:
                rgba = self._load_mitsuba_image(index)[y, x]
            else:
                rgba = self.images[image_id, y, x]  # (num_rays, 4)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.camtoworlds.device),
                torch.arange(self.HEIGHT, device=self.camtoworlds.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        
        scale = self.rfilter_sigma
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        bounds_max = [4*scale]*x.shape[0]
        loc = 0
        if self.training:
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.rep, prob_dithering=self.sample_as_per_distribution)
            s_x = (torch.clip(x + torch.from_numpy(s_x), 0, self.WIDTH).to(self.camtoworlds.device)).to(torch.float32)
            s_y = (torch.clip(y + torch.from_numpy(s_y), 0, self.HEIGHT).to(self.camtoworlds.device)).to(torch.float32)
            weights = torch.Tensor(weights).to(self.camtoworlds.device)
            #s_x = x.to(self.camtoworlds.device).to(torch.float32)
            #s_y = y.to(self.camtoworlds.device).to(torch.float32)

        elif self.testing:
            s_x, s_y, weights = spatial_filter(x, y, sigma=scale, rep = self.rep, prob_dithering=self.sample_as_per_distribution, normalize=False)
            s_x = (torch.clip(x + torch.from_numpy(s_x), 0, self.WIDTH).to(self.camtoworlds.device)).to(torch.float32)
            s_y = (torch.clip(y + torch.from_numpy(s_y), 0, self.HEIGHT).to(self.camtoworlds.device)).to(torch.float32)
            weights = torch.Tensor(weights).to(self.camtoworlds.device)
            #s_x = x.to(self.camtoworlds.device).to(torch.float32)
            #s_y = y.to(self.camtoworlds.device).to(torch.float32)
        else: 
            s_x = x
            s_y = y

        camera_dirs = F.pad(
            torch.stack(
                [
                    (s_x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (s_y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  

        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
            # here
            rgba = torch.reshape(rgba, (-1,self.n_bins*3))
        elif self.testing:
            origins = torch.reshape(origins, (-1, 3))
            viewdirs = torch.reshape(viewdirs, (-1, 3))
            # here
            try: rgba = torch.reshape(rgba, (-1,self.n_bins*3))
            except: rgba = None

        elif self.have_images:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, self.n_bins * 3))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = None

        rays = Rays(origins=origins, viewdirs=viewdirs)
        if self.training or self.testing:
            return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "weights":weights
        }

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }


def spatial_filter(x, y, sigma, rep, prob_dithering=True, normalize=True):
    pdf_fn = lambda x: np.exp(-x/(2*sigma**2)) - np.exp(-16)
    if prob_dithering:
        bounds_max = [4*sigma]*x.shape[0]
        loc = 0
        s_x = scipy.stats.truncnorm.rvs((-np.array(bounds_max)-loc)/sigma, (np.array(bounds_max)-loc)/sigma, loc=loc, scale=sigma)
        s_y = scipy.stats.truncnorm.rvs((-np.array(bounds_max)-loc)/sigma, (np.array(bounds_max)-loc)/sigma, loc=loc, scale=sigma)
        weights = np.ones_like(s_x)*1/rep
    
    else:
        s_x = np.random.uniform(low=-4*sigma, high=4*sigma, size=(rep, x.shape[0]//rep))
        s_y = np.random.uniform(low=-4*sigma, high=4*sigma, size=(rep, x.shape[0]//rep))
        dists = (s_x**2 + s_y**2)
        weights = pdf_fn(dists)
        if normalize:
            weights = weights/weights.sum(0)[None, :]
        s_x = s_x.flatten()
        s_y = s_y.flatten()
        weights = weights.flatten()
    
    return s_x, s_y, weights
