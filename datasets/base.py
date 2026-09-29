from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import h5py
import threading


class multi_ch_nifti_default_Dataset(Dataset):
    def __init__(self, image_dataset, index_dataset, subjects, radius, image_size=(160, 160), flip=False, to_normal=False):
        self.image_size = image_size
        self.images = image_dataset  # float16 numpy array, stays in RAM
        self.indice = index_dataset
        self.subjects = subjects

        self.radius = radius

        self._length = self.indice.shape[0]
        self.flip = flip
        self.to_normal = to_normal # to [-1, 1]

        self.transform_no_flip = A.Compose([A.HorizontalFlip(p=0.0), ToTensorV2()])
        self.transform_flip    = A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()])

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        if index >= self._length:
            index = index - self._length
            transform = self.transform_flip
        else:
            transform = self.transform_no_flip

        slice_number = self.indice[index, 0]
        max_slice_number = self.indice[index, 1]

        if slice_number < self.radius:
            image = self.images[:,:,index-slice_number:index+self.radius+1].astype(np.float32)
            image = np.pad(image, ((0, 0), (0, 0), (self.radius-slice_number, 0)), mode='constant')
        elif slice_number > max_slice_number - self.radius:
            image = self.images[:,:,index-self.radius:index+max_slice_number-slice_number+1].astype(np.float32)
            image = np.pad(image, ((0, 0), (0, 0), (0, self.radius+slice_number-max_slice_number)), mode='constant')
        else:
            image = self.images[:,:,index-self.radius:index+self.radius+1].astype(np.float32)

        image = transform(image=image)['image'].float()

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        return image, self.subjects[index]

    def get_subject_names(self):
        return self.subjects


class multi_ch_nifti_lazy_Dataset(Dataset):
    """Lazy-loading version that reads slices from HDF5 on demand instead of
    loading the entire array into RAM."""

    def __init__(self, hdf5_path, dataset_key, index_dataset, subjects, radius,
                 image_size=(160, 160), flip=False, to_normal=False):
        self.image_size = image_size
        self.hdf5_path = hdf5_path
        self.dataset_key = dataset_key
        self.indice = index_dataset
        self.subjects = subjects
        self.radius = radius
        self._length = self.indice.shape[0]
        self.flip = flip
        self.to_normal = to_normal

        self.transform_no_flip = A.Compose([A.HorizontalFlip(p=0.0), ToTensorV2()])
        self.transform_flip    = A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()])

        # Thread-local storage for HDF5 file handles (h5py is not thread-safe)
        self._local = threading.local()

    def __getstate__(self):
        # Drop the unpicklable threading.local before sending to worker processes
        state = self.__dict__.copy()
        del state['_local']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def _get_dataset(self):
        """Return an open HDF5 dataset handle for the current thread/worker."""
        if not hasattr(self._local, 'dataset'):
            self._local.file = h5py.File(self.hdf5_path, 'r', swmr=True)
            self._local.dataset = self._local.file[self.dataset_key]
        return self._local.dataset

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        if index >= self._length:
            index = index - self._length
            transform = self.transform_flip
        else:
            transform = self.transform_no_flip

        ds = self._get_dataset()
        slice_number = self.indice[index, 0]
        max_slice_number = self.indice[index, 1]

        if slice_number < self.radius:
            image = ds[:, :, index - slice_number:index + self.radius + 1].astype(np.float32)
            image = np.pad(image, ((0, 0), (0, 0), (self.radius - slice_number, 0)), mode='constant')
        elif slice_number > max_slice_number - self.radius:
            image = ds[:, :, index - self.radius:index + max_slice_number - slice_number + 1].astype(np.float32)
            image = np.pad(image, ((0, 0), (0, 0), (0, self.radius + slice_number - max_slice_number)), mode='constant')
        else:
            image = ds[:, :, index - self.radius:index + self.radius + 1].astype(np.float32)

        image = transform(image=image)['image'].float()

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        return image, self.subjects[index]

    def get_subject_names(self):
        return self.subjects


class multi_ch_nifti_label_Dataset(Dataset):
    """Center-slice label map, flipped in lockstep with the image datasets.

    Deliberately NOT a subclass of the image datasets: labels are integer
    structure codes, so they must never be normalised to [-1, 1] and never
    interpolated. What they must share is the flip decision, which the image
    datasets derive purely from `index >= self._length`. That is deterministic
    for a given index, so building this with the same `index_dataset` and the
    same `flip` flag guarantees a label slice is flipped exactly when its image
    is -- without the two ever communicating.

    The same albumentations transforms as the image path are used rather than a
    hand-rolled `[:, ::-1]`, so flip parity holds by construction instead of by
    a convention someone can drift away from later.
    """

    def __init__(self, label_dataset, index_dataset, subjects, flip=False):
        self.labels = label_dataset
        self.indice = index_dataset
        self.subjects = subjects
        self._length = self.indice.shape[0]
        self.flip = flip

        self.transform_no_flip = A.Compose([A.HorizontalFlip(p=0.0), ToTensorV2()])
        self.transform_flip    = A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()])

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        if index >= self._length:
            index = index - self._length
            transform = self.transform_flip
        else:
            transform = self.transform_no_flip

        # The image datasets window [index-radius, index+radius+1], so the
        # centre channel -- the actual target slice -- is exactly `index`.
        label = self.labels[:, :, index][..., None]
        label = transform(image=label)['image']
        return label.long(), self.subjects[index]

    def get_subject_names(self):
        return self.subjects
