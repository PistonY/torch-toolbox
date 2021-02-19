from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import _setup_size

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class DynamicRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self._active_size = self.size

    @property
    def active_size(self):
        return self._active_size

    @active_size.setter
    def active_size(self, size):
        self._active_size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self._active_size, self.interpolation)
