# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
import nvidia.dali as dali
# from nvidia.dali import types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


class TrainPipe(dali.pipeline.Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id, crop, color_jit=0.4):
        super(TrainPipe, self).__init__(batch_size, num_threads, device_id)
        device_memory_padding = 211025920
        host_memory_padding = 140544512

        self.input = dali.ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=1,
                                         shuffle_after_epoch=True)

        self.decode = dali.ops.ImageDecoderRandomCrop(device='mixed', output_type=dali.types.RGB,
                                                      device_memory_padding=device_memory_padding,
                                                      host_memory_padding=host_memory_padding,
                                                      num_attempts=100)

        self.res = dali.ops.Resize(device='gpu', resize_x=crop, resize_y=crop, interp_type=dali.types.INTERP_TRIANGULAR)

        self.bri = dali.ops.Brightness(device="gpu")
        self.con = dali.ops.Contrast(device="gpu")
        self.sat = dali.ops.Saturation(device="gpu")

        self.cmnp = dali.ops.CropMirrorNormalize(device='gpu', output_dtype=dali.types.FLOAT,
                                                 output_layout=dali.types.NCHW,
                                                 crop=(crop, crop), image_type=dali.types.RGB,
                                                 mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                 std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.coin = dali.ops.CoinFlip(probability=0.5)
        self.uniform = dali.ops.Uniform(range=(max(0., 1 - color_jit), 1 + color_jit))

    def define_graph(self):
        imgs, labels = self.input(name='Reader')
        imgs = self.decode(imgs)
        imgs = self.res(imgs)
        imgs = self.bri(imgs, brightness=self.uniform())
        imgs = self.con(imgs, contrast=self.uniform())
        imgs = self.sat(imgs, saturation=self.uniform())
        imgs = self.cmnp(imgs, mirror=self.coin())
        return imgs, labels.gpu()


if __name__ == '__main__':
    pipe = TrainPipe('/media/piston/data/test', 1000, 4, 0, 224)
    pipe.build()

    train_loader = DALIClassificationIterator(pipe, 1281000, fill_last_batch=False)
    for i, data in enumerate(train_loader):
        imgs = data[0]["data"]
        labs = data[0]["label"].long()
        print(imgs.shape)
        # print(labs)
