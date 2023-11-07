import os
import torch
from os.path import join as pjoin
import shutil

class Config:

    for_try = False

    use_rotloss = True
    use_newdecoder = True

    # model paths
    main_dir = None
    model_dir = None
    tb_dir = None
    info_dir = None
    output_dir = None

    weight_decay = 0.0001          # weight decay
    lr_gen = 0.0001                # learning rate for the generator
    weight_init = 'gaussian'
    lr_policy = None

    rot_channels = 128  # added one more y-axis rotation
    pos3d_channels = 64  # changed to be the same as rfree
    proj_channels = 42

    num_channel = rot_channels
    num_style_joints = 21

    style_channel_2d = proj_channels
    style_channel_3d = 63   # kit_ml,xia,  joints position

    """
    encoder for class
    [down_n] stride=[enc_cl_stride], dim=[enc_cl_channels] convs, 
    followed by [enc_cl_global_pool]

    """
    enc_cl_down_n = 2  # 64 -> 32 -> 16 -> 8 -> 4
    enc_cl_channels = [0, 96, 144]
    enc_cl_kernel_size = 8
    enc_cl_stride = 2

    """
    encoder for content
    [down_n] stride=[enc_co_stride], dim=[enc_co_channels] convs (with IN)
    followed by [enc_co_resblks] resblks with IN
    """
    enc_co_down_n = 1
    enc_co_channels = [63, 144]
    enc_co_kernel_size = 8
    enc_co_stride = 2
    enc_co_resblks = 1


    """
    mlp
    map from class output [enc_cl_channels[-1] * 1]
    to AdaIN params (dim calculated at runtime)
    """
    mlp_dims = [enc_cl_channels[-1], 192, 256]

    """
    decoder
    [dec_resblks] resblks with AdaIN
    [dec_up_n] Upsampling followed by stride=[dec_stride] convs
    """

    dec_bt_channel = 144
    dec_resblks = enc_co_resblks
    dec_channels = enc_co_channels.copy()
    dec_channels.reverse()
    dec_up_n = enc_co_down_n
    dec_kernel_size = 8
    dec_stride = 1


    def initialize(self, args=None, save=True):

        if hasattr(args, 'name') and args.name is not None:
            print("args.name= ", args.name)
            self.name = args.name

        if hasattr(args, 'batch_size') and args.name is not None:
            self.batch_size = args.batch_size

        self.main_dir = os.path.join(self.expr_dir, self.name)
        self.model_dir = os.path.join(self.main_dir, "pth")
        self.tb_dir = os.path.join(self.main_dir, "log")
        self.info_dir = os.path.join(self.main_dir, "info")
        self.output_dir = os.path.join(self.main_dir, "output")

        self.device = torch.device("cuda:%d" % self.cuda_id if torch.cuda.is_available() else "cpu")

        if save:
            self.config_name = args.config
            cfg_file = "%s.py" % self.config_name
            shutil.copy(pjoin(
                "F:/LZY/test01",
                cfg_file), os.path.join(self.info_dir, cfg_file))

