import torch
import torch.nn as nn

class MotionDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, device, output_size=12, use_noise=None):
        super(MotionDiscriminator, self).__init__()
        self.device = device

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        lin2 = self.linear2(lin1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            # motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)

        # select the last valid, instead of: gru_o[-1, :, :]
        out = gru_o[tuple(torch.stack((lengths-1, torch.arange(bs, device=self.device))))]

        # dim (num_samples, 30)
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        return lin1

classifier_model_files = {
    "humanact12": "../actionrecognition/humanact12_gru.pth",
    "xia": "../actionrecognition/2000.tar",
}

def load_classifier(dataset_type, input_size_raw, num_classes, device):
    model = torch.load(classifier_model_files[dataset_type], map_location=device)
    classifier = MotionDiscriminator(input_size_raw, 128, 2, device=device, output_size=num_classes).to(device)
    # classifier.load_state_dict(model["model"])
    classifier.load_state_dict(model['motion_classifier'])
    classifier.eval()
    return classifier


def load_classifier_for_fid(dataset_type, input_size_raw, num_classes, device):
    model = torch.load(classifier_model_files[dataset_type], map_location=device)
    classifier = MotionDiscriminatorForFID(input_size_raw, 128, 2, device=device, output_size=num_classes).to(device)
    # classifier.load_state_dict(model["model"])
    classifier.load_state_dict(model['motion_classifier'])
    classifier.eval()
    return classifier
