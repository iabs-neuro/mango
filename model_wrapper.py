from collections import OrderedDict
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms


TORCHHOME = torch.hub._get_torch_home()
LAYERS = {
    'alexnet':[
        'conv1', 'conv1_relu', 'pool1',
        'conv2', 'conv2_relu', 'pool2',
        'conv3', 'conv3_relu',
        'conv4', 'conv4_relu',
        'conv5', 'conv5_relu',
        'pool3',
        'dropout1', 'fc6', 'fc6_relu',
        'dropout2', 'fc7', 'fc7_relu',
        'fc8'
    ],
    'vgg16':[
        'conv1', 'conv1_relu',
        'conv2', 'conv2_relu', 'pool1',
        'conv3', 'conv3_relu',
        'conv4', 'conv4_relu', 'pool2',
        'conv5', 'conv5_relu',
        'conv6', 'conv6_relu',
        'conv7', 'conv7_relu', 'pool3',
        'conv8', 'conv8_relu',
        'conv9', 'conv9_relu',
        'conv10', 'conv10_relu', 'pool4',
        'conv11', 'conv11_relu',
        'conv12', 'conv12_relu',
        'conv13', 'conv13_relu', 'pool5',
        'fc1', 'fc1_relu', 'dropout1',
        'fc2', 'fc2_relu', 'dropout2',
        'fc3'],
    'densenet121':[
        'conv1',
        'bn1', 'bn1_relu', 'pool1',
        'denseblock1', 'transition1',
        'denseblock2', 'transition2',
        'denseblock3', 'transition3',
        'denseblock4',
        'bn2',
        'fc1'
    ]
}


class ModelWrapper:
    def __init__(self, model_name, imgpix=227, rawlayername=True, device="cpu"):
        self.imgpix = imgpix

        if isinstance(model_name, torch.nn.Module):
            self.model = model_name
            self.inputsize = (3, imgpix, imgpix)
            self.layername = None

        elif isinstance(model_name, str):
            if model_name == "vgg16":
                self.model = models.vgg16(pretrained=True)
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else LAYERS["vgg16"]
                self.inputsize = (3, imgpix, imgpix)

            elif model_name == "vgg16-face":
                self.model = models.vgg16(pretrained=False, num_classes=2622)
                self.model.load_state_dict(
                    torch.load(join(TORCHHOME, "vgg16_face.pt")))
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else LAYERS["vgg16"]
                self.inputsize = (3, imgpix, imgpix)

            elif model_name == "alexnet":
                self.model = models.alexnet(pretrained=True)
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else LAYERS[model_name]
                self.inputsize = (3, imgpix, imgpix)

            elif model_name == "densenet121":
                self.model = models.densenet121(pretrained=True)
                self.layers = list(self.model.features) + [self.model.classifier]
                self.layername = None if rawlayername else LAYERS[model_name]
                self.inputsize = (3, imgpix, imgpix)

            elif model_name == "densenet169":
                self.model = models.densenet169(pretrained=True)
                self.layername = None
                self.inputsize = (3, imgpix, imgpix)

            elif model_name == "resnet101":
                self.model = models.resnet101(pretrained=True)
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None

            elif "resnet50" in model_name:
                if "resnet50-face" in model_name:
                    self.model = models.resnet50(
                        pretrained=False, num_classes=8631)
                    if model_name == "resnet50-face_ft":
                        self.model.load_state_dict(
                            torch.load(
                                join(TORCHHOME, "resnet50_ft_weight.pt")))
                    elif model_name == "resnet50-face_scratch":
                        self.model.load_state_dict(
                            torch.load(
                                join(TORCHHOME, "resnet50_scratch_weight.pt")))
                    else:
                        raise NotImplementedError("Feasible names are resnet50-face_scratch, resnet50-face_ft")

                else:
                    self.model = models.resnet50(pretrained=True)

                    if model_name == "resnet50_linf_8":
                        self.model.load_state_dict(
                            torch.load(
                                join(TORCHHOME, "imagenet_linf_8_pure.pt")))

                    elif model_name == "resnet50_linf_4":
                        self.model.load_state_dict(
                            torch.load(
                                join(TORCHHOME, "imagenet_linf_4_pure.pt")))

                    elif model_name == "resnet50_l2_3_0":
                        self.model.load_state_dict(
                            torch.load(
                                join(TORCHHOME, "imagenet_l2_3_0_pure.pt")))

                self.inputsize = (3, imgpix, imgpix)
                self.layername = None

            elif model_name == "cornet_s":
                from cornet import cornet_s
                Cnet = cornet_s(pretrained=True)
                self.model = Cnet.module
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None

            else:
                raise NotImplementedError("Cannot find the specified model %s"%model_name)

        else:
            raise NotImplementedError("model_name need to be either string or nn.Module")

        self.model.to(device).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGBmean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).to(device)
        self.RGBstd = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).to(device)

        self.device = device
        self.hooks = []
        self.artiphys = False
        self.record_layers = []
        self.recordings = {}

        self.activation = {}

    def get_activation(self, name, unit=None, unitmask=None, ingraph=False):
        """
        :parameter
            name: key to retrieve the recorded activation `self.activation[name]`
            unit: a tuple of 3 element or single element. (chan, i, j ) or (chan)
            unitmask: used in population recording, it could be a binary mask of the same shape / length as the
                element number in feature tensor. Or it can be an array of integers.

            *Note*: when unit and unitmask are both None, this function will record the whole output feature tensor.

            ingraph: if True, then the recorded activation is still connected to input, so can pass grad.
                    if False then cannot
        :return
            hook:  Return a hook function that record the unit activity into the entry in activation dict of scorer.
        """
        if unit is None and unitmask is None:  # if no unit is given, output the full tensor.
            def hook(model, input, output):
                self.activation[name] = output if ingraph else output.detach()

        elif unitmask is not None:
            # has a unit mask, which could be an index list or a tensor mask same shape of the 3 dimensions.
            def hook(model, input, output):
                out = output if ingraph else output.detach()
                Bsize = out.shape[0]
                self.activation[name] = out.view([Bsize, -1])[:, unitmask.reshape(-1)]

        else:
            def hook(model, input, output):
                out = output if ingraph else output.detach()
                if len(output.shape) == 4:
                    self.activation[name] = out[:, unit[0], unit[1], unit[2]]
                elif len(output.shape) == 2:
                    self.activation[name] = out[:, unit[0]]

        return hook

    def set_unit(self, reckey, layer, unit=None, ingraph=False):
        if self.layername is not None:
            # if the network is a single stream feedforward structure, we can index it and use it to find the
            # activation
            idx = self.layername.index(layer)
            handle = self.layers[idx].register_forward_hook(self.get_activation(reckey, unit, ingraph=ingraph)) # we can get the layer by indexing
            self.hooks.append(handle)  # save the hooks in case we will remove it.
        else:
            # if not, we need to parse the architecture of the network.
            # indexing is not available, we need to register by recursively visit the layers and find match.
            handle, modulelist, moduletype = register_hook_by_module_names(layer, self.get_activation(reckey, unit, ingraph=ingraph),
                                self.model, self.inputsize, device="cpu")
            self.hooks.extend(handle)  # handle here is a list.
        return handle

    def select_unit(self, unit_tuple):
        self.layer = str(unit_tuple[1])
        self.chan = int(unit_tuple[2])
        if len(unit_tuple) == 5:
            self.unit_x = int(unit_tuple[3])
            self.unit_y = int(unit_tuple[4])
        else:
            self.unit_x = None
            self.unit_y = None
        self.set_unit("score", self.layer,
            (self.chan, self.unit_x, self.unit_y))

    def preprocess(self, img, input_scale=255):
        """preprocess single image array or a list (minibatch) of images
        This includes Normalize using RGB mean and std and resize image to (227, 227)
        """
        if type(img) is list: # the following lines have been optimized for speed locally.
            img_tsr = torch.stack(tuple(torch.from_numpy(im) for im in img)).to(self.device).float().permute(0, 3, 1, 2) / input_scale
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is torch.Tensor:
            img_tsr = (img.to(self.device) / input_scale - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is np.ndarray and img.ndim == 4:
            img_tsr = torch.tensor(img / input_scale).float().permute(0,3,1,2).to(self.device)
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is np.ndarray and img.ndim in [2, 3]:  # assume it's individual image
            img_tsr = transforms.ToTensor()(img / input_scale).float()
            img_tsr = self.normalize(img_tsr).unsqueeze(0)
            resz_out_img = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_img
        else:
            raise ValueError

    def score_tsr(self, img_tsr, with_grad=False, B=42, input_scale=1.0):
        """Score in batch will accelerate processing greatly!
        img_tsr is already torch.Tensor
        """
        # assume image is using 255 range
        imgn = img_tsr.shape[0]
        scores = np.zeros(img_tsr.shape[0])
        for layer in self.recordings:
            self.recordings[layer] = []
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(img_tsr[csr:csr_end,:,:,:], input_scale=input_scale)
            with torch.no_grad():
                self.model(img_batch.to(self.device))
            if "score" in self.activation: # if score is not there set trace to zero.
                scores[csr:csr_end] = self.activation["score"].squeeze().cpu().numpy().squeeze()

            if self.artiphys:  # record the whole layer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer]
                    self.recordings[layer].append(score_full.cpu().numpy())

            csr = csr_end

        for layer in self.recordings:
            self.recordings[layer] = np.concatenate(self.recordings[layer],axis=0)

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores


def named_apply(model, name, func, prefix=None):
    """ resemble the apply function but suits the functions here. """
    cprefix = "" if prefix is None else prefix + "." + name
    for cname, child in model.named_children():
        named_apply(child, cname, func, cprefix)

    func(model, name, "" if prefix is None else prefix)


def register_hook_by_module_names(target_name, target_hook, model, input_size=(3, 256, 256), device="cpu", ):
    module_names = OrderedDict()
    module_types = OrderedDict()
    target_hook_h = []
    def register_hook(module, name, prefix):
        # register forward hook and save the handle to the `hooks` for removal.
        def hook(module, input, output):
            # during forward pass, this hook will append the ReceptiveField information to `receptive_field`
            # if a module is called several times, this hook will append several times as well.
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_name = prefix + "." + class_name + name
            module_idx = len(module_names)
            module_names[str(module_idx)] = module_name
            module_types[str(module_idx)] = class_name
            if module_name == target_name:
                h = module.register_forward_hook(target_hook)
                target_hook_h.append(h)
        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                # and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    else:
        x = torch.rand(2, *input_size).type(dtype)

    # create properties
    module_names["0"] = "Image"
    module_types["0"] = "Input"
    hooks = []

    # register hook recursively at any module in the hierarchy
    named_apply(model, "", register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()
    if not(len(target_hook_h) == 1):
        print("Cannot hook the layer with the name %s\nAvailable names are listed here"%target_name)
        print("------------------------------------------------------------------------------")
        line_new = "{:>14}  {:>12}   {:>15} ".format("Layer Id", "Type", "ReadableStr", )
        print(line_new)
        print("==============================================================================")
        for layer in module_names:
            print("{:7} {:8} {:>12} {:>15}".format("", layer,
                module_types[layer], module_names[layer],))
        raise ValueError("Cannot hook the layer with the name %s\nAvailable names are listed here"%target_name)
    return target_hook_h, module_names, module_types
