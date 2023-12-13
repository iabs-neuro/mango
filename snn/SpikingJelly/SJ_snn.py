from spikingjelly.activation_based.model.train_classify import Trainer
from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet

import torch
class SResNetTrainer(Trainer):
    def preprocess_train_sample(self, args, x: torch.Tensor):
        # define how to process train sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        # define how to process test sample before send it to model
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate

    def get_args_parser(self, add_help=True):
        parser = super().get_args_parser()
        parser.add_argument('--T', type=int, help="total time-steps")
        #parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument('--cupy', type=bool, default=False, help="set the neurons to use cupy backend")
        return parser

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'

    def load_model(self, args, num_classes):
        if args.model in spiking_resnet.__all__:
            model = spiking_resnet.__dict__[args.model](pretrained=args.pretrained,
                                                        spiking_neuron=neuron.LIFNode,
                                                        surrogate_function=surrogate.ATan(),
                                                        detach_reset=True,
                                                        num_classes=10)

            functional.set_step_mode(model, step_mode='m')
            if args.cupy:
                functional.set_backend(model, 'cupy', neuron.LIFNode)

            return model
        else:
            raise ValueError(f"args.model should be one of {spiking_resnet.__all__}")