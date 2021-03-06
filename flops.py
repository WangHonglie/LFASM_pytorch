import sys
import argparse

import torchvision.models as models
import torch

from ptflops import get_model_complexity_info
from model import SelfMatchNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='resnet18')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    net = SelfMatchNet(776)

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    flops, params = get_model_complexity_info(net, (3, 234, 234),
                                              as_strings=True,
                                              print_per_layer_stat=True,
                                              ost=ost)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
