import pickle
import torch
import zmq
from superpoint import SuperPoint
import numpy as np

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:5555')
    
    extractor_model = SuperPoint({
        'descriptor_dim': 256,
        'nms_radius': 3,
        'max_keypoints': 4096,
        'keypoints_threshold': 0.6
    })
    extractor_model.cuda()
    extractor_model.eval()
    extractor_model.load_state_dict(torch.load('data/models/superpoint_v1.pth'), strict=True)
    while True:
        gray = pickle.loads(socket.recv())
        print('processing')
        with torch.no_grad():
            res = extractor_model(torch.from_numpy(gray / 255.).float()[None, None].cuda())
            
        res = {
            'keypoints': res['keypoints'][0].cpu().numpy(),
            'descriptors': res['descriptors'][0].cpu().numpy().T,
            'raw_descs': np.moveaxis(res['raw_descs'][0].cpu().numpy(), 0, -1)
        }
        socket.send(pickle.dumps(res))
        