import torch.nn as nn
from utils.utils import init_weights

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
def create_cls(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, test=False):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset,
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
