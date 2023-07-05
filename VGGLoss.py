from torch import nn
from torchvision.models.vgg import vgg16




class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss,self).__init__()
        vgg = vgg16(pretrained=True)
        #our input images are 21x21 while vgg16 was trained on 224x244 which yielded a 7x7 feature map. so we only kept the layers upto which we have a 2x2 feature map so we can at least retain SOME spatially relevant information
        self.net = nn.Sequential(*list(vgg.features)[0:17]).eval()
        for param in self.net.parameters():
            param.requires_grad=False
        self.loss = nn.MSELoss()

    def forward(self, inputs, labels):

        #repeating Y channel across axis
        if(inputs.shape[1] == 1):
            inputs = inputs.repeat(1,3,1,1)
            labels = labels.repeat(1,3,1,1)
        
        input_features = self.net(inputs)
        label_features = self.net(labels)


        loss = self.loss(input_features, label_features) *0.006 #factor that srgan paper used
        return loss