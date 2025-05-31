import torch
from torch import nn
from torchvision.models.vgg import vgg16_bn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.seq(inputs)

class NestedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetPlusPlus(nn.Module):
    def __init__(self, encoder_blocks, encoder_channels, n_cls, deep_supervision=False):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.depth = len(self.encoder_channels)
        self.deep_supervision = deep_supervision
        
        # Codificador
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        
        # Decodificador con conexiones densas (características principales de UNet++)
        self.decoders = nn.ModuleList([nn.ModuleList() for _ in range(self.depth)])
        self.ups = nn.ModuleList([nn.ModuleList() for _ in range(self.depth)])
        
        # Inicializar los bloques del decodificador y operaciones de upsampling
        for layer in range(1, self.depth):
            for depth in range(layer + 1):
                if depth == 0:
                    in_channels = self.encoder_channels[layer]
                    skip_channels = self.encoder_channels[layer - 1]
                    self.ups[layer][0] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    self.decoders[layer][0] = NestedBlock(in_channels + skip_channels, self.encoder_channels[layer - 1])
                else:
                    skip_connections = self.encoder_channels[layer - 1] * (depth + 1)
                    in_channels = self.encoder_channels[layer - 1]
                    self.ups[layer][depth] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                    self.decoders[layer][depth] = NestedBlock(in_channels + skip_connections, self.encoder_channels[layer - 1])
        
        # Clasificación final
        if self.deep_supervision:
            self.final_convs = nn.ModuleList()
            for i in range(self.depth - 1):
                self.final_convs.append(nn.Conv2d(self.encoder_channels[0], n_cls, kernel_size=1))
        else:
            self.final_conv = nn.Conv2d(self.encoder_channels[0], n_cls, kernel_size=1)

    def forward(self, x):
        # Codificador
        encoder_features = []
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            encoder_features.append(x)
        
        # Decodificador
        nested_features = {}
        # Para cada nivel de profundidad
        for layer in range(self.depth - 1, 0, -1):
            # Para cada bloque densamente conectado en ese nivel
            for depth in range(layer + 1):
                if depth == 0:
                    # X_0,i = Up(X_i,j) + X_i-1,j
                    decoder_feature = self.ups[layer][0](encoder_features[layer])
                    skip_feature = encoder_features[layer - 1]
                    decoder_feature = torch.cat([decoder_feature, skip_feature], dim=1)
                    decoder_feature = self.decoders[layer][0](decoder_feature)
                    nested_features[(layer, depth)] = decoder_feature
                else:
                    # X_d,i = Up(X_d-1,i) + [X_d-1,i-1, X_d-2,i-1, ..., X_0,i-1]
                    prev_decoder_feature = nested_features[(layer, depth - 1)]
                    decoder_feature = self.ups[layer][depth](prev_decoder_feature)
                    
                    # Concatenar todas las características del mismo nivel
                    skip_features = [nested_features[(layer - 1, d)] for d in range(depth + 1)]
                    skip_features = torch.cat(skip_features, dim=1)
                    
                    decoder_feature = torch.cat([decoder_feature, skip_features], dim=1)
                    decoder_feature = self.decoders[layer][depth](decoder_feature)
                    nested_features[(layer, depth)] = decoder_feature
        
        # Predicción final
        if self.deep_supervision:
            outputs = []
            for i in range(self.depth - 1):
                outputs.append(self.final_convs[i](nested_features[(i + 1, i + 1)]))
            return outputs
        else:
            # Usar la salida del último bloque para la predicción final
            output = self.final_conv(nested_features[(self.depth - 1, self.depth - 1)])
            return output

def _get_encoder_blocks(model):
    # igual que en tu código
    layers_last_module_names = ['5', '12', '22', '32', '42']
    result = []
    cur_block = nn.Sequential()
    for name, child in model.named_children():
        if name == 'features':
            for name2, child2 in child.named_children():
                cur_block.add_module(name2, child2)
                if name2 in layers_last_module_names:
                    result.append(cur_block)
                    cur_block = nn.Sequential()
            break
    return result

def construct_unetplusplus(n_cls, deep_supervision=False):
    model = vgg16_bn(weights='DEFAULT')
    encoder_blocks = _get_encoder_blocks(model)
    encoder_channels = [64, 128, 256, 512, 1024]  # vgg16 channels
    
    return UNetPlusPlus(encoder_blocks, encoder_channels, n_cls, deep_supervision)