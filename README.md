# Streamable Neural Audio Synthesis With Non-Causal Convolutions

Deep learning models are mostly used in an offline inference fashion. However, this strongly limits the use of these models inside audio generation setups, as most creative workflows are based on real-time digital signal processing. Although approaches based on recurrent networks can be naturally adapted to this buffer-based computation, the use of convolutions still poses some serious challenges. To tackle this issue, the use of _causal streaming convolutions_ have been proposed. However, this requires specific complexified training and can impact the resulting audio quality.

In this paper, we introduce a new method allowing to produce _non-causal streaming_ models. This allows to make any convolutional model compatible with real-time buffer-based processing. As our method is based on a post-training reconfiguration of the model, we show that it is able to transform models trained without causal constraints into a streaming model. We show how our method can be adapted to fit complex architectures with parallel branches. To evaluate our method, we apply it on the recent RAVE model, which provides high-quality real-time audio synthesis. We test our approach on multiple music and speech datasets and show that it is faster than overlap-add methods, while having no impact on the generation quality. Finally, we introduce two open-source implementation of our work as Max/MSP and PureData externals, and as a VST audio plugin. This allows to endow traditional digital audio workstations with real-time neural audio synthesis on any laptop CPU.

## Streamable RAVE for live audio processing

Applying our method on the RAVE model allows its use on realtime audio signals, on a wide range of platforms.

|                                                    RAVE x nn~                                                     |                                                   embedded RAVE                                                   |
| :---------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
| [![RAVE x nn~](http://img.youtube.com/vi/dMZs04TzxUI/mqdefault.jpg)](https://www.youtube.com/watch?v=dMZs04TzxUI) | [![RAVE x nn~](http://img.youtube.com/vi/jAIRf4nGgYI/mqdefault.jpg)](https://www.youtube.com/watch?v=jAIRf4nGgYI) |


## Building a Streamable Convolutional Neural Network

Let's define a simple autoencoder model

```python
import torch
import torch.nn as nn
import cached_conv as cc

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = cc.Sequential(
            cc.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            cc.Conv1d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            cc.Conv1d(16, 16, 3, stride=2, padding=1),
        )

        self.decoder = cc.Sequential(
            cc.ConvTranspose1d(16, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            cc.ConvTranspose1d(16, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            cc.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

Notice that we use convolutions defined by the `cached_conv` package instead of `torch.nn`. If we stop here, we get a model that behaves exactly as its `torch.nn` counterpart. However, if we enable cached convs and then instanciate the model

```python
import cached_conv as cc

cc.use_cached_conv(True)

model = AutoEncoder()
```

we now have a streamable model, i.e that can work on live streams ! We can now export it as a torchscript model

```python
model.register_buffer("forward_params", torch.tensor([1, 1, 1, 1]))
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "exported_model.ts")
```

And load it inside [nn~ for max/msp and PureData](https://github.com/acids-ircam/nn_tilde) for real-time neural audio processing ! Note that nn~ requires a `METHOD_params` buffer in the model for each exported method. It must be a tensor with 4 values:

- in channel number
- in sampling rate divider (1 = audio rate, 100 = audio rate / 100)
- out channel number
- out sampling rate divider

We can also export the encode method like this

```python
class AutoEncoder(nn.Module):
    @torch.jit.export
    def encode(self, x):
        return self.encoder(x)

...

model = AutoEncoder()
model.register_buffer("encode_params", torch.tensor([1, 1, 16, 8]))
```

## Realtime applications

### [nn~](https://github.com/acids-ircam/nn_tilde)

The **nn~** external for max/msp and PureData allows to interface pre-trained deep learning models in a graphical way, giving full control to the user on the different dimensions of input and output tensors.

![max_msp_screenshot](https://github.com/acids-ircam/RAVE/raw/master/docs/maxmsp_screenshot.png)


### [RAVE vst](https://github.com/acids-ircam/rave_vst)

The RAVE vst is a VST2/VST3/AU plugin designed to allow the use of the [RAVE model](https://github.com/acids-ircam/RAVE) inside regular digital audio workstations such as Ableton Live or Bitwig Studio.

![plugin_screenshot](https://github.com/acids-ircam/rave_vst/blob/main/assets/rave_screenshot_audio_panel.png?raw=true)

