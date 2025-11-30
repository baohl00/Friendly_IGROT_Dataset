import torch
from bidirectional_cross_attention import BidirectionalCrossAttention

video = torch.randn(32, 1, 512)
audio = torch.randn(32, 1, 512)

video_mask = torch.ones((1, 1)).bool()
audio_mask = torch.ones((1, 1)).bool()

joint_cross_attn = BidirectionalCrossAttention(
            dim = 512,
            heads = 8,
            dim_head = 64,
            context_dim = 512
            )

video_out, audio_out = joint_cross_attn(
            video,
            audio,
            mask = video_mask,
            context_mask = audio_mask
            )

# attended output should have the same shape as input

assert video_out.shape == video.shape
assert audio_out.shape == audio.shape
print(video_out.shape)


image = torch.randn(32, 2) 
a = torch.randn(32, 512)
b = torch.randn(32, 512) 
output = image[:, 0:1] * a + image[:, 1:2] * b
print(output.shape)
