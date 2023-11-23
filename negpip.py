import torch
import copy
from comfy.sd1_clip import gen_empty_tokens

class Negpip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, clip):
        new_clip = copy.copy(clip)
        if hasattr(new_clip.cond_stage_model, "clip_g"):
            new_clip.cond_stage_model.clip_g.encode_token_weights = hook_clip_encode_token_weights(new_clip.cond_stage_model.clip_g)
        if hasattr(new_clip.cond_stage_model, "clip_h"):
            new_clip.cond_stage_model.clip_h.encode_token_weights = hook_clip_encode_token_weights(new_clip.cond_stage_model.clip_h)
        if hasattr(new_clip.cond_stage_model, "clip_l"):
            new_clip.cond_stage_model.clip_l.encode_token_weights = hook_clip_encode_token_weights(new_clip.cond_stage_model.clip_l)
        new_model = model.clone()

        def negpip_apply(q, k, v, extra_options):
            new_k = k[:, 0::2]
            new_v = v[:, 1::2]
            return q, new_k, new_v

        new_model.set_model_attn2_patch(negpip_apply)

        return new_model, new_clip

# prompt weightingの計算で無理やりk,vを計算
# k,vはattn2_patchで分離する
# weightがマイナスのときはvだけマイナスにする
def hook_clip_encode_token_weights(self):
    
    def encode_token_weights(token_weight_pairs):
        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            to_encode.append(gen_empty_tokens(self.special_tokens, max_token_len))

        out, pooled = self.encode(to_encode)
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            zk = out[k:k+1].clone()
            zv = out[k:k+1].clone()
            if has_weights:
                z_empty = out[-1]
                for i in range(len(zk)):
                    for j in range(len(zk[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight < 0:
                            weight = -weight
                            sign = -1
                        else:
                            sign = 1
                        zk[i][j] = (zk[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
                        zv[i][j] = sign * ((zv[i][j] - z_empty[0][j]) * weight + z_empty[0][j])

            z = torch.zeros_like(zk).repeat(1, 2, 1)
            for i in range(zk.shape[1]):  # 頭悪いのでfor文
                z[:, 2*i, :] += zk[:, i, :]
                z[:, 2*i+1, :] += zv[:, i, :]
            output.append(z)

        if (len(output) == 0):
            return out[-1:].cpu(), first_pooled
        return torch.cat(output, dim=-2).cpu(), first_pooled

    return encode_token_weights


NODE_CLASS_MAPPINGS = {
    "Negpip": Negpip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Negpip": "Apply Negpip",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
