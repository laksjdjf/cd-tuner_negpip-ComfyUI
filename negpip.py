import torch
import copy


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
            new_clip.cond_stage_model.clip_l.encode_token_weights = hook_clip_encode_token_weights(new_clip.cond_stage_model.clip_l)
        else:
            new_clip.cond_stage_model.encode_token_weights = hook_clip_encode_token_weights(new_clip.cond_stage_model)
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
        to_encode = list(self.empty_tokens)
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            to_encode.append(tokens)

        out, pooled = self.encode(to_encode)
        z_empty = out[0:1]
        if pooled.shape[0] > 1:
            first_pooled = pooled[1:2]
        else:
            first_pooled = pooled[0:1]

        output = []

        for k in range(1, out.shape[0]):
            zk = out[k:k+1].clone()
            zv = out[k:k+1].clone()
            for i in range(len(zk)):
                for j in range(len(zk[i])):
                    weight = token_weight_pairs[k - 1][j][1]
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
            return z_empty.cpu(), first_pooled.cpu()
        return torch.cat(output, dim=-2).cpu(), first_pooled.cpu()

    return encode_token_weights


NODE_CLASS_MAPPINGS = {
    "Negpip": Negpip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Negpip": "Apply Negpip",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
