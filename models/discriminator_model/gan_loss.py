import torch.nn
import torch.nn.functional
import dct


def texture_loss(minc_model, gen_output, gen_target):
    gen_features = minc_model(gen_output)
    real_features = minc_model(gen_target).detach()

    return torch.nn.functional.l1_loss(gen_features, real_features)


def pixel_loss(output, target):
    return torch.nn.functional.l1_loss(output, target)


def adversarial_loss(decision_real, decision_fake, device):
    # reverse labels
    label_real = torch.zeros_like(decision_real).to(device)
    label_fake = torch.ones_like(decision_fake).to(device)

    l_real = torch.nn.functional.binary_cross_entropy_with_logits(decision_real - decision_fake.mean(), label_real)
    l_fake = torch.nn.functional.binary_cross_entropy_with_logits(decision_fake - decision_real.mean(), label_fake)

    return (l_real + l_fake) / 2


def gan_loss(gen_spatial,
             real_spatial,
             decision_real,
             decision_fake,
             minc_model,
             pixel_weight,
             adversarial_weight,
             texture_weight,
             device):
    
    l_pix = pixel_loss(gen_spatial, real_spatial) * pixel_weight
    l_adv = adversarial_loss(decision_real, decision_fake, device) * adversarial_weight
    # l_tex = texture_loss(minc_model, gen_spatial, real_spatial) * texture_weight

    # return l_pix + l_tex + l_adv
    return l_pix + l_adv
