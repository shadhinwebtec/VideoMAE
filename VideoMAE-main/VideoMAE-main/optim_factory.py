import tensorflow as tf
import json

def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner:
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for var in model.trainable_variables:
        name = var.name
        if len(var.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f"layer_{layer_id}_{group_name}"
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(var)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return parameter_group_vars

import tensorflow_addons as tfa

def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    if weight_decay and filter_bias_and_bn:
        skip = set()
        if skip_list is not None:
            skip = set(skip_list)
        elif hasattr(model, 'no_weight_decay'):
            skip = set(model.no_weight_decay())
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = {"default": {"params": model.trainable_variables}}

    opt_args = {"learning_rate": args.lr}
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['epsilon'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['beta_1'] = args.opt_betas[0]
        opt_args['beta_2'] = args.opt_betas[1]

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    
    if opt_lower == 'sgd':
        optimizer = tf.keras.optimizers.SGD(**opt_args, momentum=args.momentum, nesterov=True)
    elif opt_lower == 'momentum':
        optimizer = tf.keras.optimizers.SGD(**opt_args, momentum=args.momentum, nesterov=False)
    elif opt_lower == 'adam':
        optimizer = tf.keras.optimizers.Adam(**opt_args)
    elif opt_lower == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(**opt_args, weight_decay=weight_decay)
    elif opt_lower == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(**opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(**opt_args, rho=0.9, momentum=args.momentum)
    elif opt_lower == 'radam':
        optimizer = tfa.optimizers.RectifiedAdam(**opt_args)
    elif opt_lower == 'novograd':
        optimizer = tfa.optimizers.NovoGrad(**opt_args)
    elif opt_lower == 'lamb':
        optimizer = tfa.optimizers.LAMB(**opt_args)
    elif opt_lower == 'lookahead':
        base_optimizer = tf.keras.optimizers.Adam(**opt_args)
        optimizer = tfa.optimizers.Lookahead(base_optimizer)
    else:
        raise ValueError(f"Invalid optimizer: {opt_lower}")

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = tfa.optimizers.Lookahead(optimizer)

    return optimizer


class Args:
    def __init__(self, opt='adam', lr=1e-3, weight_decay=1e-5, momentum=0.9, opt_eps=None, opt_betas=None):
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.opt_eps = opt_eps
        self.opt_betas = opt_betas

args = Args()
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

optimizer = create_optimizer(args, model)

