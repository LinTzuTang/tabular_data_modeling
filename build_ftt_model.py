import torch
import rtdl

def build_ftt_model(
    n_num_features=12,
    d_token=192,  # must be multiple of 8
    n_blocks=1,
    attention_dropout=0.2,
    ffn_d_hidden=256,
    ffn_dropout=0.1,
    d_out=1,
    device="cpu",
    cardinalities=None,
):
    """build ftt model

    Args:
        n_num_features (int, optional): the number of numerical features. Defaults to 12.
        d_token (int, optional): the token size for each feature. Must be a multiple of :code:`n_heads=8`. Defaults to 192.
        attention_dropout (float, optional): the dropout for attention blocks. Defaults to 0.2.
        ffn_d_hidden (int, optional): : the *input* size for the *second* linear layer in `Transformer.FFN`. Defaults to 256.
        ffn_dropout (float, optional): the dropout rate after the first linear layer in `Transformer.FFN`. Defaults to 0.1.
        d_out (int, optional): output dim. Defaults to 1.
        device (str, optional): Defaults to "cpu".
        cardinalities (list, optional): use get_cardinalities_from_X in x_y_data_preprocessing.py to get cardinalities.json then load to list. Defaults to None (no categorical features)

    Returns:
        class: ftt model
    """
    cat_cardinalities = cardinalities or None
    model = rtdl.FTTransformer.make_baseline(
        n_num_features=n_num_features,
        cat_cardinalities=cardinalities,
        d_token=d_token,
        n_blocks=n_blocks,
        attention_dropout=attention_dropout,
        ffn_d_hidden=ffn_d_hidden,
        ffn_dropout=ffn_dropout,
        residual_dropout=0.0,
        d_out=1,
    )
    #model = rtdl.FTTransformer.make_default(
    #n_num_features=12,
    #cat_cardinalities=cardinalities,
    #last_layer_query_idx=[
    #    -1
    #],  # it makes the model faster and does NOT affect its output
    #d_out=1,
    #)
    device = torch.device(device)
    model.to(device)
    return model