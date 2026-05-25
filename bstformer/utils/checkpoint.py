import torch


def load_checkpoint(model, checkpoint, strict=False, map_location=None):
    state = torch.load(checkpoint, map_location=map_location)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if strict:
        model.load_state_dict(state)
        return

    model_state = model.state_dict()
    filtered_state = {}
    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            print(f"layer: {key} parameters size is not same!")
            continue
        filtered_state[key] = value
    model_state.update(filtered_state)
    model.load_state_dict(model_state, strict=False)
