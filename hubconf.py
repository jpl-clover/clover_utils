import torch
import clover_utils.finetune_model as finetune_model

# Further documentation:
# https://pytorch.org/docs/stable/hub.html

def finetuned_supervised_resnet():
    # The following docstring shows up in hub.help()
    """
    A resnet50 that was trained in a supervised manner on ImageNet, and has now been finetuned for MSL 2.1,
    in a fully supervised manner
    """
    # Where can we find the weights for this model? (Must be publically accessible, i.e.,
    # when you go to this link it immediately starts downloading without requiring permission)
    checkpoint_url = 'https://drive.google.com/uc?export=download&id=16ieQLL2053YLdofR4i_OJ3qQCLFJYAWz'
    
    # There are two possible approaches here, one is to load the state dict, construct
    # the model of interest, and load the state dict into that model like so
    # model = *build model from somewhere*
    # state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
    # model.load_state_dict(state_dict)

    # The other approach is to load the model directly from the checkpoint. This does require
    # that you model class' constructor save its own hyperparameters to the checkpoint. 
    # This can be achieved most easily by adding 'self.save_hyperparameters()' to the 
    # contructor and then training the model (an example of which is provided in finetune_model.py). 
    # The load then works like so:
    return finetune_model.Model.load_from_checkpoint(checkpoint_path=checkpoint_url)